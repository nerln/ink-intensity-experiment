from __future__ import annotations
import os
"""Sliding-window ink-detection inference engine.

Takes a rendered surface-volume ROI and returns a per-pixel ink probability map
at the ROI's own resolution. Built for a CUDA-free Apple-silicon host.

Input contract
--------------
``surface_volume`` is ``(H, W, D)`` — depth LAST — with ``D == 62`` for the
``PHerc.1667-iteration-*`` checkpoints. Raw intensities, normally ``uint8``.
This axis order matches both references: ``LayersSource`` in
``optimized_inference/inference.py`` documents "(H, W, C)", and the model card's
tiling example annotates ``image: (H, W, D) uint8 stack of the 62 layers``.

Per tile the engine reproduces the training-time pipeline exactly:

    read (tile, tile, D) with zero-padding out of bounds
    -> optional reverse along depth
    -> valid = any(raw != 0, axis=-1)          (computed BEFORE the clip)
    -> clip to [0, 200], divide by 255         (see preprocess.py)
    -> (1, 1, D, tile, tile) float32
    -> logits (1, 1, tile/4, tile/4)
    -> sigmoid
    -> bilinear upsample to (tile, tile)
    -> accumulate with a blending weight, masked by `valid`

then divides the accumulator by the accumulated weight.

Determinism
-----------
``predict`` is bit-exact across calls: eval mode (BatchNorm uses running stats),
``torch.inference_mode``, a fixed sequential tile order, no autocast, and no
dropout. Verified by ``tests/test_determinism.py``.
"""


import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import InkDetectionModel, load_model
from .preprocess import preprocess

# Where the 1.7 GB of checkpoints live. Relative to the repository so a clean checkout
# works; overridable because they do not belong on a small system disk.
_WEIGHTS = os.environ.get("OP6_WEIGHTS") or str(Path(__file__).resolve().parent.parent / "weights")


DEFAULT_CHECKPOINT = (
    f"{_WEIGHTS}/hf/hub/"
    "models--scrollprize--PHerc.1667-iteration-0/snapshots/"
    "06c306449b39df42c745608ceade243498d24243"
)

TIMESFORMER_CHECKPOINT = (
    f"{_WEIGHTS}/hf/hub/"
    "models--scrollprize--timesformer_scroll5_july_retreat/snapshots/"
    "5b714296b256b8f993ce69e8e57aea585125d782"
)


# --------------------------------------------------------------------------
# Tiling helpers (ported from optimized_inference/inference.py)
# --------------------------------------------------------------------------
def grid_1d(length: int, tile: int, stride: int) -> List[int]:
    """1-D tile origins covering ``[0, length)``, last tile forced to the border."""
    xs = list(range(0, max(1, length - tile + 1), stride))
    end = max(0, length - tile)
    if not xs or xs[-1] != end:
        xs.append(end)
    return xs


def hann2d(h: int, w: int) -> np.ndarray:
    """Normalised 2-D Hann window (sum = 1), as used by ``optimized_inference``."""
    k = np.outer(np.hanning(h).astype(np.float32), np.hanning(w).astype(np.float32))
    s = k.sum()
    return (k / (s if s > 0 else 1.0)).astype(np.float32)


def uniform2d(h: int, w: int) -> np.ndarray:
    """Flat window, as used by the model card's tiling example."""
    return np.full((h, w), 1.0 / float(h * w), dtype=np.float32)


def read_roi(volume: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    """Read ``(y2-y1, x2-x1, D)``, zero-padding anything out of bounds."""
    h, w, _ = volume.shape
    yy1, yy2 = max(0, y1), min(h, y2)
    xx1, xx2 = max(0, x1), min(w, x2)
    out = np.zeros((y2 - y1, x2 - x1, volume.shape[2]), dtype=volume.dtype)
    if yy2 > yy1 and xx2 > xx1:
        out[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)] = volume[yy1:yy2, xx1:xx2, :]
    return out


def pick_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class EngineConfig:
    #: "resnet3d" -> PHerc.1667-iteration-* (62 layers, 256 tile)
    #: "timesformer" -> timesformer_scroll5_* (layer count is a free parameter)
    model_kind: Literal["resnet3d", "timesformer"] = "resnet3d"
    #: Defaults to the checkpoint matching ``model_kind`` when left None.
    checkpoint_dir: Optional[str] = None
    #: Number of z-layers for the timesformer, whose rotary embeddings do not
    #: constrain it. Ignored (and must be None) for resnet3d, where the config
    #: pins it at 62.
    num_frames: Optional[int] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    #: None -> take the checkpoint's trained input size.
    tile: Optional[int] = None
    #: None -> tile // 2.
    stride: Optional[int] = None
    batch_size: int = 1
    blend: Literal["hann", "uniform"] = "hann"
    #: Reverse the depth axis before inference. The model card notes that l_2
    #: inference uses reverse_layers=True "to match the training-segment
    #: convention". Whether a given rendered ROI needs it depends on the
    #: renderer's normal orientation, so this is left explicit rather than
    #: guessed. Both arms of a causal comparison must use the same value.
    reverse_layers: bool = False


@dataclass
class PredictStats:
    tiles: int = 0
    batches: int = 0
    seconds_total: float = 0.0
    seconds_forward: float = 0.0
    per_tile_seconds: float = 0.0
    device: str = ""
    dtype: str = ""
    shape_in: Tuple[int, int, int] = (0, 0, 0)
    shape_out: Tuple[int, int] = (0, 0)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class InkEngine:
    """Loads the checkpoint once, then runs many ROIs through it."""

    def __init__(self, config: Optional[EngineConfig] = None, **overrides):
        self.config = config or EngineConfig(**overrides)
        cfg = self.config
        self.device = pick_device(cfg.device)

        if cfg.model_kind == "resnet3d":
            if cfg.num_frames is not None:
                raise ValueError(
                    "num_frames does not apply to resnet3d: its config pins input_depth=62"
                )
            cfg.checkpoint_dir = cfg.checkpoint_dir or DEFAULT_CHECKPOINT
            self.model, self.model_config = load_model(cfg.checkpoint_dir)
        elif cfg.model_kind == "timesformer":
            if cfg.num_frames is None:
                raise ValueError(
                    "timesformer needs an explicit num_frames: its rotary embeddings do "
                    "not constrain the z-window, and the repo records no default. "
                    "Establish it against a published prediction before trusting a run."
                )
            from .timesformer import load_timesformer

            cfg.checkpoint_dir = cfg.checkpoint_dir or TIMESFORMER_CHECKPOINT
            self.model, self.model_config = load_timesformer(
                cfg.checkpoint_dir, num_frames=cfg.num_frames
            )
        else:
            raise ValueError(f"unknown model_kind={cfg.model_kind!r}")

        self.model = self.model.to(self.device, dtype=cfg.dtype)
        self.model.eval()

        self.input_depth: int = int(self.model_config["input_depth"])
        self.input_size: int = int(self.model_config["input_size"])

        if cfg.tile is None:
            cfg.tile = self.input_size
        if cfg.stride is None:
            cfg.stride = cfg.tile // 2
        self.last_stats = PredictStats()

        if self.config.tile != self.input_size:
            raise ValueError(
                f"tile={self.config.tile} but the checkpoint was trained at "
                f"input_size={self.input_size}. The network is fully convolutional so "
                "another size would run, but it would not be the trained operating "
                "point. Pass tile explicitly only if you mean it."
            )

    # ---------------------------------------------------------------- public
    def predict(self, surface_volume: np.ndarray) -> np.ndarray:
        """Run tiled inference over an ROI.

        Args:
            surface_volume: ``(H, W, D)`` raw intensities, ``D == input_depth``.

        Returns:
            ``(H, W)`` float32 ink probability in ``[0, 1]``. Pixels no tile
            could vote on (fully-zero columns, or zero accumulated blend
            weight) come back as 0.0.
        """
        vol = self._validate(surface_volume)
        cfg = self.config
        h, w, _ = vol.shape

        t_start = time.perf_counter()
        t_forward = 0.0

        weight = (hann2d if cfg.blend == "hann" else uniform2d)(cfg.tile, cfg.tile)
        weight_t = torch.from_numpy(weight).to(self.device)

        origins = [
            (x1, y1)
            for y1 in grid_1d(h, cfg.tile, cfg.stride)
            for x1 in grid_1d(w, cfg.tile, cfg.stride)
        ]

        acc_pred = np.zeros((h, w), dtype=np.float32)
        acc_weight = np.zeros((h, w), dtype=np.float32)

        batches = 0
        with torch.inference_mode():
            for start in range(0, len(origins), cfg.batch_size):
                chunk = origins[start : start + cfg.batch_size]
                tensors, valids, coords = [], [], []

                for x1, y1 in chunk:
                    x2, y2 = x1 + cfg.tile, y1 + cfg.tile
                    raw = read_roi(vol, y1, y2, x1, x2)          # (t, t, D)
                    if cfg.reverse_layers:
                        raw = raw[:, :, ::-1]
                    # Validity is read off the RAW tile, before the clip, so it
                    # reflects true surface coverage rather than intensity.
                    valid = np.any(raw != 0, axis=-1).astype(np.float32)
                    scaled = preprocess(raw)                      # (t, t, D) float32
                    # (t, t, D) -> (D, t, t) -> (1, D, t, t)
                    tensors.append(torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1))).unsqueeze(0))
                    valids.append(valid)
                    coords.append((x1, y1, x2, y2))

                # (B, 1, D, t, t)
                batch = torch.stack(tensors, dim=0).to(self.device, dtype=cfg.dtype)

                t0 = time.perf_counter()
                logits = self.model(batch)                       # (B, 1, t/4, t/4)
                probs = torch.sigmoid(logits.float())
                probs = F.interpolate(
                    probs, size=(cfg.tile, cfg.tile), mode="bilinear", align_corners=False
                )                                                # (B, 1, t, t)
                weighted = (probs * weight_t).squeeze(1)         # (B, t, t)
                weighted_np = weighted.to("cpu").numpy()
                self._sync()
                t_forward += time.perf_counter() - t0
                batches += 1

                for i, (x1, y1, x2, y2) in enumerate(coords):
                    v = valids[i]
                    # Clip destination to the ROI; tiles never exceed it because
                    # grid_1d caps origins at length - tile, but a ROI smaller
                    # than one tile does overhang.
                    dy, dx = min(y2, h) - y1, min(x2, w) - x1
                    acc_pred[y1 : y1 + dy, x1 : x1 + dx] += (weighted_np[i] * v)[:dy, :dx]
                    acc_weight[y1 : y1 + dy, x1 : x1 + dx] += (weight * v)[:dy, :dx]

        out = np.divide(
            acc_pred, acc_weight, out=np.zeros_like(acc_pred), where=acc_weight > 0.0
        )
        np.clip(out, 0.0, 1.0, out=out)

        total = time.perf_counter() - t_start
        self.last_stats = PredictStats(
            tiles=len(origins),
            batches=batches,
            seconds_total=total,
            seconds_forward=t_forward,
            per_tile_seconds=t_forward / max(1, len(origins)),
            device=str(self.device),
            dtype=str(cfg.dtype),
            shape_in=tuple(vol.shape),
            shape_out=(h, w),
        )
        return out

    # --------------------------------------------------------------- private
    def _validate(self, volume: np.ndarray) -> np.ndarray:
        if not isinstance(volume, np.ndarray):
            raise TypeError(f"surface_volume must be np.ndarray, got {type(volume).__name__}")
        if volume.ndim != 3:
            raise ValueError(
                f"surface_volume must be 3-D (H, W, D) with depth last; got shape {volume.shape}"
            )
        h, w, d = volume.shape
        if d != self.input_depth:
            raise ValueError(
                f"surface_volume has D={d} but the checkpoint expects D={self.input_depth}. "
                "Axis order is (H, W, D) with depth LAST — if your renderer emits "
                f"({d}, {h}, {w}) transpose it before calling predict()."
            )
        if h < 1 or w < 1:
            raise ValueError(f"degenerate ROI: {volume.shape}")
        return volume

    def _sync(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Convenience one-shot API
# --------------------------------------------------------------------------
_CACHED: dict = {}


def predict(surface_volume: np.ndarray, **overrides) -> np.ndarray:
    """One-shot ``predict``. Caches the engine per distinct configuration.

    For a real experiment prefer constructing :class:`InkEngine` once and
    reusing it, so checkpoint load time is not charged to each ROI.
    """
    key = tuple(sorted((k, str(v)) for k, v in overrides.items()))
    if key not in _CACHED:
        _CACHED[key] = InkEngine(**overrides)
    return _CACHED[key].predict(surface_volume)
