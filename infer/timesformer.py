"""TimeSformer ink-detection model (``scrollprize/timesformer_scroll5_july_retreat``).

This checkpoint ships as a bare PyTorch-Lightning ``.ckpt`` with no
``config.json``, so the architecture has to be reconstructed from the repo code
(``villa/ink-detection/optimized_inference/model_timesformer.py``) and pinned
against the checkpoint's own tensor shapes.

Two things the weights do NOT tell us
-------------------------------------
The backbone uses **rotary** position embeddings
(``frame_rot_emb.inv_freqs``, ``image_rot_emb.scales``), which are generated on
the fly for whatever sequence length arrives. So:

* the **number of z-layers** (frames) is not encoded in the weights, and
* the checkpoint loads identically for any ``num_frames``.

The layer window is a runtime parameter and cannot be recovered from the
checkpoint. The reference inference source records 26 frames starting at legacy
layer 17; ``calibrate.py`` maps that to the local render and checks it against a
published prediction.

What the weights DO pin
-----------------------
``to_patch_embedding.weight`` is ``(512, 256)`` with ``256 = 16 * 16 * 1``,
fixing ``patch_size=16`` and ``channels=1``. ``to_out.1.weight`` is
``(16, 512)``, fixing ``num_classes=16``, which the wrapper reshapes to a 4x4
logit map. With ``image_size=64`` that is a 16x upsample to the 64x64 tile.
``hyper_parameters`` in the checkpoint records ``size: 64`` and
``with_norm: False``.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

# Shapes pinned by the checkpoint's own tensors.
DIM = 512
IMAGE_SIZE = 64
PATCH_SIZE = 16
NUM_CLASSES = 16  # -> 4x4 logit map
DEPTH = 8
HEADS = 6
DIM_HEAD = 64


class TimesformerInkModel(nn.Module):
    """Matches ``RegressionPLModel`` from the repo, minus the Lightning parts.

    ``with_norm`` is False for this checkpoint (recorded in its
    ``hyper_parameters``, and confirmed by the absence of any
    ``normalization.*`` tensor), so the optional ``BatchNorm3d`` is omitted.
    """

    def __init__(self, num_frames: int):
        super().__init__()
        from timesformer_pytorch import TimeSformer

        self.num_frames = int(num_frames)
        self.backbone = TimeSformer(
            dim=DIM,
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            num_frames=self.num_frames,
            num_classes=NUM_CLASSES,
            channels=1,
            depth=DEPTH,
            heads=HEADS,
            dim_head=DIM_HEAD,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, D, H, W)`` -> logits ``(B, 1, 4, 4)``.

        The library wants ``(B, frames, channels, H, W)``, so the singleton
        channel axis and the depth axis are swapped, exactly as the repo does.
        """
        if x.ndim == 4:
            x = x[:, None]
        if x.ndim != 5:
            raise ValueError(
                f"expected 4-D (B, D, H, W) or 5-D (B, 1, D, H, W); got {tuple(x.shape)}"
            )
        x = torch.permute(x, (0, 2, 1, 3, 4))  # (B, D, 1, H, W)
        x = self.backbone(x)                   # (B, 16)
        return x.view(-1, 1, 4, 4)


def _safe_load(path: Path) -> dict:
    """``torch.load`` with ``weights_only=True`` and a minimal allowlist.

    The checkpoint pickles an optimizer and an LR scheduler alongside the
    weights. Rather than dropping to ``weights_only=False``, which would permit
    arbitrary code execution from a downloaded file, we allow exactly the
    benign container/optimizer types it needs.
    """
    from torch.optim.adamw import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    allow = [CosineAnnealingLR, AdamW, collections.defaultdict, dict]
    with torch.serialization.safe_globals(allow):
        return torch.load(str(path), map_location="cpu", weights_only=True)


def find_checkpoint(snapshot_dir: str | Path) -> Path:
    matches = sorted(Path(snapshot_dir).glob("*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"no .ckpt under {snapshot_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous: {len(matches)} .ckpt files under {snapshot_dir}")
    return matches[0]


def load_timesformer(
    snapshot_dir: str | Path, num_frames: int
) -> Tuple[TimesformerInkModel, dict]:
    """Build the model and load the checkpoint.

    ``num_frames`` must be supplied by the caller: it is the z-window size and
    the weights do not constrain it. Raises on any key mismatch.
    """
    ckpt_path = find_checkpoint(snapshot_dir)
    raw = _safe_load(ckpt_path)
    state = raw["state_dict"]

    hparams = raw.get("hyper_parameters", {})
    if hparams.get("with_norm"):
        raise NotImplementedError(
            "checkpoint was trained with_norm=True; this loader omits the BatchNorm3d"
        )
    if "size" in hparams and int(hparams["size"]) != IMAGE_SIZE:
        raise ValueError(f"checkpoint size={hparams['size']}, expected {IMAGE_SIZE}")

    model = TimesformerInkModel(num_frames=num_frames)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match architecture: "
            f"{len(missing)} missing {list(missing)[:5]}, "
            f"{len(unexpected)} unexpected {list(unexpected)[:5]}"
        )

    model.eval()
    config = {
        "model_kind": "timesformer",
        "input_depth": int(num_frames),
        "input_size": IMAGE_SIZE,
        "in_channels": 1,
        "output_grid": 4,
        "checkpoint": str(ckpt_path),
        "global_step": raw.get("global_step"),
        "epoch": raw.get("epoch"),
    }
    return model, config
