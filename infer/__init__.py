"""Ink-detection inference engine for the op6 causal experiment (villa#1372).

Typical use::

    from infer import InkEngine
    engine = InkEngine()                       # loads the checkpoint once
    ink = engine.predict(surface_volume)       # (H, W, 62) -> (H, W) float32

``surface_volume`` is ``(H, W, D)`` with depth LAST.

Two model families are supported::

    InkEngine(model_kind="resnet3d")                    # D == 62, tile 256
    InkEngine(model_kind="timesformer", num_frames=33)  # D free, tile 64

``num_frames`` is mandatory for the timesformer because its rotary position
embeddings leave the z-window unconstrained. The reference inference source records
26 frames starting at legacy layer 17; callers still pass it explicitly here.
"""

from ._version import __version__

from .engine import (
    DEFAULT_CHECKPOINT,
    TIMESFORMER_CHECKPOINT,
    EngineConfig,
    InkEngine,
    PredictStats,
    grid_1d,
    hann2d,
    predict,
    read_roi,
    uniform2d,
)
from .model import InkDetectionModel, MaxPool2dOverDepth, load_model
from .preprocess import CLIP_MAX, CLIP_MIN, NORMALIZE_DIVISOR, preprocess
from .timesformer import TimesformerInkModel, load_timesformer

__all__ = [
    "__version__",
    "CLIP_MAX",
    "CLIP_MIN",
    "DEFAULT_CHECKPOINT",
    "EngineConfig",
    "InkDetectionModel",
    "InkEngine",
    "MaxPool2dOverDepth",
    "NORMALIZE_DIVISOR",
    "PredictStats",
    "TIMESFORMER_CHECKPOINT",
    "TimesformerInkModel",
    "grid_1d",
    "hann2d",
    "load_model",
    "load_timesformer",
    "predict",
    "preprocess",
    "read_roi",
    "uniform2d",
]
