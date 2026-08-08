"""The input contract: clip to a fixed absolute range, then scale to [0, 1].

This is the single most falsifiable part of the pipeline, so it lives in its own
module with its two constants named and sourced.

Training (``villa/ink-detection``) does exactly two things to a raw uint8 tile:

    image = np.clip(image, 0, 200)                              # 7 files agree
    A.Normalize(mean=[0] * in_chans, std=[1] * in_chans)        # -> image / 255

``A.Normalize`` computes ``(img - mean * max_pixel_value) / (std * max_pixel_value)``.
With ``mean=0`` and ``std=1`` this degenerates to a plain division by
albumentations' default ``max_pixel_value=255.0``. So despite the API name,
*no* mean-subtraction or standard-deviation scaling happens: the tensor the
network saw in training is ``clip(x, 0, 200) / 255``, spanning [0, 0.784].

Two published statements conflict with this and are wrong (see REPORT notes):

* ``modeling_inkdetection.py`` docstring says "intensity already z-score
  normalised". That is a loose description of the ``A.Normalize`` call above,
  which does not z-score anything. The model card's own Quick Start spells out
  the real operation ("clipped raw uint8 layers to [0, 200] then applied
  Normalize(mean=0, std=1)"), and agrees with the training code.
* The ``optimized_inference`` branch divided by the clip value (200) instead of
  255, feeding the network inputs 1.275x larger than anything it saw in
  training. Fixed upstream in PR #1371.

We deliberately implement the *training* normalisation, because the experiment
asks what the model sees when it sees what it was trained on.
"""

from __future__ import annotations

import numpy as np

#: Absolute clip ceiling, applied to raw uint8 intensities before scaling.
CLIP_MIN = 0
CLIP_MAX = 200

#: Divisor applied after the clip. This is albumentations' default
#: ``max_pixel_value``, NOT ``CLIP_MAX``. Tying it to ``CLIP_MAX`` is the
#: defect fixed by PR #1371.
NORMALIZE_DIVISOR = 255.0


def preprocess(tile: np.ndarray) -> np.ndarray:
    """Apply the training-time input contract to a raw tile.

    Args:
        tile: raw intensities, any shape, typically uint8.

    Returns:
        float32 array of the same shape, values in ``[0, CLIP_MAX / 255]``
        i.e. ``[0, 0.7843...]``.
    """
    clipped = np.clip(tile, CLIP_MIN, CLIP_MAX)
    return clipped.astype(np.float32) / np.float32(NORMALIZE_DIVISOR)


def expected_value(raw: float) -> float:
    """Reference implementation for tests and sanity checks."""
    return min(max(raw, CLIP_MIN), CLIP_MAX) / NORMALIZE_DIVISOR
