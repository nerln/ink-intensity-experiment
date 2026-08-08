"""The preprocessing contract must be /255, not /200 and not 1.0.

The headline case: a surface volume that is constant 200 (the clip ceiling)
must reach the network as 0.7843, NOT as 1.0. Getting this wrong by using the
clip value as the divisor inflates every input by 255/200 = 1.275x relative to
training, which is the defect PR #1371 fixes and which would silently bias the
causal comparison.
"""

import numpy as np
import pytest
import torch

from conftest import requires_checkpoint
from infer import CLIP_MAX, NORMALIZE_DIVISOR, preprocess

EXPECTED_AT_CEILING = 200.0 / 255.0  # 0.78431372...


# --------------------------------------------------------------------------
# Unit level: the function itself
# --------------------------------------------------------------------------
def test_constant_200_maps_to_200_over_255():
    out = preprocess(np.full((4, 4, 62), 200, dtype=np.uint8))
    assert np.allclose(out, EXPECTED_AT_CEILING, atol=1e-7)
    assert pytest.approx(float(out.max()), abs=1e-7) == 0.7843137


def test_constant_200_is_not_1_point_0():
    """Guards specifically against dividing by the clip value instead of 255."""
    out = preprocess(np.full((4, 4, 62), 200, dtype=np.uint8))
    assert not np.allclose(out, 1.0), "input was divided by the clip value (200), not by 255"
    assert float(out.max()) < 0.79


def test_divisor_is_255_not_clip_max():
    assert NORMALIZE_DIVISOR == 255.0
    assert CLIP_MAX == 200
    assert NORMALIZE_DIVISOR != float(CLIP_MAX)


def test_clip_ceiling_saturates():
    """Everything at or above 200 collapses onto the same value."""
    raw = np.array([[[200, 201, 254, 255]]], dtype=np.uint8)
    out = preprocess(raw)
    assert np.allclose(out, EXPECTED_AT_CEILING, atol=1e-7)


def test_below_ceiling_is_untouched_before_scaling():
    raw = np.array([[[0, 1, 100, 199]]], dtype=np.uint8)
    out = preprocess(raw)
    assert np.allclose(out, np.array([0, 1, 100, 199]) / 255.0, atol=1e-7)


def test_output_range_and_dtype():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(16, 16, 62), dtype=np.uint8)
    out = preprocess(raw)
    assert out.dtype == np.float32
    assert out.shape == raw.shape
    assert out.min() >= 0.0
    assert out.max() <= EXPECTED_AT_CEILING + 1e-7


def test_zero_stays_zero():
    """Validity masking relies on zero meaning 'no surface'."""
    assert float(preprocess(np.zeros((2, 2, 62), dtype=np.uint8)).max()) == 0.0


# --------------------------------------------------------------------------
# End-to-end: what actually reaches the network's first layer
# --------------------------------------------------------------------------
@requires_checkpoint
def test_tensor_entering_the_model_is_200_over_255(engine):
    """Hook the real first conv and inspect the tensor the engine feeds it.

    Unit-testing `preprocess` proves the function is right; this proves the
    engine actually calls it on the path to the network.
    """
    seen = {}

    def hook(_module, inputs):
        seen["x"] = inputs[0].detach().float().cpu()

    handle = engine.model.backbone.conv1.register_forward_pre_hook(hook)
    try:
        roi = np.full((engine.input_size, engine.input_size, engine.input_depth), 200, dtype=np.uint8)
        engine.predict(roi)
    finally:
        handle.remove()

    x = seen["x"]
    assert x.ndim == 5, f"model input must be (B,1,D,H,W); got {tuple(x.shape)}"
    assert tuple(x.shape) == (1, 1, engine.input_depth, engine.input_size, engine.input_size)
    assert float(x.max()) == pytest.approx(EXPECTED_AT_CEILING, abs=1e-5)
    assert float(x.min()) == pytest.approx(EXPECTED_AT_CEILING, abs=1e-5)
    assert float(x.max()) < 0.99, "the engine fed the model 1.0 — divisor is wrong"


@requires_checkpoint
def test_engine_clips_before_scaling(engine):
    """A 255-valued ROI must arrive as 0.784, not 1.0 — proving the clip ran."""
    seen = {}
    handle = engine.model.backbone.conv1.register_forward_pre_hook(
        lambda _m, i: seen.__setitem__("x", i[0].detach().float().cpu())
    )
    try:
        roi = np.full((engine.input_size, engine.input_size, engine.input_depth), 255, dtype=np.uint8)
        engine.predict(roi)
    finally:
        handle.remove()

    assert float(seen["x"].max()) == pytest.approx(EXPECTED_AT_CEILING, abs=1e-5)
