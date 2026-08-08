"""The engine must be bit-for-bit reproducible.

The experiment measures the difference between two ink maps. Any run-to-run
jitter in the engine itself shows up as signal, so "close enough" is not good
enough here: the same input must produce byte-identical output.
"""

import hashlib

import numpy as np
import pytest
import torch

from conftest import requires_checkpoint
from infer import InkEngine, grid_1d, hann2d, read_roi


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _synthetic_roi(h: int, w: int, d: int, seed: int = 42) -> np.ndarray:
    """Structured, not pure noise, so the network produces varied output."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        90.0
        + 55.0 * np.sin(xx / 17.0)
        + 40.0 * np.cos(yy / 23.0)
        + rng.normal(0.0, 12.0, size=(h, w))
    )
    vol = base[:, :, None] + np.linspace(-25.0, 25.0, d)[None, None, :]
    return np.clip(vol, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Determinism of the full engine
# --------------------------------------------------------------------------
@requires_checkpoint
def test_two_calls_same_engine_are_bit_identical(engine):
    roi = _synthetic_roi(384, 384, engine.input_depth)

    first = engine.predict(roi)
    second = engine.predict(roi)

    assert first.shape == second.shape == (384, 384)
    assert first.dtype == second.dtype == np.float32
    assert _digest(first) == _digest(second), "same input produced different bytes"
    assert np.array_equal(first, second)
    # Guard against a degenerate all-zero map trivially passing.
    assert float(first.std()) > 0.0, "output is constant; the test proves nothing"


@requires_checkpoint
def test_repeated_calls_stay_identical(engine):
    """Three calls, in case state accumulates only after the first repeat."""
    roi = _synthetic_roi(256, 256, engine.input_depth, seed=7)
    digests = {_digest(engine.predict(roi)) for _ in range(3)}
    assert len(digests) == 1, f"engine drifted across calls: {digests}"


@requires_checkpoint
def test_fresh_engine_reproduces_the_same_map():
    """A newly constructed engine must reproduce a prior engine's output."""
    roi = _synthetic_roi(256, 256, 62, seed=11)
    a = InkEngine().predict(roi)
    b = InkEngine().predict(roi)
    assert _digest(a) == _digest(b), "checkpoint load path is not deterministic"


@requires_checkpoint
def test_interleaving_a_different_roi_does_not_change_the_result(engine):
    """No hidden state carries between ROIs (e.g. BatchNorm running stats)."""
    roi = _synthetic_roi(256, 256, engine.input_depth, seed=3)
    other = _synthetic_roi(256, 256, engine.input_depth, seed=99)

    before = engine.predict(roi)
    engine.predict(other)
    after = engine.predict(roi)

    assert _digest(before) == _digest(after)


@requires_checkpoint
def test_model_stays_in_eval_mode(engine):
    assert not engine.model.training
    for module in engine.model.modules():
        assert not module.training


# --------------------------------------------------------------------------
# Determinism of the pure helpers
# --------------------------------------------------------------------------
def test_grid_covers_and_touches_the_border():
    xs = grid_1d(384, 256, 128)
    assert xs == [0, 128]
    assert xs[-1] + 256 == 384, "last tile must touch the border"

    # Non-multiple length still gets a border-touching final tile.
    xs = grid_1d(300, 256, 128)
    assert xs[-1] == 300 - 256
    assert xs == sorted(xs)

    # An ROI smaller than a tile yields exactly one origin at 0.
    assert grid_1d(100, 256, 128) == [0]


def test_read_roi_zero_pads_out_of_bounds():
    vol = np.ones((10, 10, 4), dtype=np.uint8)
    out = read_roi(vol, 5, 15, 5, 15)
    assert out.shape == (10, 10, 4)
    assert out[:5, :5].all(), "in-bounds region must be preserved"
    assert not out[5:, :].any(), "out-of-bounds region must be zero"
    assert not out[:, 5:].any()


def test_hann_window_is_deterministic_and_normalised():
    a, b = hann2d(256, 256), hann2d(256, 256)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32
    assert float(a.sum()) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# Input contract enforcement
# --------------------------------------------------------------------------
@requires_checkpoint
def test_wrong_depth_is_rejected_with_a_useful_message(engine):
    with pytest.raises(ValueError, match="depth LAST"):
        engine.predict(np.zeros((62, 256, 256), dtype=np.uint8))


@requires_checkpoint
def test_two_dimensional_input_is_rejected(engine):
    with pytest.raises(ValueError, match=r"3-D"):
        engine.predict(np.zeros((256, 256), dtype=np.uint8))


@requires_checkpoint
def test_all_zero_roi_returns_all_zero_map(engine):
    """Fully invalid ROIs must not fabricate ink."""
    out = engine.predict(np.zeros((256, 256, engine.input_depth), dtype=np.uint8))
    assert out.shape == (256, 256)
    assert not out.any(), "zero-coverage ROI produced non-zero ink"
