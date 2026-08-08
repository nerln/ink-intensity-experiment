"""The timesformer path, whose z-window the checkpoint does not constrain.

The dangerous property of this checkpoint is that it loads and runs happily for
any frame count, because its position embeddings are rotary. A wrong z-window
produces no error at all, just a quietly invalid result. These tests pin that
property so nobody later mistakes "it ran" for "it was configured right".
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import requires_checkpoint
from infer import TIMESFORMER_CHECKPOINT, InkEngine, load_timesformer

TS_AVAILABLE = Path(TIMESFORMER_CHECKPOINT).exists() and bool(
    list(Path(TIMESFORMER_CHECKPOINT).glob("*.ckpt"))
)
requires_timesformer = pytest.mark.skipif(
    not TS_AVAILABLE, reason=f"timesformer checkpoint not found at {TIMESFORMER_CHECKPOINT}"
)


@requires_timesformer
@pytest.mark.parametrize("num_frames", [26, 30, 33])
def test_loads_cleanly_for_any_frame_count(num_frames):
    """Rotary embeddings leave the frame count free, so every value loads 0/0."""
    model, config = load_timesformer(TIMESFORMER_CHECKPOINT, num_frames=num_frames)
    assert sum(v.numel() for v in model.state_dict().values()) == 37_959_744
    assert config["input_depth"] == num_frames
    assert config["input_size"] == 64


@requires_timesformer
def test_frame_count_does_not_change_the_weights():
    """Pins WHY the window cannot be recovered from the checkpoint."""
    a, _ = load_timesformer(TIMESFORMER_CHECKPOINT, num_frames=26)
    b, _ = load_timesformer(TIMESFORMER_CHECKPOINT, num_frames=33)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


@requires_timesformer
def test_output_is_a_4x4_logit_map():
    model, _ = load_timesformer(TIMESFORMER_CHECKPOINT, num_frames=30)
    with torch.inference_mode():
        out = model(torch.zeros(2, 1, 30, 64, 64))
    assert tuple(out.shape) == (2, 1, 4, 4)


@requires_timesformer
def test_engine_refuses_to_guess_the_window():
    """num_frames is mandatory: silently defaulting it would invalidate a run."""
    with pytest.raises(ValueError, match="explicit num_frames"):
        InkEngine(model_kind="timesformer")


@requires_timesformer
def test_engine_rejects_a_mismatched_depth():
    engine = InkEngine(model_kind="timesformer", num_frames=30, stride=32)
    with pytest.raises(ValueError, match="expects D=30"):
        engine.predict(np.zeros((64, 64, 33), dtype=np.uint8))


@requires_timesformer
def test_resnet3d_rejects_num_frames():
    with pytest.raises(ValueError, match="does not apply to resnet3d"):
        InkEngine(model_kind="resnet3d", num_frames=30)


@requires_timesformer
def test_timesformer_predict_is_deterministic():
    import hashlib

    engine = InkEngine(model_kind="timesformer", num_frames=30, stride=32)
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:128, 0:128]
    base = 90 + 55 * np.sin(xx / 9.0) + 40 * np.cos(yy / 11.0) + rng.normal(0, 10, (128, 128))
    roi = np.clip(base[:, :, None] + np.linspace(-20, 20, 30)[None, None, :], 0, 255).astype(np.uint8)

    def digest(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    first, second = engine.predict(roi), engine.predict(roi)
    assert digest(first) == digest(second)
    assert first.shape == (128, 128)
    assert float(first.std()) > 0.0


@requires_timesformer
def test_engine_defaults_tile_to_the_trained_size():
    engine = InkEngine(model_kind="timesformer", num_frames=30)
    assert engine.input_size == 64
    assert engine.config.tile == 64
    assert engine.config.stride == 32  # tile // 2 when unset
