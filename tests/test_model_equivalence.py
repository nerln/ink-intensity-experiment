"""Our rebuilt model must be numerically identical to the published one.

We rebuild the architecture by hand instead of using
``AutoModel.from_pretrained(..., trust_remote_code=True)``, because the shipped
remote code targets ``transformers==4.57.6`` and raises
``AttributeError: 'InkDetectionModel' object has no attribute
'all_tied_weights_keys'`` under ``transformers>=5``.

A hand rebuild is only legitimate if it computes the same function. These tests
import the vendored remote code straight out of the downloaded snapshot, build
the official module tree with it, and compare against ours on identical
weights.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import requires_checkpoint
from infer import DEFAULT_CHECKPOINT, MaxPool2dOverDepth, load_model


def _load_reference_model(tmp_path: Path):
    """Import the snapshot's remote code as a package and build the official model."""
    snapshot = Path(DEFAULT_CHECKPOINT)
    pkg = tmp_path / "refpkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name in ("modeling_inkdetection.py", "configuration_inkdetection.py"):
        shutil.copy(snapshot / name, pkg / name)

    sys.path.insert(0, str(tmp_path))
    try:
        from refpkg.configuration_inkdetection import InkDetectionConfig
        from refpkg.modeling_inkdetection import InkDetectionModel as ReferenceModel
    finally:
        sys.path.remove(str(tmp_path))

    raw = json.loads((snapshot / "config.json").read_text())
    skip = {"architectures", "auto_map", "model_type", "torch_dtype", "transformers_version"}
    config = InkDetectionConfig(**{k: v for k, v in raw.items() if k not in skip})

    from safetensors.torch import load_file

    model = ReferenceModel(config)
    missing, unexpected = model.load_state_dict(
        load_file(str(snapshot / "model.safetensors")), strict=False
    )
    assert not missing and not unexpected
    return model.eval()


@requires_checkpoint
def test_rebuilt_model_matches_official_bit_for_bit(tmp_path):
    reference = _load_reference_model(tmp_path)
    ours, _ = load_model(DEFAULT_CHECKPOINT)

    generator = torch.Generator().manual_seed(1234)
    # Inputs in the real operating range: clip(x, 0, 200) / 255.
    x = torch.rand(1, 1, 62, 256, 256, generator=generator) * (200.0 / 255.0)

    with torch.inference_mode():
        expected = reference(x).logits
        actual = ours(x)

    assert tuple(actual.shape) == tuple(expected.shape) == (1, 1, 64, 64)
    assert torch.equal(actual, expected), (
        f"max abs diff {(actual - expected).abs().max().item():.3e}"
    )


@requires_checkpoint
def test_checkpoint_loads_with_no_missing_or_unexpected_keys():
    """`load_model` raises on any mismatch; this pins the element count too."""
    model, config = load_model(DEFAULT_CHECKPOINT)
    assert sum(v.numel() for v in model.state_dict().values()) == 83_374_585
    assert len(model.state_dict()) == 338
    assert config["input_depth"] == 62
    assert config["input_size"] == 256
    assert config["in_channels"] == 1


# --------------------------------------------------------------------------
# The MPS workaround
# --------------------------------------------------------------------------
def test_maxpool_substitution_equals_maxpool3d():
    """MaxPool2dOverDepth must equal the MaxPool3d it replaces, exactly.

    Kernel and stride along depth are 1 and depth padding is 0, so the 3-D pool
    is an independent 2-D pool per slice. Max is a selection, not an
    arithmetic reduction, so equality is exact rather than approximate.
    """
    reference = torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    ours = MaxPool2dOverDepth(kernel_size=3, stride=2, padding=1)

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 62, 64, 64, generator=generator)

    assert torch.equal(ours(x), reference(x))


def test_maxpool_substitution_rejects_wrong_rank():
    with pytest.raises(ValueError, match="5-D"):
        MaxPool2dOverDepth()(torch.zeros(2, 3, 8, 8))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_maxpool3d_is_still_unavailable_on_mps():
    """Pin the reason the workaround exists. If this ever fails, torch fixed it."""
    x = torch.randn(1, 2, 4, 16, 16, device="mps")
    pool = torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    try:
        pool(x)
        torch.mps.synchronize()
    except NotImplementedError as exc:
        assert "max_pool3d_with_indices" in str(exc)
    else:
        pytest.skip("torch now implements max_pool3d on MPS; the workaround is redundant")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_maxpool_substitution_matches_across_devices():
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(1, 4, 62, 32, 32, generator=generator)
    ours = MaxPool2dOverDepth()
    on_cpu = ours(x)
    on_mps = ours(x.to("mps")).cpu()
    assert torch.equal(on_cpu, on_mps)
