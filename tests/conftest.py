import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer import DEFAULT_CHECKPOINT  # noqa: E402


def checkpoint_available() -> bool:
    d = Path(DEFAULT_CHECKPOINT)
    return (d / "config.json").exists() and (d / "model.safetensors").exists()


requires_checkpoint = pytest.mark.skipif(
    not checkpoint_available(),
    reason=f"checkpoint not found at {DEFAULT_CHECKPOINT}",
)


@pytest.fixture(scope="session")
def engine():
    """One engine for the whole session; loading the checkpoint costs seconds."""
    from infer import InkEngine

    return InkEngine()
