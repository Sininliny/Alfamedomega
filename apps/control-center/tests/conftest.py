import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

FAKE_TRAIN = HERE / "fake_train.py"


@pytest.fixture
def fake_export(tmp_path):
    """A training export folder as training/alfamedomega_train.py export writes it."""
    import subprocess
    out = tmp_path / "export"
    subprocess.run([sys.executable, str(FAKE_TRAIN), "export", "--tier", "lite", "--out", str(out)],
                   check=True, capture_output=True)
    return out


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    from alfacc.context import Context
    c = Context(tmp_path / "home", tmp_path / "Models")
    c.settings.training_python = sys.executable
    c.settings.training_script = str(FAKE_TRAIN)
    return c
