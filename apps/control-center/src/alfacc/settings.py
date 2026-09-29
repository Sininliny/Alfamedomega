"""Where things live, and the user's settings.

Default locations follow docs/packaging/packaging-plan.md section 3 for the shared Models folder.
Every path can be overridden in settings.json or with an environment variable, which the tests
and CI use to stay out of the real system folders.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_DIR_NAME = "Alfamedomega"


def default_models_dir() -> Path:
    if env := os.environ.get("ALFA_MODELS_DIR"):
        return Path(env)
    if sys.platform == "darwin":
        return Path("/Library/Application Support") / APP_DIR_NAME / "Models"
    if sys.platform == "win32":
        return Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / APP_DIR_NAME / "Models"
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / APP_DIR_NAME / "Models"


def default_home() -> Path:
    """Per-user folder for settings, jobs, logs and preference data."""
    if env := os.environ.get("ALFA_CC_HOME"):
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME / "ControlCenter"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / APP_DIR_NAME / "ControlCenter"
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / APP_DIR_NAME / "ControlCenter"


def _find_training_script() -> str:
    """The training script in a source checkout, if the app runs from one."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "training" / "alfamedomega_train.py"
        if candidate.is_file():
            return str(candidate)
    return ""


@dataclass
class Settings:
    models_dir: str = ""
    # Python interpreter that has PyTorch and ONNX Runtime installed, used to run training.
    training_python: str = ""
    training_script: str = ""
    runs_dir: str = ""
    # Command that runs the preference learner; see docs/control-center/preference-batch.md.
    preference_learner: str = ""
    extra: dict = field(default_factory=dict)

    def resolved_models_dir(self) -> Path:
        return Path(self.models_dir) if self.models_dir else default_models_dir()

    def resolved_training_python(self) -> str:
        return self.training_python or sys.executable

    def resolved_training_script(self) -> str:
        return self.training_script or _find_training_script()

    def resolved_runs_dir(self, home: Path) -> Path:
        return Path(self.runs_dir) if self.runs_dir else home / "runs"


def load_settings(home: Path) -> Settings:
    path = home / "settings.json"
    if not path.exists():
        return Settings()
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in data.items() if k in known})


def save_settings(home: Path, settings: Settings) -> None:
    home.mkdir(parents=True, exist_ok=True)
    write_json_atomic(home / "settings.json", asdict(settings))


def write_json_atomic(path: Path, data) -> None:
    """Write JSON so a reader (the plugin, another window) never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
