"""Everything a front end (CLI or GUI) needs, built once from the user's home folder."""

from pathlib import Path

from . import training  # noqa: F401  (registers the install_export job action)
from .jobs import JobManager
from .models import ModelStore
from .preference import PreferenceStore
from .settings import Settings, default_home, load_settings, save_settings


class Context:
    def __init__(self, home: Path | None = None, models_dir: Path | None = None):
        self.home = Path(home) if home else default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.settings: Settings = load_settings(self.home)
        self._models_override = Path(models_dir) if models_dir else None
        self.jobs = JobManager(self.home)
        self.prefs = PreferenceStore(self.home)

    @property
    def models_dir(self) -> Path:
        return self._models_override or self.settings.resolved_models_dir()

    @property
    def store(self) -> ModelStore:
        return ModelStore(self.models_dir)

    @property
    def python(self) -> str:
        return self.settings.resolved_training_python()

    @property
    def script(self) -> str:
        return self.settings.resolved_training_script()

    @property
    def runs_dir(self) -> Path:
        return self.settings.resolved_runs_dir(self.home)

    def save_settings(self) -> None:
        save_settings(self.home, self.settings)
