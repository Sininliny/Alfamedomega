import shlex
import shutil
import sys
from pathlib import Path

import pytest

from alfacc import preference, settings
from alfacc.settings import Settings

# The real tokenizer when this checkout has preference/, else a snapshot of it.
PREF_TOKENS = Path(settings.find_in_checkout("preference", "alfapref", "tokens.py")
                   or Path(__file__).with_name("_pref_tokens_fixture.py.txt"))


@pytest.fixture
def repo(tmp_path):
    """A checkout with the files the control center looks for."""
    r = tmp_path / "Alfamedomega"
    (r / "training").mkdir(parents=True)
    (r / "training" / "alfamedomega_train.py").write_text("# script\n")
    (r / "preference" / "alfapref").mkdir(parents=True)
    (r / "preference" / "learner.py").write_text("# learner\n")
    (r / "preference" / "alfapref" / "tokens.py").write_text(PREF_TOKENS.read_text())
    return r


def test_repo_folder_locates_script_learner_and_venv(repo):
    venv = repo / ".venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.write_text("")
    s = Settings(repo_dir=str(repo))
    assert s.resolved_training_script() == str(repo / "training" / "alfamedomega_train.py")
    assert s.resolved_training_python() == str(venv)
    assert shlex.split(s.resolved_preference_learner()) == [str(venv), str(repo / "preference" / "learner.py")]
    s.preference_learner = "custom --flag"
    assert s.resolved_preference_learner() == "custom --flag"


def test_packaged_app_needs_the_repo_folder(monkeypatch, repo):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    s = Settings()
    assert s.resolved_training_script() == ""        # nothing to find inside the app bundle
    assert s.resolved_preference_learner() == ""
    assert s.resolved_training_python() == (shutil.which("python3") or shutil.which("python") or "python3")
    s.repo_dir = str(repo)
    assert s.resolved_training_script().endswith("alfamedomega_train.py")


def test_labels_come_from_the_preference_tokenizer(repo, monkeypatch):
    monkeypatch.setattr(preference, "_SHARED_WORDS", None)
    preference.use_repo(str(repo))
    assert preference._SHARED_WORDS is not None
    assert preference.tokens("warm_pad_03") == ["warm", "pad"]
    assert preference.tokens("AiryVocal v2 (final)") == ["airy", "vocal"]


def test_settings_round_trip_ignores_unknown_keys(tmp_path):
    settings.save_settings(tmp_path, Settings(repo_dir="/src/alfa"))
    (tmp_path / "settings.json").write_text(
        (tmp_path / "settings.json").read_text().replace('"extra"', '"from_the_future": 1, "extra"'))
    assert settings.load_settings(tmp_path).repo_dir == "/src/alfa"
