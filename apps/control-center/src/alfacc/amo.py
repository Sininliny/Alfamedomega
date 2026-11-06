"""Control Center services for the independent tensor compiler learner/runtime.

The caller explicitly supplies repository and model paths. These services never
silently route AMO operations to the legacy method-vector trainer.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import uuid


TRAINING_CATEGORIES = {"train": "wav_preset_learning", "train-runtime": "runtime",
                       "train-preference": "user_preference", "train-importance": "context_importance"}


def create_tensor_job(manager, repository, action, arguments, *, python=None,
                      checkpoint=None, resume=False, checkpoint_every=100):
    """Queue independent AMO work with durable history and cooperative cancellation."""
    from .jobs import cmd_step, FINISHED
    if action not in (*TRAINING_CATEGORIES, "preflight", "infer", "import-wav", "build-dataset", "evaluate-wav",
                      "rank-controls", "wav-to-preset"):
        raise ValueError("Unsupported AMO task")
    repository = Path(repository).resolve()
    if not (repository / "training/amo/__main__.py").is_file():
        raise ValueError("Select a repository containing the AMO training package")
    if action in TRAINING_CATEGORIES:
        if len(arguments) != 2:
            raise ValueError("Training requires a manifest and output package")
        arguments = [str(Path(value).expanduser().resolve()) for value in arguments]
    argv = [python or sys.executable, "-u", "-m", "training.amo", action, *map(str, arguments)]
    params = {"repository": str(repository), "category": TRAINING_CATEGORIES.get(action), "resume": resume}
    title = {"train": "Train WAV model", "train-runtime": "Train runtime compiler",
             "train-preference": "Train user preference", "train-importance": "Train context importance",
             "preflight": "Check training manifest", "build-dataset": "Build labelled dataset",
             "evaluate-wav": "Evaluate native reconstruction", "infer": "Inspect WAV prediction",
             "import-wav": "Import editable WAV preset", "rank-controls": "Rank sound controls",
             "wav-to-preset": "WAV to Preset"}[action]
    env = {"PYTHONUTF8": "1"}
    if action == "wav-to-preset" and (library := bundled_tensor_library(repository)):
        env["AMO_TENSOR_LIBRARY"] = str(library)
    step = cmd_step(argv, cwd=repository, label=title, env=env)
    if action in TRAINING_CATEGORIES:
        if len(arguments) != 2:
            raise ValueError("Training requires a manifest and output package")
        destination = Path(arguments[1]).resolve()
        checkpoint = Path(checkpoint).resolve() if checkpoint else destination.with_name(destination.name + ".checkpoint.pt")
        if checkpoint == destination or checkpoint == Path(arguments[0]).resolve():
            raise ValueError("Checkpoint must be separate from the model and manifest")
        if destination.is_dir():
            raise ValueError("Select a model output file, not a directory")
        if any(j.get("status") not in FINISHED and j.get("params", {}).get("destination") == str(destination) for j in manager.list()):
            raise ValueError("This output belongs to an unfinished training job; choose another output")
        cancel_file = manager.dir.resolve() / ".cancels" / (uuid.uuid4().hex + ".flag")
        step["cancel_file"] = str(cancel_file)
        step["argv"] += ["--checkpoint", str(checkpoint), "--checkpoint-every", str(checkpoint_every),
                         "--cancel-file", str(cancel_file)]
        if resume:
            step["argv"].append("--resume")
        params.update(checkpoint=str(checkpoint), destination=str(destination), manifest=str(Path(arguments[0]).resolve()))
    return manager.create("amo", title, [step], params=params)


# The structure-encoder weights shipped with the suite, relative to the repository (or the
# installed app folder that holds training/). Settings > WAV to Preset model overrides it.
DEFAULT_WAV_PRESET_MODEL = Path("models/corpus-encoder/run3/model.pt")


def default_wav_preset_model(repository) -> Path:
    return Path(repository) / DEFAULT_WAV_PRESET_MODEL


def bundled_tensor_library(repository):
    """alfa_tensor_c next to an installed suite (bin/), else None so the usual lookup applies."""
    for name in ("alfa_tensor_c.dll", "libalfa_tensor_c.so", "libalfa_tensor_c.dylib"):
        candidate = Path(repository) / "bin" / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def preset_library() -> Path:
    """The plugin's sound library folder (PresetStore::defaultDirectory)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "Alfamedomega" / "TensorPresets"


def _run_training(repository, manifest, destination, action, *, home=None, python=None):
    from .jobs import JobManager, SUCCEEDED
    from .settings import default_home, load_settings
    home = Path(home) if home else default_home()
    manager = JobManager(home)
    settings = load_settings(home)
    settings.repo_dir = str(repository)
    job = create_tensor_job(manager, repository, action,
                            [str(Path(manifest).resolve()), str(Path(destination).resolve())],
                            python=python or settings.resolved_training_python())
    done = manager.run(job["id"])
    log = manager.read_log(job["id"])
    if done["status"] != SUCCEEDED:
        raise RuntimeError(log or done["error"] or done["status"])
    for line in reversed(log.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "model" in value:
            return value
    raise RuntimeError("Training finished without its publication report")


def train_tensor_model(repository: Path, manifest: Path, destination: Path, *, home=None, python=None) -> dict:
    return _run_training(repository, manifest, destination, "train", home=home, python=python)


def train_runtime_tensor_model(repository: Path, manifest: Path, destination: Path, *, home=None, python=None) -> dict:
    return _run_training(repository, manifest, destination, "train-runtime", home=home, python=python)


def infer_tensor_preset(repository: Path, model: Path, wav: Path, output: Path,
                        *, mask_threshold: float, python=None, home=None) -> dict:
    if not 0 < mask_threshold < 1:
        raise ValueError("Explicit mask threshold must lie between zero and one")
    result = subprocess.run([_interpreter(repository, python, home), "-m", "training.amo", "infer",
                             str(Path(model).resolve()), str(Path(wav).resolve()),
                             str(Path(output).resolve()), "--mask-threshold", str(mask_threshold)],
                            cwd=Path(repository), capture_output=True, text=True, encoding="utf-8",
                            env={**os.environ, "PYTHONUTF8": "1"})
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(Path(output).read_text(encoding="utf-8"))


def validate_tensor_document(document: dict) -> None:
    """Early UI format check; authoritative semantic validation is native prepare."""
    # "aom.tensor/1" is the pre-rename spelling of the same schema. rename-compat
    if document.get("schema") not in ("amo.tensor/1", "aom.tensor/1") or not document.get("ontology"):  # rename-compat
        raise ValueError("An explicitly versioned AMO tensor document is required")
    tensors = document.get("tensors", {})
    if set(tensors) != {"activation", "parameters", "routing", "modulation"}:
        raise ValueError("AMO compiler requires activation, parameters, routing and modulation tensors")


def import_tensor_wav(repository: Path, model: Path, wav: Path, template: Path,
                      bounds: Path, output: Path, *, head: str, mask_threshold: float,
                      sample_rate: float, dll_path=None, python=None, home=None) -> dict:
    command = [_interpreter(repository, python, home), "-m", "training.amo", "import-wav",
               *(str(Path(path).resolve()) for path in (model, wav, template, bounds, output)),
               "--head", head, "--mask-threshold", str(mask_threshold), "--sample-rate", str(sample_rate)]
    if dll_path:
        command.extend(("--dll", str(Path(dll_path).resolve())))
    result = subprocess.run(command, cwd=Path(repository), capture_output=True, text=True,
                            encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(Path(output).read_text(encoding="utf-8"))


def _interpreter(repository, python=None, home=None):
    from .settings import default_home, load_settings
    settings = load_settings(Path(home) if home else default_home())
    settings.repo_dir = str(repository)
    return python or settings.resolved_training_python()
