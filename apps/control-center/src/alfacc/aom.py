"""Control Center services for the independent tensor compiler learner/runtime.

The caller explicitly supplies repository and model paths. These services never
silently route AOM operations to the legacy method-vector trainer.
"""
import json
import subprocess
import sys
from pathlib import Path


def train_tensor_model(repository: Path, manifest: Path, destination: Path) -> dict:
    result = subprocess.run([sys.executable, "-m", "training.aom", "train",
                             str(Path(manifest).resolve()), str(Path(destination).resolve())],
                            cwd=Path(repository), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def train_runtime_tensor_model(repository: Path, manifest: Path, destination: Path) -> dict:
    result = subprocess.run([sys.executable, "-m", "training.aom", "train-runtime",
                             str(Path(manifest).resolve()), str(Path(destination).resolve())],
                            cwd=Path(repository), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def infer_tensor_preset(repository: Path, model: Path, wav: Path, output: Path,
                        *, mask_threshold: float) -> dict:
    if not 0 < mask_threshold < 1:
        raise ValueError("Explicit mask threshold must lie between zero and one")
    result = subprocess.run([sys.executable, "-m", "training.aom", "infer",
                             str(Path(model).resolve()), str(Path(wav).resolve()),
                             str(Path(output).resolve()), "--mask-threshold", str(mask_threshold)],
                            cwd=Path(repository), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(Path(output).read_text(encoding="utf-8"))


def validate_tensor_document(document: dict) -> None:
    """Early UI format check; authoritative semantic validation is native prepare."""
    if document.get("schema") != "aom.tensor/1" or not document.get("ontology"):
        raise ValueError("An explicitly versioned AOM tensor document is required")
    tensors = document.get("tensors", {})
    if set(tensors) != {"activation", "parameters", "routing", "modulation"}:
        raise ValueError("AOM compiler requires activation, parameters, routing and modulation tensors")


def import_tensor_wav(repository: Path, model: Path, wav: Path, template: Path,
                      bounds: Path, output: Path, *, head: str, mask_threshold: float,
                      sample_rate: float, dll_path=None) -> dict:
    command = [sys.executable, "-m", "training.aom", "import-wav",
               *(str(Path(path).resolve()) for path in (model, wav, template, bounds, output)),
               "--head", head, "--mask-threshold", str(mask_threshold), "--sample-rate", str(sample_rate)]
    if dll_path:
        command.extend(("--dll", str(Path(dll_path).resolve())))
    result = subprocess.run(command, cwd=Path(repository), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(Path(output).read_text(encoding="utf-8"))
