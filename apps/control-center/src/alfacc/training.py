"""Training jobs: build the command lines for training/alfamedomega_train.py, locally or on Colab.

The control center never imports PyTorch. It runs the training script in the interpreter chosen in
Settings, reads its log lines for progress, and can export the result and install it into the
Models folder as the last steps of the same job.
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from . import jobs
from .models import ModelStore
from .tiers import STAGES, check_tier


@dataclass
class TrainSpec:
    tier: str = "lite"
    stage: str = "pretrain"
    steps: int = 200_000
    batch: int = 16
    accum: int = 4
    clip: float = 1.0
    lr: float | None = None
    init: str = ""        # checkpoint to continue from
    teacher: str = ""     # Studio checkpoint, for distill
    data: str = ""        # NSynth-layout folder, for finetune and distill
    seed: int = 0
    device: str = ""
    smoke: bool = False
    export: bool = True   # export ONNX graphs when training finishes
    install: bool = True  # then install them into the Models folder
    version: str = ""     # model version to install as; empty picks the next patch version
    int8: bool = False    # also write int8 copies (METHOD.md stage 5, meant for Lite)

    def validate(self) -> None:
        check_tier(self.tier)
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}; expected one of {', '.join(STAGES)}")
        if self.stage == "distill" and not self.teacher:
            raise ValueError("distillation needs a Studio teacher checkpoint")
        if self.stage == "finetune" and not self.data:
            raise ValueError("fine-tuning needs a folder of labelled notes")
        if self.stage in ("polish",) and not self.init:
            raise ValueError("the polish stage continues from a checkpoint; choose one")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.install and not self.export:
            raise ValueError("installing needs the export step")

    def run_name(self) -> str:
        return f"{self.tier}-{self.stage}" + ("-smoke" if self.smoke else "")

    def train_args(self, out: str) -> list:
        a = ["train", "--tier", self.tier, "--stage", self.stage, "--steps", self.steps,
             "--batch", self.batch, "--accum", self.accum, "--clip", self.clip,
             "--seed", self.seed, "--out", out]
        if self.lr:
            a += ["--lr", self.lr]
        for flag, value in (("--init", self.init), ("--teacher", self.teacher),
                            ("--data", self.data), ("--device", self.device)):
            if value:
                a += [flag, value]
        if self.smoke:
            a.append("--smoke")
        return [str(x) for x in a]

    def checkpoint_name(self) -> str:
        # train() saves <tier>_<stage>.pt in --out.
        return f"{self.tier}_{self.stage}.pt"

    def export_args(self, ckpt: str, out: str, version: str) -> list:
        # The script writes <out>/<tier>-<version>/ and <out>/manifest.json, a Models folder.
        a = ["export", "--ckpt", ckpt, "--tier", self.tier, "--out", out, "--version", version]
        if self.int8:
            a.append("--int8")
        return a


def check_environment(python: str, script: str, timeout: float = 60) -> dict:
    """Can this interpreter run training? Returns versions or the reason it can't."""
    out = {"python": python, "script": script, "script_ok": bool(script) and Path(script).is_file()}
    probe = ("import sys, torch; import onnxruntime as ort; "
             "print(sys.version.split()[0], torch.__version__, ort.__version__, "
             "torch.cuda.is_available())")
    try:
        r = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        out.update(ok=False, error=str(e))
        return out
    if r.returncode != 0:
        last = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
        out.update(ok=False, error=last)
        return out
    py, torch_v, ort_v, cuda = r.stdout.split()
    out.update(ok=out["script_ok"], python_version=py, torch=torch_v, onnxruntime=ort_v,
               cuda=cuda == "True", error=None if out["script_ok"] else "training script not found")
    return out


def create_training_job(manager: jobs.JobManager, spec: TrainSpec, python: str, script: str,
                        runs_dir: Path, models_dir: Path) -> dict:
    spec.validate()
    if not script or not Path(script).is_file():
        raise ValueError("training script not found; set it in Settings")
    run_dir = Path(runs_dir) / spec.run_name()
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = Path(runs_dir) / f"{spec.run_name()}-{n}"
    run_dir.mkdir(parents=True)
    steps = [jobs.cmd_step([python, script, *spec.train_args(str(run_dir))], cwd=run_dir,
                           label=f"Train {spec.tier} ({spec.stage})")]
    if spec.export:
        export_dir = run_dir / "export"
        version = spec.version or ModelStore(Path(models_dir)).next_version(spec.tier)
        steps.append(jobs.cmd_step(
            [python, script, *spec.export_args(str(run_dir / spec.checkpoint_name()), str(export_dir), version)],
            cwd=run_dir, label="Export ONNX graphs"))
        if spec.install:
            steps.append(jobs.action_step(
                "install_export", label="Install into Models folder", src=str(export_dir),
                models_dir=str(models_dir), version=version,
                notes=f"trained by control center, {spec.stage}, {spec.steps} steps"))
    title = f"Train {spec.tier.capitalize()} · {spec.stage}" + (" (smoke test)" if spec.smoke else "")
    return manager.create("train", title, steps, params={**asdict(spec), "run_dir": str(run_dir)})


def create_export_job(manager: jobs.JobManager, tier: str, ckpt: str, python: str, script: str,
                      runs_dir: Path, models_dir: Path, version: str = "", int8: bool = False,
                      install: bool = True) -> dict:
    """Export (and install) an existing checkpoint, for example one trained on Colab."""
    check_tier(tier)
    if ckpt and not Path(ckpt).is_file():
        raise ValueError(f"checkpoint not found: {ckpt}")
    spec = TrainSpec(tier=tier, int8=int8)
    base = Path(runs_dir) / f"{tier}-export"
    out, n = base, 1
    while out.exists():
        n += 1
        out = Path(f"{base}-{n}")
    out.mkdir(parents=True)
    version = version or ModelStore(Path(models_dir)).next_version(tier)
    args = spec.export_args(ckpt, str(out / "export"), version)
    if not ckpt:  # the script exports an untrained model, which tests the pipeline
        args = [a for i, a in enumerate(args) if a != "--ckpt" and args[i - 1] != "--ckpt"]
    steps = [jobs.cmd_step([python, script, *args], cwd=out, label=f"Export {tier}")]
    if install:
        steps.append(jobs.action_step("install_export", label="Install into Models folder",
                                      src=str(out / "export"), models_dir=str(models_dir),
                                      version=version, notes=f"exported from {Path(ckpt).name if ckpt else 'an untrained model'}"))
    return manager.create("export", f"Export {tier.capitalize()}", steps,
                          params={"tier": tier, "ckpt": ckpt, "run_dir": str(out)})


@jobs.action("install_export")
def _install_export(args: dict, log) -> dict:
    store = ModelStore(Path(args["models_dir"]))
    entries = store.import_export(Path(args["src"]), version=args.get("version") or None,
                                  notes=args.get("notes", ""))
    for e in entries:
        log(f"installed {e['folder']} ({e['bytes'] / 1e6:.1f} MB, {len(e['files'])} files) "
            f"into {store.root}")
    return {"installed": ", ".join(e["folder"] for e in entries)}


# ---------------------------------------------------------------------- Colab

def colab_notebook(spec: TrainSpec, script_text: str, drive_dir: str = "alfamedomega") -> dict:
    """A self-contained Colab notebook that trains and exports one tier onto Google Drive.

    The training script is embedded, so the notebook works with a private repository. When it
    finishes, download the export folder (or the zip the last cell makes) and install it with
    Models > Import.
    """
    spec.validate()
    run = f"/content/drive/MyDrive/{drive_dir}/runs/{spec.run_name()}"
    train = " ".join(_q(a) for a in spec.train_args(run))
    export = " ".join(_q(a) for a in spec.export_args(f"{run}/{spec.checkpoint_name()}", f"{run}/export",
                                                          spec.version or "0.1.0"))

    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

    def code(text):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": text.splitlines(True)}

    inputs = [x for x in (spec.init, spec.teacher, spec.data) if x]
    cells = [
        md(f"# Alfamedomega: train {spec.tier} ({spec.stage})\n\n"
           "Made by the Alfamedomega control center. Use a GPU runtime "
           "(Runtime > Change runtime type). Checkpoints and the export go to your Google Drive, "
           "so a disconnect loses at most one checkpoint interval.\n"),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"
             f"!mkdir -p {_q(run)}\n!nvidia-smi -L || true\n"),
        code("!pip -q install onnx onnxruntime\n"),
        code("%%writefile alfamedomega_train.py\n" + script_text),
    ]
    if inputs:
        cells.append(md("Upload these to Drive first and fix the paths in the next cell if they "
                        "differ: " + ", ".join(f"`{Path(x).name}`" for x in inputs) + "\n"))
    cells += [
        code(f"!python alfamedomega_train.py {train}\n"),
        code(f"!python alfamedomega_train.py {export}\n"),
        code(f"!cd {_q(run)} && zip -qr export.zip export && ls -la export.zip\n"),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}


def write_colab_notebook(spec: TrainSpec, script: str, dest: Path) -> Path:
    text = Path(script).read_text(encoding="utf-8")
    dest = Path(dest)
    if dest.is_dir():
        dest = dest / f"alfamedomega-{spec.run_name()}.ipynb"
    dest.write_text(json.dumps(colab_notebook(spec, text), indent=1), encoding="utf-8")
    return dest


def _q(s) -> str:
    s = str(s)
    return s if s and all(c.isalnum() or c in "-_./=:+" for c in s) else "'" + s.replace("'", "'\\''") + "'"
