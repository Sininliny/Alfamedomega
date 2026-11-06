#!/usr/bin/env python3
"""Write self-contained Kaggle kernels for corpus generation and encoder training.

Each kernel embeds a compressed copy of the code it needs (the native tensor
library sources and training/amo), so no GitHub access or code dataset is
required on Kaggle. Run from the repository root:

    python training/amo/corpus/kaggle/make_kernels.py --user KAGGLE_USERNAME --out build/kaggle

This writes three folders, each a valid `kaggle kernels push -p DIR` target:

- ``generate``: CPU kernel, internet on. Builds libalfa_tensor_c, generates the
  corpus with ``--no-wav``, validates it and outputs ``corpus.tar``.
- ``train-a`` / ``train-b``: GPU kernels that train the corpus encoder and
  output ``run/`` (last.pt, best.pt, model.pt, metrics.jsonl, train.log). They
  alternate: each mounts the other's latest output and resumes from the newest
  ``last.pt``, so an interrupted or time-limited run continues where it stopped.
  ``kaggle_push.py`` sets the sources for each push.
- ``sensitivity``: CPU kernel, internet on (builds libalfa_tensor_c). Mounts the
  generate output and measures how audible each (family, coordinate) is
  (``sensitivity.py``); outputs ``sensitivity.npz``. With ``--perceptual`` the
  train kernels mount it and pass ``--perceptual-weights`` to the trainer.
- ``evaluate``: GPU kernel, internet on (builds libalfa_tensor_c). Scores the newest
  mounted ``run/model.pt`` on the test and validation splits and renders the
  predicted constructions of ``--resynth`` test examples per family.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[4]
BUNDLE = ["CMakeLists.txt", "CMakePresets.json", "cmake", "src/dsp", "src/inference",
          "src/tensor", "training/amo"]
EXCLUDE = {"__pycache__", "reports"}
GENERATE_SLUG = "amo-corpus-generate"
TRAIN_SLUGS = ("amo-corpus-train-a", "amo-corpus-train-b")
EVALUATE_SLUG = "amo-corpus-evaluate"
SENSITIVITY_SLUG = "amo-corpus-sensitivity"


def bundle():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as archive:
        for entry in BUNDLE:
            def keep(info):
                return None if set(Path(info.name).parts) & EXCLUDE else info
            archive.add(ROOT / entry, arcname=entry, filter=keep)
    return base64.b64encode(buffer.getvalue()).decode()


def commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *BUNDLE], cwd=ROOT, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


BOOTSTRAP = r'''
import base64, io, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

SOURCE_COMMIT = {commit!r}
CODE = Path("/tmp/amo-src")


def run(command, **kwargs):
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def unpack():
    if CODE.exists():
        shutil.rmtree(CODE)
    CODE.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as archive:
        archive.extractall(CODE)
    os.environ["AMO_SOURCE_COMMIT"] = SOURCE_COMMIT
    os.environ["PYTHONPATH"] = str(CODE) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONUNBUFFERED"] = "1"
    print(f"code {{SOURCE_COMMIT}} unpacked to {{CODE}}", flush=True)
'''

GENERATE = r'''
CANONICAL_PER_TYPE = {canonical_per_type}


def build_library():
    try:
        version = subprocess.check_output(["cmake", "--version"], text=True).split()[2]
        ok = tuple(map(int, version.split(".")[:2])) >= (3, 25)
    except (OSError, subprocess.CalledProcessError):
        ok = False
    if not ok or shutil.which("ninja") is None:
        run([sys.executable, "-m", "pip", "install", "-q", "cmake>=3.25", "ninja"])
    build = Path("/tmp/amo-build")
    run(["cmake", "-S", CODE, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DALFA_BUILD_PLUGIN=OFF", "-DALFA_BUILD_TESTS=OFF", "-DCPM_SOURCE_CACHE=/tmp/cpm"])
    run(["cmake", "--build", build, "--target", "alfa_tensor_c", "--parallel", str(os.cpu_count())])
    library = build / "src/tensor/libalfa_tensor_c.so"
    os.environ["AMO_TENSOR_LIBRARY"] = str(library)
    return library


def main():
    began = time.time()
    unpack()
    build_library()
    corpus = Path("/tmp/corpus/corpus-v3")
    workers = os.cpu_count() or 4
    run([sys.executable, "-m", "training.amo.corpus.prepare", "build", corpus,
         "--canonical-per-type", CANONICAL_PER_TYPE, "--workers", workers, "--no-wav"], cwd=CODE)
    run([sys.executable, "-m", "training.amo.corpus.prepare", "validate", corpus], cwd=CODE)
    output = Path("/kaggle/working")
    with tarfile.open(output / "corpus.tar", "w") as archive:
        archive.add(corpus, arcname="corpus-v3")
    for name in ("report.json", "verification.json"):
        shutil.copy(corpus / name, output / name)
    size = (output / "corpus.tar").stat().st_size / 2**30
    print(f"corpus.tar {{size:.2f}} GiB written in {{(time.time() - began) / 60:.1f}} min", flush=True)


main()
'''

TRAIN = r'''
TRAIN_ARGS = {train_args!r}
PERCEPTUAL = {perceptual!r}


def find_corpus():
    for archive in sorted(Path("/kaggle/input").rglob("corpus.tar")):
        target = Path("/tmp/corpus")
        if not (target / "corpus-v3" / "shards.json").exists():
            target.mkdir(parents=True, exist_ok=True)
            print(f"extracting {{archive}}", flush=True)
            with tarfile.open(archive) as stream:
                stream.extractall(target)
        return target / "corpus-v3"
    raise SystemExit("corpus.tar not found under /kaggle/input: add the amo-corpus-generate output")


def main():
    unpack()
    corpus = find_corpus()
    run(["nvidia-smi"])
    output = Path("/kaggle/working/run")
    extra = []
    if PERCEPTUAL:
        found = sorted(Path("/kaggle/input").rglob("sensitivity.npz"))
        if not found:
            raise SystemExit("sensitivity.npz not found under /kaggle/input: add the amo-corpus-sensitivity output")
        extra = ["--perceptual-weights", found[0]]
    command = [sys.executable, "-m", "training.amo.corpus.train", corpus, output,
               "--resume", "auto", "--resume-search", "/kaggle/input", *extra, *TRAIN_ARGS]
    print("$", " ".join(map(str, command)), flush=True)
    code = subprocess.run(list(map(str, command)), cwd=CODE).returncode
    # Exit code 2 = stopped by the time budget with last.pt saved: a normal partial run.
    print({{0: "training finished; run/model.pt exported",
           2: "stopped early with run/last.pt saved; push the other train kernel to continue"}}
          .get(code, f"training failed with exit code {{code}}"), flush=True)
    if code not in (0, 2):
        raise SystemExit(code)


main()
'''

SENSITIVITY = r'''
def main():
    unpack()
    corpus = find_corpus()
    build_library()
    run([sys.executable, "-m", "training.amo.corpus.sensitivity", corpus, "/kaggle/working/sensitivity.npz",
         "--per-type", SENSITIVITY_PER_TYPE, "--workers", os.cpu_count() or 4], cwd=CODE)


main()
'''

EVALUATE = r'''
def main():
    unpack()
    corpus = find_corpus()
    models = sorted(Path("/kaggle/input").rglob("run/model.pt"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise SystemExit("run/model.pt not found under /kaggle/input: add a finished train kernel output")
    build_library()
    output = Path("/kaggle/working")
    run([sys.executable, "-m", "training.amo.corpus.evaluate", corpus, models[-1], "--split", "test",
         "--resynth", RESYNTH, "--out", output / "evaluation-test.json"], cwd=CODE)
    run([sys.executable, "-m", "training.amo.corpus.evaluate", corpus, models[-1], "--split", "validation",
         "--out", output / "evaluation-validation.json"], cwd=CODE)


main()
'''


def write_kernel(directory, slug, title, user, source, *, gpu, internet, kernel_sources=()):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(source, encoding="utf-8")
    metadata = {
        "id": f"{user}/{slug}", "title": title, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": internet, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": list(kernel_sources), "model_sources": [],
    }
    if gpu:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="Kaggle username (owner of the kernels)")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "kaggle")
    parser.add_argument("--canonical-per-type", type=int, default=512)
    parser.add_argument("--time-budget-hours", type=float, default=11.0,
                        help="stop and save before Kaggle's 12 h session limit")
    parser.add_argument("--resynth", type=int, default=256,
                        help="evaluate kernel: render predictions for N test examples per family")
    parser.add_argument("--perceptual", action="store_true",
                        help="train kernels mount amo-corpus-sensitivity and weight the value loss by audibility")
    parser.add_argument("--sensitivity-per-type", type=int, default=64,
                        help="sensitivity kernel: training instances probed per family")
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="extra arguments for training.amo.corpus.train after --, e.g. -- --epochs 150")
    args = parser.parse_args()
    if "@" in args.user:
        parser.error("--user is the Kaggle handle shown in kaggle.com/<handle>, not the account e-mail")
    extra = [a for a in args.train_args if a != "--"]
    header = "BUNDLE = " + repr(bundle()) + "\n"
    common = BOOTSTRAP.format(commit=commit())
    write_kernel(args.out / "generate", GENERATE_SLUG, "AMO corpus generate", args.user,
                 header + common + GENERATE.format(canonical_per_type=args.canonical_per_type),
                 gpu=False, internet=True)
    train_args = ["--time-budget-hours", str(args.time_budget_hours), *extra]
    train_sources = [f"{args.user}/{GENERATE_SLUG}"]
    if args.perceptual:
        train_sources.append(f"{args.user}/{SENSITIVITY_SLUG}")
    for slug in TRAIN_SLUGS:
        write_kernel(args.out / f"train-{slug[-1]}", slug,
                     "AMO corpus train " + slug[-1].upper(), args.user,
                     header + common + TRAIN.format(train_args=train_args, perceptual=args.perceptual),
                     gpu=True, internet=False, kernel_sources=train_sources)
    # Evaluates the newest run/model.pt among the mounted train outputs; kaggle_push.py
    # `evaluate` mounts the train kernel that ran last.
    find_corpus = TRAIN[TRAIN.index("def find_corpus"):TRAIN.index("def main")]
    build_library = GENERATE[GENERATE.index("def build_library"):GENERATE.index("def main")]
    write_kernel(args.out / "sensitivity", SENSITIVITY_SLUG, "AMO corpus sensitivity", args.user,
                 header + common + f"SENSITIVITY_PER_TYPE = {args.sensitivity_per_type}\n"
                 + (find_corpus + build_library + SENSITIVITY).format(), gpu=False, internet=True,
                 kernel_sources=[f"{args.user}/{GENERATE_SLUG}"])
    write_kernel(args.out / "evaluate", EVALUATE_SLUG, "AMO corpus evaluate", args.user,
                 header + common + f"RESYNTH = {args.resynth}\n"
                 + (find_corpus + build_library + EVALUATE).format(), gpu=True, internet=True,
                 kernel_sources=[f"{args.user}/{GENERATE_SLUG}"])
    size = len(header) / 1024
    print(f"wrote {args.out} (embedded code {size:.0f} KiB, commit {commit()})")


if __name__ == "__main__":
    main()
