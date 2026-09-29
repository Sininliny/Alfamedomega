"""alfa-cc: every control center action from a terminal."""

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

from . import __version__
from .context import Context
from .jobs import SUCCEEDED
from .models import ModelStoreError
from .preference import create_batch_job
from .settings import Settings
from .tiers import STAGES, TIERS
from .training import (TrainSpec, check_environment, create_export_job, create_training_job,
                       write_colab_notebook)


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _run_job(ctx: Context, job: dict, quiet: bool = False) -> int:
    print(f"job {job['id']}: {job['title']}", file=sys.stderr)
    done = ctx.jobs.run(job["id"], echo=None if quiet else print)
    if done["status"] != SUCCEEDED:
        print(f"job {done['status']}: {done.get('error') or ''}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------- commands

def cmd_info(ctx: Context, a) -> int:
    env = check_environment(ctx.python, ctx.script)
    print(json.dumps({"version": __version__, "home": str(ctx.home), "models_dir": str(ctx.models_dir),
                      "runs_dir": str(ctx.runs_dir), "training": env,
                      "preference_learner": ctx.settings.resolved_preference_learner() or None}, indent=2))
    return 0


def cmd_settings(ctx: Context, a) -> int:
    known = {f.name for f in fields(Settings) if f.name != "extra"}
    for item in a.set or []:
        key, _, value = item.partition("=")
        if key not in known:
            print(f"unknown setting {key!r}; known: {', '.join(sorted(known))}", file=sys.stderr)
            return 2
        setattr(ctx.settings, key, value)
    if a.set:
        ctx.save_settings()
    print(json.dumps(asdict(ctx.settings), indent=2))
    return 0


def cmd_models(ctx: Context, a) -> int:
    store = ctx.store
    if a.action == "list":
        man = store.manifest()
        print(f"Models folder: {store.root}   default tier: {man['default_tier']}")
        if not man["models"]:
            print("  (no models installed)")
        for m in man["models"]:
            print(f"  {m['id']:<20} {_mb(m['bytes']):>10}  {len(m['files'])} files  "
                  f"installed {m['installed_at']}  {m.get('notes', '')}")
    elif a.action == "verify":
        bad = 0
        for r in store.verify(a.id):
            print(f"  {r.model_id:<20} {'ok' if r.ok else 'FAILED: ' + '; '.join(r.problems)}")
            bad += not r.ok
        return 1 if bad else 0
    elif a.action == "import":
        e = store.import_any(Path(a.path), version=a.version, notes=a.notes or "", replace=a.replace)
        print(f"installed {e['id']} ({_mb(e['bytes'])}) into {store.root}")
    elif a.action == "export":
        print(f"wrote {store.export_bundle(a.id, Path(a.dest))}")
    elif a.action == "remove":
        store.remove(a.id)
        print(f"removed {a.id}")
    elif a.action == "default":
        store.set_default_tier(a.tier)
        print(f"default tier is now {a.tier}")
    return 0


def _spec(a) -> TrainSpec:
    return TrainSpec(tier=a.tier, stage=a.stage, steps=a.steps, batch=a.batch, accum=a.accum,
                     clip=a.clip, lr=a.lr, init=a.init or "", teacher=a.teacher or "",
                     data=a.data or "", seed=a.seed, device=a.device or "", smoke=a.smoke,
                     export=not a.no_export, install=not (a.no_install or a.no_export),
                     version=a.version or "", int8=a.int8)


def cmd_train(ctx: Context, a) -> int:
    spec = _spec(a)
    if a.colab:
        spec.validate()
        if not ctx.script:
            print("training script not found; set training_script", file=sys.stderr)
            return 2
        print(f"wrote {write_colab_notebook(spec, ctx.script, Path(a.colab))}")
        return 0
    job = create_training_job(ctx.jobs, spec, ctx.python, ctx.script, ctx.runs_dir, ctx.models_dir)
    return _run_job(ctx, job, a.quiet)


def cmd_export(ctx: Context, a) -> int:
    job = create_export_job(ctx.jobs, a.tier, a.ckpt or "", ctx.python, ctx.script, ctx.runs_dir,
                            ctx.models_dir, version=a.version or "", int8=a.int8,
                            install=not a.no_install)
    return _run_job(ctx, job, a.quiet)


def cmd_pref(ctx: Context, a) -> int:
    p = ctx.prefs
    if a.action == "add":
        lib = p.add_library(Path(a.path), a.label or "")
        print(f"added {lib['path']}")
    elif a.action == "remove":
        p.remove_library(Path(a.path))
        print(f"removed {a.path}")
    elif a.action == "list":
        for lib in p.libraries():
            print(f"  [{'x' if lib['enabled'] else ' '}] {lib['label']:<20} {lib['path']}")
        voc = p.vocabulary()
        if voc["summary"]:
            s = voc["summary"]
            print(f"index: {s['files']} files, {s['labels']} labels, scanned {s['scanned_at']}")
            top = list(voc["labels"].items())[: a.top]
            print("top labels: " + ", ".join(f"{k} ({v})" for k, v in top))
    elif a.action == "scan":
        p.scan(print)
    elif a.action == "batch":
        job = create_batch_job(ctx.jobs, ctx.home, ctx.settings.resolved_preference_learner())
        return _run_job(ctx, job, a.quiet)
    elif a.action == "forget":
        p.forget()
        print("deleted the library index, vocabulary, behaviour log and learned profiles")
    return 0


def cmd_jobs(ctx: Context, a) -> int:
    if a.action == "list":
        for j in ctx.jobs.list()[: a.limit]:
            prog = f"{j['progress'][0]}/{j['progress'][1]}" if j.get("progress") else ""
            print(f"  {j['id']:<36} {j['status']:<11} {prog:>12}  {j['title']}")
    elif a.action == "log":
        print(ctx.jobs.read_log(a.id, tail=a.tail))
    elif a.action == "show":
        print(json.dumps(ctx.jobs.get(a.id), indent=2))
    return 0


# ---------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alfa-cc", description="Alfamedomega control center")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--home", help="control center data folder (default: per-user app data)")
    ap.add_argument("--models-dir", help="Models folder (default: the shared system folder)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="paths and whether training can run here").set_defaults(fn=cmd_info)
    s = sub.add_parser("settings", help="show or change settings")
    s.add_argument("--set", action="append", metavar="KEY=VALUE")
    s.set_defaults(fn=cmd_settings)

    m = sub.add_parser("models", help="installed model weights")
    msub = m.add_subparsers(dest="action", required=True)
    msub.add_parser("list")
    v = msub.add_parser("verify")
    v.add_argument("id", nargs="?")
    i = msub.add_parser("import", help="install a .alfamodel bundle, an export folder or a zipped export")
    i.add_argument("path")
    i.add_argument("--version", help="MAJOR.MINOR.PATCH (default: next patch for the tier)")
    i.add_argument("--notes")
    i.add_argument("--replace", action="store_true")
    e = msub.add_parser("export", help="pack an installed model into a .alfamodel bundle")
    e.add_argument("id")
    e.add_argument("dest")
    r = msub.add_parser("remove")
    r.add_argument("id")
    d = msub.add_parser("default")
    d.add_argument("tier", choices=list(TIERS))
    m.set_defaults(fn=cmd_models)

    t = sub.add_parser("train", help="train a tier locally, or write a Colab notebook")
    t.add_argument("--tier", default="lite", choices=list(TIERS))
    t.add_argument("--stage", default="pretrain", choices=STAGES)
    t.add_argument("--steps", type=int, default=200_000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--lr", type=float)
    t.add_argument("--init")
    t.add_argument("--teacher")
    t.add_argument("--data")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device")
    t.add_argument("--smoke", action="store_true", help="tiny CPU run to test the pipeline")
    t.add_argument("--no-export", action="store_true")
    t.add_argument("--no-install", action="store_true")
    t.add_argument("--version")
    t.add_argument("--int8", action="store_true")
    t.add_argument("--colab", metavar="NOTEBOOK", help="write a Colab notebook instead of training here")
    t.add_argument("--quiet", action="store_true")
    t.set_defaults(fn=cmd_train)

    x = sub.add_parser("export", help="export a checkpoint to ONNX and install it")
    x.add_argument("--tier", required=True, choices=list(TIERS))
    x.add_argument("--ckpt", help="omit to export an untrained model (pipeline test)")
    x.add_argument("--version")
    x.add_argument("--int8", action="store_true")
    x.add_argument("--no-install", action="store_true")
    x.add_argument("--quiet", action="store_true")
    x.set_defaults(fn=cmd_export)

    p = sub.add_parser("pref", help="Adaptive Preference Learning libraries and batches")
    psub = p.add_subparsers(dest="action", required=True)
    pa = psub.add_parser("add")
    pa.add_argument("path")
    pa.add_argument("--label")
    pr = psub.add_parser("remove")
    pr.add_argument("path")
    pl = psub.add_parser("list")
    pl.add_argument("--top", type=int, default=20)
    psub.add_parser("scan")
    pb = psub.add_parser("batch", help="index the libraries and run the preference learner")
    pb.add_argument("--quiet", action="store_true")
    psub.add_parser("forget", help="delete everything learned from your libraries")
    p.set_defaults(fn=cmd_pref)

    j = sub.add_parser("jobs", help="job history")
    jsub = j.add_subparsers(dest="action", required=True)
    jl = jsub.add_parser("list")
    jl.add_argument("--limit", type=int, default=20)
    jg = jsub.add_parser("log")
    jg.add_argument("id")
    jg.add_argument("--tail", type=int)
    js = jsub.add_parser("show")
    js.add_argument("id")
    j.set_defaults(fn=cmd_jobs)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    ctx = Context(a.home, a.models_dir)
    try:
        return a.fn(ctx, a)
    except (ModelStoreError, ValueError, FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
