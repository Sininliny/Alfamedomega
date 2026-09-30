"""alfa-cc: every control center action from a terminal."""

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

from . import __version__
from .aom import train_tensor_model, train_runtime_tensor_model, infer_tensor_preset, import_tensor_wav
from .assimilate.harvest import create_assimilation_job, create_scan_job, host_formats
from .assimilate.plans import DENSITIES, custom_density, estimate
from .context import Context
from .jobs import SUCCEEDED
from .models import ModelStoreError, model_bytes
from .preference import CATEGORIES, create_batch_job, reset_learning
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
            print(f"  {m['folder']:<20} {_mb(model_bytes(m)):>10}  {len(m['files'])} files  "
                  f"installed {m.get('installed_at', '?')}  {m.get('notes', '')}")
    elif a.action == "verify":
        bad = 0
        for r in store.verify(a.id):
            print(f"  {r.model_id:<20} {'ok' if r.ok else 'FAILED: ' + '; '.join(r.problems)}")
            bad += not r.ok
        return 1 if bad else 0
    elif a.action == "import":
        for e in store.import_any(Path(a.path), version=a.version, notes=a.notes or "", replace=a.replace):
            print(f"installed {e['folder']} ({_mb(e['bytes'])}) into {store.root}")
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
    elif a.action == "learn":
        cats = {}
        for item in a.category or []:
            name, _, value = item.partition("=")
            cats[name] = value.lower() in ("on", "true", "1", "yes")
        learn = None if a.state is None else a.state == "on"
        c = p.set_consent(learn, cats, a.personalization) if (learn is not None or cats or
                                                              a.personalization is not None) else p.consent()
        print(f"learn from how I play: {'on' if c['learn'] else 'off'}")
        for name, on in c["categories"].items():
            print(f"  [{'x' if on else ' '}] {name:<12} {CATEGORIES[name]}")
        print(f"personalization: {c['personalization']:.2f}")
    elif a.action == "explain":
        lines = p.explain()
        print("\n".join(lines) if lines else "Nothing learned yet. Run 'alfa-cc pref batch'.")
    elif a.action == "reset":
        print(reset_learning(p, ctx.settings.resolved_preference_learner(), a.all))
    elif a.action == "label":
        split = (lambda v: None if v is None else [w for w in v.split(",")])
        entry = p.annotate(Path(a.path), split(a.tags), a.description, split(a.labels))
        print(json.dumps(entry, ensure_ascii=False) if entry else f"cleared {a.path}")
    return 0


def _pick_plugin(ctx: Context, query: str) -> dict:
    """A scanned instrument by uid, or by a name that matches only one."""
    plugins = ctx.assimilation.instruments()
    exact = [p for p in plugins if p["uid"] == query]
    hits = exact or [p for p in plugins if query.lower() in p["name"].lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no scanned instrument matches {query!r}; run 'alfa-cc plugins scan' first")
    raise ValueError(f"{query!r} matches {len(hits)} plugins: " + ", ".join(f"{p['name']} ({p['format']})"
                                                                         for p in hits))


def _density(a):
    if a.step is None and not a.velocities and not a.durations:
        return DENSITIES[a.density]
    vel = [int(v) for v in a.velocities.split(",")] if a.velocities else None
    dur = [float(d) for d in a.durations.split(",")] if a.durations else None
    return custom_density(a.density, a.step, vel, dur)


def cmd_plugins(ctx: Context, a) -> int:
    store = ctx.assimilation
    if a.action == "discover":
        formats = host_formats(ctx.render_host) if ctx.render_host else ["VST3"]
        cat = store.discover(formats, ctx.plugin_folders())
        for f in cat["files"]:
            mark = "ok " if f["loadable"] else "-- "
            note = f"  ({f['reason']}{'; twin: ' + f['twin'] if f['twin'] else ''})" if f["reason"] else ""
            print(f"  {mark}{f['format']:<5} {f['name']:<32} {f['path']}{note}")
        print(f"{sum(f['loadable'] for f in cat['files'])} of {len(cat['files'])} plugin files can be opened")
    elif a.action == "scan":
        files = a.files
        if not files:
            cat = store.catalog() if store.catalog()["files"] else store.discover(
                host_formats(ctx.render_host) if ctx.render_host else ["VST3"], ctx.plugin_folders())
            files = [f["path"] for f in cat["files"] if f["loadable"]]
        return _run_job(ctx, create_scan_job(ctx.jobs, ctx.home, ctx.render_host, files), a.quiet)
    elif a.action == "list":
        for row in store.summary():
            print(f"  {row['name']:<28} {row['format']:<5} {row['vendor']:<20} "
                  f"{row['harvested']}/{row['presets']} presets, {row['renders']} renders   {row['uid']}")
    elif a.action == "presets":
        plugin = _pick_plugin(ctx, a.plugin)
        res = store.results(plugin)
        for ref in store.presets_for(plugin):
            r = res.get(ref.key, {})
            rng = r.get("range") or r.get("status", "")
            print(f"  {ref.key:<8} {ref.name:<36} {r.get('status', 'new'):<10} {rng}")
    elif a.action == "estimate":
        plugin = _pick_plugin(ctx, a.plugin)
        n = len(a.presets.split(",")) if a.presets else len(store.presets_for(plugin))
        for d in ([_density(a)] if a.density_given else DENSITIES.values()):
            print(f"  {d.label:<13} {estimate(n, d).text()}")
    elif a.action == "assimilate":
        plugin = _pick_plugin(ctx, a.plugin)
        job = create_assimilation_job(ctx.jobs, ctx.home, plugin["uid"], _density(a), ctx.render_host,
                                      ctx.python, ctx.settings.resolved_descriptor(),
                                      a.presets.split(",") if a.presets else None, a.keep_audio)
        return _run_job(ctx, job, a.quiet)
    elif a.action == "show":
        plugin = _pick_plugin(ctx, a.plugin)
        for rec in store.features(plugin, a.preset):
            rec.pop("descriptor", None)
            print(json.dumps(rec))
    elif a.action == "forget":
        store.forget(_pick_plugin(ctx, a.plugin) if a.plugin else None)
        print("deleted what was learned from " + (a.plugin or "all plugins"))
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


def cmd_aom(ctx, a) -> int:
    if a.action == "train":
        result = train_tensor_model(a.repository, a.manifest, a.model)
    elif a.action == "train-runtime":
        result = train_runtime_tensor_model(a.repository, a.manifest, a.model)
    elif a.action == "infer":
        result = infer_tensor_preset(a.repository, a.model, a.wav, a.output, mask_threshold=a.mask_threshold)
    else:
        result = import_tensor_wav(a.repository, a.model, a.wav, a.template, a.bounds, a.output,
                                  head=a.head, mask_threshold=a.mask_threshold, sample_rate=a.sample_rate,
                                  dll_path=a.dll)
    print(json.dumps(result, indent=2))
    return 0

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alfa-cc", description="Alfamedomega control center")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--home", help="control center data folder (default: per-user app data)")
    ap.add_argument("--models-dir", help="Models folder (default: the shared system folder)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    aom = sub.add_parser("aom", help="tensor compiler training, native runtime models and editable WAV import")
    aom.add_argument("--repository", type=Path, required=True, help="repository containing the tensor learner and native runtime")
    actions = aom.add_subparsers(dest="action", required=True)
    for action, help_text in (("train", "dual-supervised offline WAV tensor learning"),
                              ("train-runtime", "train recurrent runtime tensor compiler and export native GRU weights")):
        command = actions.add_parser(action, help=help_text)
        command.add_argument("manifest", type=Path)
        command.add_argument("model", type=Path)
    inference = actions.add_parser("infer", help="inspect inferred canonical and rendered tensor targets")
    inference.add_argument("model", type=Path)
    inference.add_argument("wav", type=Path)
    inference.add_argument("output", type=Path)
    inference.add_argument("--mask-threshold", type=float, required=True)
    importing = actions.add_parser("import-wav", help="infer and compile an editable AOM preset")
    for name in ("model", "wav", "template", "bounds", "output"):
        importing.add_argument(name, type=Path)
    importing.add_argument("--head", choices=("primary_canonical", "secondary_rendered_instance"), required=True)
    importing.add_argument("--mask-threshold", type=float, required=True)
    importing.add_argument("--sample-rate", type=float, required=True)
    importing.add_argument("--dll", type=Path)
    aom.set_defaults(fn=cmd_aom)

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

    t = sub.add_parser("train", help="legacy method-vector training, or write its Colab notebook")
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

    x = sub.add_parser("export", help="legacy method-vector ONNX export and installation")
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
    pw = psub.add_parser("learn", help="show or set what the plugin may learn from how you play")
    pw.add_argument("state", nargs="?", choices=["on", "off"])
    pw.add_argument("--category", action="append", metavar="NAME=on|off",
                    help=f"one of {', '.join(CATEGORIES)}; repeatable")
    pw.add_argument("--personalization", type=float, help="0 (plain model) to 1 (your taste)")
    psub.add_parser("explain", help="what the latest profile learned, in plain words")
    px = psub.add_parser("reset", help="forget the learned model and the behaviour log")
    px.add_argument("--all", action="store_true", help="also drop every example")
    pt = psub.add_parser("label", help="tag, describe or hand-label a file")
    pt.add_argument("path")
    pt.add_argument("--tags", help="comma-separated; empty clears")
    pt.add_argument("--labels", help="comma-separated hand labels; empty clears")
    pt.add_argument("--description")
    p.set_defaults(fn=cmd_pref)

    g = sub.add_parser("plugins", help="Local Plugin Assimilation: learn from your own instruments")
    gsub = g.add_subparsers(dest="action", required=True)
    gsub.add_parser("discover", help="find installed plugin files and say which can be opened")
    gs = gsub.add_parser("scan", help="open plugin files (all loadable ones by default)")
    gs.add_argument("files", nargs="*")
    gs.add_argument("--quiet", action="store_true")
    gsub.add_parser("list", help="scanned instruments and what was learned from them")
    gp = gsub.add_parser("presets")
    gp.add_argument("plugin", help="uid or part of the name")

    def density_args(parser):
        parser.add_argument("--density", choices=list(DENSITIES), default="normal")
        parser.add_argument("--step", type=int, help="semitones between keys (overrides the density)")
        parser.add_argument("--velocities", help="comma-separated, e.g. 40,80,120")
        parser.add_argument("--durations", help="held seconds, comma-separated, e.g. 0.25,1,4")
        parser.add_argument("--presets", help="preset keys from 'plugins presets', comma-separated")

    ge = gsub.add_parser("estimate", help="how many renders each density means")
    ge.add_argument("plugin")
    density_args(ge)
    ga = gsub.add_parser("assimilate", help="find ranges, render and analyse")
    ga.add_argument("plugin")
    density_args(ga)
    ga.add_argument("--keep-audio", action="store_true", help="keep the rendered WAV files")
    ga.add_argument("--quiet", action="store_true")
    gw = gsub.add_parser("show", help="print the analysed notes (JSON lines)")
    gw.add_argument("plugin")
    gw.add_argument("--preset")
    gf = gsub.add_parser("forget", help="delete what was learned from one plugin, or all")
    gf.add_argument("plugin", nargs="?")
    g.set_defaults(fn=cmd_plugins)

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
    argv = sys.argv[1:] if argv is None else list(argv)
    a = build_parser().parse_args(argv)
    a.density_given = "--density" in argv
    ctx = None if a.cmd == "aom" else Context(a.home, a.models_dir)
    try:
        return a.fn(ctx, a)
    except (ModelStoreError, ValueError, FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
