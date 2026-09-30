"""Scan plugins, find each preset's playable range, render it across that range, analyse it.

Everything lives under <home>/assimilation/:

    catalog.json                 plugin files found, what the render host saw inside them
    <plugin-slug>/
      plugin.json                the plugin as the host reported it, and its presets
      presets.json               per preset: range found, pitched or not, status, counts
      features.jsonl             one line per rendered note (see analyze_worker.py)
      work/                      render jobs in progress; audio is deleted after analysis
                                 unless "keep audio" is on

A plugin is loaded only by the render host, one process per call, so a crash or hang in a
third-party plugin fails that call (and is recorded), never the control center.
"""

import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import jobs
from ..settings import write_json_atomic
from .discovery import discover, find_preset_files
from .plans import (DENSITIES, Density, PresetRef, grid_notes, note_name, probe_notes,
                    refine_notes, PROBE_HIGH, PROBE_LOW, PROBE_STEP)

WORKER = Path(__file__).with_name("analyze_worker.py")
SCAN_TIMEOUT_S = 90
QUIET_DB = 40.0          # a probe this far below the preset's loudest probe is "too quiet"
AUDIBLE_DB = -60.0
PITCHED_SHARE = 0.6      # a preset is pitched if this share of its audible probes track pitch


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "plugin"


def find_render_host(setting: str = "") -> str:
    """The render host binary: Settings, $ALFA_RENDER_HOST, next to a packaged app, or PATH."""
    if setting:
        return setting
    if env := os.environ.get("ALFA_RENDER_HOST"):
        return env
    name = "alfa-render-host.exe" if sys.platform == "win32" else "alfa-render-host"
    if getattr(sys, "frozen", False):
        bundled = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / name
        if bundled.is_file():
            return str(bundled)
    return shutil.which(name) or ""


class _Collect:
    """A log callback that keeps a command's output instead of printing it."""

    def __init__(self, forward=None):
        self.lines, self.forward = [], forward

    def __call__(self, line: str) -> None:
        line = line.rstrip("\n")
        if line.startswith("$ "):
            return
        self.lines.append(line)
        if self.forward and (line.startswith("warning") or line.startswith("error")):
            self.forward(line)

    def json(self):
        text = "\n".join(self.lines)
        start = text.find("{") if text.find("{") >= 0 else text.find("[")
        return json.loads(text[start:])


def host_formats(host: str) -> list:
    out = _Collect()
    jobs.spawn([host, "formats"], out, timeout=30, label="render host")
    return out.json()


class AssimilationStore:
    def __init__(self, home: Path):
        self.dir = Path(home) / "assimilation"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ catalog

    @property
    def catalog_path(self) -> Path:
        return self.dir / "catalog.json"

    def catalog(self) -> dict:
        if self.catalog_path.exists():
            return json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return {"files": [], "plugins": {}, "discovered_at": None}

    def save_catalog(self, cat: dict) -> None:
        write_json_atomic(self.catalog_path, cat)

    def discover(self, host_formats_list, extra_folders=(), folders=None) -> dict:
        cat = self.catalog()
        found = discover(extra_folders, host_formats_list, folders=folders)
        cat["files"] = [p.to_dict() for p in found]
        cat["discovered_at"] = _now()
        self.save_catalog(cat)
        return cat

    def instruments(self) -> list:
        """Scanned instrument plugins, one per uid."""
        return [p for p in self.catalog()["plugins"].values() if p.get("is_instrument")]

    def plugin_dir(self, plugin: dict) -> Path:
        return self.dir / slug(f"{plugin.get('vendor', '')}-{plugin['name']}-{plugin['format']}")

    # ------------------------------------------------------------------ presets and results

    def presets_for(self, plugin: dict) -> list:
        refs = [PresetRef(f"p{i:04d}", "program", name, index=i)
                for i, name in enumerate(plugin.get("programs") or [])]
        if plugin.get("format") == "VST3":
            for i, f in enumerate(find_preset_files(plugin.get("vendor", ""), plugin["name"])):
                refs.append(PresetRef(f"f{i:05d}", "vstpreset", f["name"], path=f["path"],
                                      category=f["category"]))
        # Programs named "Default" or "Init" alone say nothing more than the default state.
        if not refs or all(r.name.strip().lower() in ("", "default", "init") for r in refs):
            refs = [PresetRef("default", "default", "Default state")]
        return refs

    def results(self, plugin: dict) -> dict:
        p = self.plugin_dir(plugin) / "presets.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def features(self, plugin: dict, preset: str | None = None) -> list:
        path = self.plugin_dir(plugin) / "features.jsonl"
        if not path.exists():
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if preset is None or rec["preset"] == preset:
                    out.append(rec)
        return out

    def summary(self) -> list:
        rows = []
        for plugin in self.instruments():
            res = self.results(plugin)
            done = [r for r in res.values() if r.get("status") == "harvested"]
            rows.append({"name": plugin["name"], "vendor": plugin.get("vendor", ""),
                         "format": plugin["format"], "uid": plugin["uid"],
                         "presets": len(self.presets_for(plugin)), "harvested": len(done),
                         "renders": sum(r.get("renders", 0) for r in done)})
        return rows

    def forget(self, plugin: dict | None = None) -> None:
        """Delete what was learned from one plugin, or from all of them (the catalog stays)."""
        if plugin is not None:
            shutil.rmtree(self.plugin_dir(plugin), ignore_errors=True)
            return
        for d in self.dir.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------- range finding

def decide_range(records: list) -> dict:
    """Playable range of one preset from its probe (and refine) analyses.

    A key is usable if it is audible, within QUIET_DB of the preset's loudest key, and, for a
    pitched preset, sounds at the pitch it was asked for (octave shifts allowed). The range is
    the longest unbroken run of usable keys.
    """
    recs = sorted((r for r in records if "error" not in r), key=lambda r: r["midi"])
    audible = [r for r in recs if r.get("audible") and r["peak_db"] > AUDIBLE_DB]
    if not audible:
        return {"status": "silent", "pitched": False, "low": None, "high": None}
    loudest = max(r["peak_db"] for r in audible)
    pitched = sum(1 for r in audible if r.get("pitched")) >= PITCHED_SHARE * len(audible)

    def usable(r):
        ok = r.get("audible") and r["peak_db"] > max(AUDIBLE_DB, loudest - QUIET_DB)
        return bool(ok and (r.get("pitched") or not pitched))

    best, run = [], []
    for r in recs:
        if usable(r) and (not run or r["midi"] - run[-1]["midi"] <= PROBE_STEP):
            run.append(r)
        elif usable(r):
            run = [r]
        else:
            run = []
        if len(run) > len(best):
            best = list(run)
    if not best:
        return {"status": "unusable", "pitched": pitched, "low": None, "high": None}
    octaves = [r.get("octave", 0) for r in best if r.get("pitched")]
    return {"status": "ranged", "pitched": pitched, "low": best[0]["midi"], "high": best[-1]["midi"],
            "octave_shift": max(set(octaves), key=octaves.count) if octaves else 0}


def refine_keys(rng: dict, step: int = PROBE_STEP) -> list:
    """Keys between the coarse range's ends and the next probes out, to find the exact edges."""
    if rng.get("status") != "ranged":
        return []
    lo, hi = rng["low"], rng["high"]
    # Stay inside the probed span: a preset still sounding at C0 or C9 gets no wider range.
    below = range(lo - step + 1, lo) if lo > PROBE_LOW else range(0)
    above = range(hi + 1, hi + step) if hi < PROBE_HIGH else range(0)
    return list(below) + list(above)


def extend_range(rng: dict, refined: list, all_records: list) -> dict:
    """Grow the coarse range key by key into the refined probes."""
    if rng.get("status") != "ranged":
        return rng
    by_midi = {r["midi"]: r for r in refined}
    loudest = max(r["peak_db"] for r in all_records if r.get("audible"))

    def usable(r):
        ok = r and r.get("audible") and r["peak_db"] > max(AUDIBLE_DB, loudest - QUIET_DB)
        return bool(ok and (r.get("pitched") or not rng["pitched"]))

    lo, hi = rng["low"], rng["high"]
    while lo - 1 in by_midi and usable(by_midi[lo - 1]):
        lo -= 1
    while hi + 1 in by_midi and usable(by_midi[hi + 1]):
        hi += 1
    return {**rng, "low": lo, "high": hi, "range": f"{note_name(lo)}–{note_name(hi)}"}


# ---------------------------------------------------------------------- render + analyse

def _render(host: str, plugin: dict, presets: list, notes: list, work: Path, log,
            progress_base: int, progress_total: int, sample_rate: int = 48000) -> dict:
    """One render host call: every note for every preset given. Returns renders.json."""
    work.mkdir(parents=True, exist_ok=True)
    job = {"plugin": plugin["file"], "uid": plugin["uid"], "sample_rate": sample_rate, "block": 512,
           "out_dir": str(work), "presets": [p.to_job() for p in presets],
           "notes": [n.to_dict() for n in notes]}
    job_path = work / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    audio_s = len(presets) * sum(n.dur_s + n.tail_s for n in notes)

    def relay(line: str) -> None:
        m = jobs.PROGRESS_RE.match(line)
        if m:
            log(f"progress {progress_base + int(m.group(1))}/{progress_total}")
        elif line.startswith(("warning", "error", "$ ")):
            log(line)

    jobs.spawn([host, "render", str(job_path)], relay, timeout=120 + 2 * audio_s,
               label=f"render {plugin['name']}")
    return json.loads((work / "renders.json").read_text(encoding="utf-8"))


def _analyse(python: str, descriptor: str, work: Path, notes: list, out: Path, log,
             delete_audio: bool) -> list:
    (work / "notes.json").write_text(json.dumps({n.id: n.to_dict() for n in notes}), encoding="utf-8")
    tmp = work / "features.jsonl"
    tmp.unlink(missing_ok=True)
    argv = [python, str(WORKER), "--renders", str(work / "renders.json"),
            "--notes", str(work / "notes.json"), "--out", str(tmp)]
    if descriptor:
        argv += ["--descriptor", descriptor]
    if delete_audio:
        argv.append("--delete-audio")
    quiet = _Collect(forward=log)
    jobs.spawn(argv, quiet, timeout=600 + 0.5 * len(notes), label="analyse renders")
    recs = [json.loads(line) for line in tmp.read_text(encoding="utf-8").splitlines() if line]
    if out is not None:
        with open(out, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return recs


def _plugin_and_presets(store: AssimilationStore, args: dict):
    plugin = store.catalog()["plugins"][args["uid"]]
    refs = store.presets_for(plugin)
    wanted = args.get("presets")
    if wanted:
        refs = [r for r in refs if r.key in set(wanted)]
    if not refs:
        raise ValueError("no presets selected")
    return plugin, refs


def _probe_pass(args, plugin, refs, notes, work, log, base, total):
    """Render and analyse probe notes for several presets in one host call.

    If the call fails (the plugin crashed or hung on some preset), each preset is retried in a
    call of its own so one bad preset doesn't cost the others. Returns (records by preset key,
    {failed key: error}).
    """
    shutil.rmtree(work, ignore_errors=True)
    try:
        _render(args["host"], plugin, refs, notes, work, log, base, total)
        groups = {"": work}
        failed = {}
    except RuntimeError as e:
        if len(refs) == 1:
            return {}, {refs[0].key: str(e)}
        log(f"the plugin failed during a shared render ({e}); retrying each preset on its own")
        groups, failed = {}, {}
        for i, ref in enumerate(refs):
            sub = work / f"one-{ref.key}"
            try:
                _render(args["host"], plugin, [ref], notes, sub, log, base + i * len(notes), total)
                groups[ref.key] = sub
            except RuntimeError as e2:
                failed[ref.key] = str(e2)
    by_preset = {}
    for sub in groups.values():
        for rec in _analyse(args["python"], "", sub, notes, None, log, True):
            by_preset.setdefault(rec["preset"], []).append(rec)
    return by_preset, failed


@jobs.action("asm_scan")
def _scan(args: dict, log) -> dict:
    """Open each chosen plugin file in its own render host process and record what is inside."""
    store = AssimilationStore(Path(args["home"]))
    cat = store.catalog()
    files = args["files"]
    ok = 0
    for i, path in enumerate(files, 1):
        out = _Collect(forward=log)
        try:
            jobs.spawn([args["host"], "scan", "--programs", path], out, timeout=SCAN_TIMEOUT_S,
                       label=f"scan {Path(path).name}")
            for p in out.json()["plugins"]:
                p["scanned_at"] = _now()
                p["status"] = "ok"
                cat["plugins"][p["uid"]] = p
                kind = "instrument" if p.get("is_instrument") else "effect (skipped)"
                log(f"{p['name']} ({p['format']}, {kind}): {len(p.get('programs') or [])} programs")
                ok += 1
        except (RuntimeError, ValueError) as e:  # crash, hang or garbage: remember and move on
            cat.setdefault("failed", {})[path] = {"error": str(e), "at": _now()}
            log(f"{Path(path).name}: {e}")
        log(f"progress {i}/{len(files)}")
        store.save_catalog(cat)
    return {"scanned": ok, "failed": len(files) - ok}


@jobs.action("asm_probe")
def _probe(args: dict, log) -> dict:
    """Find each selected preset's playable range: coarse probes, then key-by-key edges."""
    store = AssimilationStore(Path(args["home"]))
    plugin, refs = _plugin_and_presets(store, args)
    pdir = store.plugin_dir(plugin)
    write_json_atomic(pdir / "plugin.json", {**plugin, "presets": [r.to_dict() for r in refs]})
    results = store.results(plugin)
    probes = probe_notes()
    log(f"finding the playable range of {len(refs)} presets ({len(probes)} probe notes each)")
    work = pdir / "work" / "probe"
    shutil.rmtree(work, ignore_errors=True)
    coarse_n = len(refs) * len(probes)
    total = coarse_n + len(refs) * 2 * (PROBE_STEP - 1)   # edges are refined after this pass
    by_preset, failed = _probe_pass(args, plugin, refs, probes, work, log, 0, total)
    coarse = {r.key: decide_range(by_preset.get(r.key, [])) for r in refs if r.key not in failed}

    # Edges, one semitone at a time, for all ranged presets in one more host call.
    edge_keys = sorted({k for rng in coarse.values() for k in refine_keys(rng)})
    refined = {}
    ranged = [r for r in refs if coarse.get(r.key, {}).get("status") == "ranged"]
    if edge_keys and ranged:
        edges = refine_notes(edge_keys)
        refined, failed2 = _probe_pass(args, plugin, ranged, edges, pdir / "work" / "refine", log,
                                       coarse_n, max(total, coarse_n + len(edges) * len(ranged)))
        failed.update(failed2)
    for r in refs:
        if r.key in failed:
            results[r.key] = {**results.get(r.key, {}), "name": r.name, "kind": r.kind,
                              "status": "failed", "error": failed[r.key], "probed_at": _now()}
            log(f"{r.name}: the plugin failed on this preset ({failed[r.key]}), skipped")
            continue
        rng = extend_range(coarse[r.key], refined.get(r.key, []), by_preset.get(r.key, []))
        results[r.key] = {**results.get(r.key, {}), "name": r.name, "kind": r.kind, **rng,
                          "probed_at": _now()}
        results[r.key].pop("error", None)
        if rng["status"] == "ranged":
            log(f"{r.name}: {rng['range']}{' pitched' if rng['pitched'] else ' unpitched'}")
        else:
            log(f"{r.name}: {rng['status']}, skipped")
    write_json_atomic(pdir / "presets.json", results)
    shutil.rmtree(pdir / "work", ignore_errors=True)
    log(f"progress {total}/{total}")
    ranged_n = sum(1 for r in refs if results[r.key]["status"] == "ranged")
    if len(failed) == len(refs):
        raise RuntimeError("the plugin failed on every selected preset; see the log")
    return {"ranged": ranged_n, "failed": len(failed)}


@jobs.action("asm_harvest")
def _harvest(args: dict, log) -> dict:
    """Render every ranged preset across its range at the chosen density and analyse it."""
    store = AssimilationStore(Path(args["home"]))
    plugin, refs = _plugin_and_presets(store, args)
    density = Density(**{**args["density"], "velocities": tuple(args["density"]["velocities"]),
                         "durations": tuple(args["density"]["durations"])})
    pdir = store.plugin_dir(plugin)
    results = store.results(plugin)
    todo = [r for r in refs if results.get(r.key, {}).get("status") in ("ranged", "harvested")]
    plans = {r.key: grid_notes(results[r.key]["low"], results[r.key]["high"], density) for r in todo}
    total = sum(len(n) for n in plans.values())
    log(f"rendering {len(todo)} presets at {density.label}: {total} notes")
    feats = pdir / "features.jsonl"
    # A new harvest replaces earlier features for the same presets.
    if feats.exists():
        keep = [line for line in feats.read_text(encoding="utf-8").splitlines()
                if line and json.loads(line)["preset"] not in plans]
        feats.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    done, failed = 0, 0
    for ref in todo:
        notes = plans[ref.key]
        work = pdir / "work" / ref.key
        shutil.rmtree(work, ignore_errors=True)
        try:
            _render(args["host"], plugin, [ref], notes, work, log, done, total)
            recs = _analyse(args["python"], args.get("descriptor", ""), work, notes, feats, log,
                            not args.get("keep_audio"))
            results[ref.key].update(status="harvested", density=density.name, renders=len(recs),
                                    harvested_at=_now())
        except jobs.JobCancelled:
            raise
        except RuntimeError as e:  # this preset crashed or hung the plugin: record it, go on
            failed += 1
            results[ref.key].update(status="failed", error=str(e))
            log(f"{ref.name}: {e}")
        done += len(notes)
        log(f"progress {done}/{total}")
        write_json_atomic(pdir / "presets.json", results)
        if not args.get("keep_audio"):
            shutil.rmtree(work, ignore_errors=True)
    if not args.get("keep_audio"):
        shutil.rmtree(pdir / "work", ignore_errors=True)
    if todo and failed == len(todo):
        raise RuntimeError("every preset failed to render; see the log")
    return {"harvested": len(todo) - failed, "renders": done}


def create_scan_job(manager: jobs.JobManager, home: Path, host: str, files: list) -> dict:
    if not host:
        raise ValueError("the render host (alfa-render-host) wasn't found; set it in Settings")
    if not files:
        raise ValueError("choose at least one plugin to scan")
    return manager.create("plugin-scan", f"Scan {len(files)} plugin{'s' if len(files) > 1 else ''}",
                          [jobs.action_step("asm_scan", label="Open plugins", home=str(home), host=host,
                                            files=list(files))])


def create_assimilation_job(manager: jobs.JobManager, home: Path, uid: str, density: Density,
                            host: str, python: str, descriptor: str = "", presets=None,
                            keep_audio: bool = False) -> dict:
    if not host:
        raise ValueError("the render host (alfa-render-host) wasn't found; set it in Settings")
    store = AssimilationStore(home)
    plugin = store.catalog()["plugins"].get(uid)
    if plugin is None:
        raise ValueError("scan the plugin first")
    common = dict(home=str(home), uid=uid, host=host, python=python, descriptor=descriptor,
                  presets=list(presets) if presets else None)
    steps = [jobs.action_step("asm_probe", label="Find playable ranges", **common),
             jobs.action_step("asm_harvest", label=f"Render and analyse ({density.label})",
                              density=density.to_dict(), keep_audio=keep_audio, **common)]
    return manager.create("assimilate", f"Assimilate {plugin['name']} · {density.label}", steps,
                          params={"uid": uid, "density": density.name})


__all__ = ["AssimilationStore", "DENSITIES", "create_assimilation_job", "create_scan_job",
           "decide_range", "find_render_host", "host_formats"]
