#!/usr/bin/env python3
"""Inference-time candidate rescoring for the mix decomposer (MIX-DESIGN.md §2, §5, §11).

The network proposes; the objective decides. For one mix ``x`` we build a small set of
candidate scenes ``S`` (decoded from several count tokens, plus "echo merges" that fold
a delayed, transposed copy of an instrument back into it as note echo), render every
candidate with the native DSP, and pick

    auto:      argmin_S  D(x, x(S)) + λ_I K + λ_N N + λ_F F + λ_B B
    fixed K:   argmin_{S: |S| = K}  D(x, x(S)) + λ_N N + λ_F F + λ_B B

with K instruments, N stored notes (echo taps are not stored notes, which is exactly why
folding an echo copy into its source is cheaper), F insert FX + note echoes, B bus FX.

``tune`` caches every candidate's (D, K, N, F, B) for the validation split, grid-searches
the λ on it (target: ≥ 90 % of FX-echo scenes explained as one instrument without hurting
the overall count accuracy) and reports the chosen λ on the test split.

Example::

    python -m training.amo.corpus.mixinfer tune EVAL_CORPUS runs/mix/model.pt --out rescore.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus import mixscene  # noqa: E402

ONSET_TOLERANCE = 0.03  # s; about one roll frame and a half
MIN_MATCH = 0.6  # share of the follower's notes that must be a shifted copy of the leader's
MAX_TAPS = 3
MAX_SHIFT = MAX_TAPS * 1.0 + ONSET_TOLERANCE  # the generator's longest echo delay is 1 s
DECAY_RANGE = (0.35, 0.75)
LAMBDA_GRID = {
    "instrument": [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
    "note": [0.0, 0.0005, 0.001, 0.002, 0.004],
    "fx": [0.0, 0.002, 0.005, 0.01],
}
ECHO_TARGET = 0.9


# ------------------------------------------------------------------ echo merges


def echo_shift(leader, follower):
    """Best (delay s, transpose semitones, matched share) with follower ≈ leader shifted."""
    a = np.asarray([[n[0], n[2]] for n in leader["notes"]], float).reshape(-1, 2)
    b = np.asarray([[n[0], n[2]] for n in follower["notes"]], float).reshape(-1, 2)
    if not len(a) or not len(b):
        return None
    dt = b[:, None, 0] - a[None, :, 0]
    dp = b[:, None, 1] - a[None, :, 1]
    ok = (dt > 0.04) & (dt <= MAX_SHIFT) & (np.abs(dp) <= MAX_TAPS * 24)
    if not ok.any():
        return None
    votes = {}
    for t, p in zip(np.round(dt[ok] / 0.01).astype(int), dp[ok].astype(int)):
        votes[(t, p)] = votes.get((t, p), 0) + 1
    best = None
    for (t, p), _ in sorted(votes.items(), key=lambda kv: -kv[1])[:8]:
        delay = t * 0.01
        hit = (np.abs(dt - delay) <= ONSET_TOLERANCE) & (dp == p)
        matched = hit.any(axis=1)
        if matched.any():  # refine the delay on the matched pairs
            delay = float(np.median(dt[hit]))
        share = float(matched.mean())
        if best is None or share > best[2]:
            best = (delay, int(p), share, int(matched.sum()))
    if best is None or best[2] < MIN_MATCH or best[3] < min(2, len(b)):
        return None
    return best[:3]


def echo_groups(instruments):
    """Leader -> followers that are its note-echo taps 1..MAX_TAPS (one consistent delay/transpose)."""
    groups = []
    used = set()
    order = sorted(range(len(instruments)), key=lambda i: instruments[i]["notes"][0][0]
                   if instruments[i]["notes"] else 1e9)
    for lead in order:
        if lead in used or instruments[lead].get("note_echo") or len(instruments[lead]["layers"]) != 1:
            continue
        shifts = {}
        for other in order:
            if other == lead or other in used or instruments[other].get("note_echo"):
                continue
            if instruments[other]["layers"][0]["family"] != instruments[lead]["layers"][0]["family"]:
                continue  # an echo tap is the same preset; same rhythm in another timbre is a real part
            found = echo_shift(instruments[lead], instruments[other])
            if found:
                shifts[other] = found
        if not shifts:
            continue
        first = min(shifts, key=lambda i: shifts[i][0])
        delay, transpose, _ = shifts[first]
        taps = {}
        for other, (dt, dp, share) in shifts.items():
            j = int(round(dt / delay))
            if 1 <= j <= MAX_TAPS and abs(dt - j * delay) <= ONSET_TOLERANCE and dp == j * transpose:
                if j not in taps or share > shifts[taps[j]][2]:
                    taps[j] = other
        if taps:
            groups.append({"leader": lead, "taps": taps, "delay": delay, "transpose": transpose})
            used.add(lead)
            used.update(taps.values())
    return groups


def apply_echo_group(instruments, group):
    """Fold the group's followers into the leader as note echo; returns a new instrument list."""
    lead = instruments[group["leader"]]
    ratios = [(10 ** ((instruments[f]["gain_db"] - lead["gain_db"]) / 20)) ** (1 / j)
              for j, f in group["taps"].items()]
    decay = float(np.clip(np.exp(np.mean(np.log(np.maximum(ratios, 1e-6)))), *DECAY_RANGE))
    merged = copy.deepcopy(lead)
    merged["note_echo"] = {"taps": max(group["taps"]), "delay": float(group["delay"]),
                           "transpose": int(group["transpose"]), "decay": decay}
    gone = set(group["taps"].values())
    return [merged if i == group["leader"] else inst for i, inst in enumerate(instruments) if i not in gone]


def merge_sequence(instruments):
    """The instrument list after 0, 1, 2, ... echo merges (greedy, largest group first)."""
    states = [instruments]
    current = instruments
    while True:
        groups = echo_groups(current)
        if not groups:
            return states
        group = max(groups, key=lambda g: len(g["taps"]))
        current = apply_echo_group(current, group)
        states.append(current)


# ---------------------------------------------------------------- candidates


def cost_terms(scene):
    instruments = scene["instruments"]
    return {"K": len(instruments), "N": int(sum(len(i["notes"]) for i in instruments)),
            "F": int(sum(len(i["insert_fx"]) + (1 if i.get("note_echo") else 0) for i in instruments)),
            "B": len(scene["bus_fx"])}


def candidate_scenes(model, features, frames, seconds, fixed=None, top_counts=3):
    """Decoded candidate scenes with their provenance.

    Auto (``fixed`` None): the top ``top_counts`` counts of the count head (count token 0 decides
    slot order), each with its echo-merge sequence. Fixed K: count tokens K, K+1, K+2, merged or
    with the weakest slots dropped until exactly K instruments remain.
    """
    from training.amo.corpus import mixtargets as T, mixtrain

    found = []
    if fixed is None:
        out = mixtrain.forward(model, features, frames, 0)
        probabilities = mixtrain.count_probabilities(out)
        for k in [int(i) + 1 for i in np.argsort(-probabilities)[:top_counts]]:
            units, bus_present, bus_values = mixtrain.slot_units(out, frames, k)
            base = T.decode_scene(units, bus_present, bus_values, seconds)
            for step, instruments in enumerate(merge_sequence(base["instruments"])):
                found.append(({**base, "instruments": instruments},
                              {"count_token": 0, "slots": k, "merges": step, "dropped": 0}))
        return found
    for token in range(fixed, min(fixed + 2, mixscene.MAX_INSTRUMENTS) + 1):
        out = mixtrain.forward(model, features, frames, token)
        units, bus_present, bus_values = mixtrain.slot_units(out, frames, token)
        base = T.decode_scene(units, bus_present, bus_values, seconds)
        for step, instruments in enumerate(merge_sequence(base["instruments"])):
            if len(instruments) == fixed:
                found.append(({**base, "instruments": instruments},
                              {"count_token": token, "slots": token, "merges": step, "dropped": 0}))
        if len(base["instruments"]) > fixed:  # slots are most-likely first: drop the tail
            found.append(({**base, "instruments": base["instruments"][:fixed]},
                          {"count_token": token, "slots": token, "merges": 0,
                           "dropped": len(base["instruments"]) - fixed}))
        if token == fixed and len(base["instruments"]) < fixed:  # empty slots decoded away: keep it
            found.append((base, {"count_token": token, "slots": token, "merges": 0, "dropped": 0}))
    return found


def render_distance(job):
    """D for one candidate (inf when it does not render)."""
    from training.amo.corpus import resynth

    scene, target = job
    if not scene["instruments"]:
        return float("inf")
    try:
        audio = mixscene.render_scene(scene)[1]
    except Exception:  # noqa: BLE001 - an unrenderable candidate scores inf
        return float("inf")
    return resynth.distance(resynth.log_spectra(audio), target)


def objective(c, lam, fixed=False):
    value = c["D"] + lam["note"] * c["N"] + lam["fx"] * (c["F"] + c["B"])
    return value if fixed else value + lam["instrument"] * c["K"]


def choose(candidates, lam, fixed=False):
    finite = [c for c in candidates if np.isfinite(c["D"])]
    return min(finite, key=lambda c: objective(c, lam, fixed)) if finite else None


# ---------------------------------------------------------------- tuning


def score_split(model, corpus, indices, workers, top_counts):
    """Every candidate's cost terms for each scene, auto and fixed L0/L1/L2."""
    from training.amo.corpus import mixtargets as T, resynth

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index in indices:
            scene = corpus["scenes"][index]
            _, mix, _ = mixscene.render_scene(scene)
            target = resynth.log_spectra(mix)
            features = corpus["mix"][index].astype(np.float32)
            frames = int(corpus["frames"][index])
            counts = {lv: int(max(scene["ladder"][lv]) + 1) for lv in T.LEVELS}
            row = {"index": index, "counts": counts,
                   "note_echo": any(i["note_echo"] for i in scene["instruments"]),
                   "doubles": any(i["doubles"] is not None for i in scene["instruments"])}
            modes = {"auto": None, **{f"fixed_{lv}": counts[lv] for lv in T.LEVELS}}
            for name, fixed in modes.items():
                found = candidate_scenes(model, features, frames, scene["seconds"], fixed, top_counts)
                distances = list(pool.map(render_distance, [(s, target) for s, _ in found]))
                row[name] = [{**info, **cost_terms(s), "D": d} for (s, info), d in zip(found, distances)]
            rows.append(row)
            print(f"scene {index}: {len(row['auto'])} auto candidates", flush=True)
    return rows


def baseline(candidates):
    """The network alone: argmax count, no merges (what ``mixtrain evaluate`` scores)."""
    first = [c for c in candidates if c["merges"] == 0 and c["dropped"] == 0]
    return first[0] if first else None


def metrics(rows, pick):
    """Summary for a selection rule ``pick(row, mode) -> candidate``."""
    out = {}
    for mode in ("auto", "fixed_L0", "fixed_L1", "fixed_L2"):
        chosen = [pick(r, mode) for r in rows]
        d = np.asarray([c["D"] if c else np.inf for c in chosen])
        out[mode] = {"median_distance": float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else None,
                     "failed": int((~np.isfinite(d)).sum())}
    autos = [(r, pick(r, "auto")) for r in rows]
    hit = lambda r, c: c is not None and c["K"] == r["counts"]["L0"]
    out["auto_count_accuracy_L0"] = float(np.mean([hit(r, c) for r, c in autos]))
    echo = [(r, c) for r, c in autos if r["note_echo"]]
    out["echo_scenes"] = len(echo)
    out["echo_auto_count_is_L0"] = float(np.mean([hit(r, c) for r, c in echo])) if echo else None
    doubles = [(r, c) for r, c in autos if r["doubles"]]
    out["doubles_scenes"] = len(doubles)
    out["doubles_auto_count_is_L0"] = float(np.mean([hit(r, c) for r, c in doubles])) if doubles else None
    out["over_merged"] = float(np.mean([c is not None and c["K"] < r["counts"]["L0"] for r, c in autos]))
    return out


def tune_lambdas(rows):
    """Grid search: reach the echo target if possible, then maximise count accuracy, then lower D."""
    results = []
    for li, ln, lf in itertools.product(*LAMBDA_GRID.values()):
        lam = {"instrument": li, "note": ln, "fx": lf}
        m = metrics(rows, lambda r, mode: choose(r[mode], lam, mode != "auto"))
        echo = m["echo_auto_count_is_L0"] or 0.0
        key = (min(echo, ECHO_TARGET), m["auto_count_accuracy_L0"], -(m["auto"]["median_distance"] or 9e9))
        results.append((key, lam, m))
    results.sort(key=lambda t: t[0], reverse=True)
    return results[0][1], results[0][2], [{"lambda": lam, **{k: m[k] for k in (
        "auto_count_accuracy_L0", "echo_auto_count_is_L0", "over_merged")}, "auto_D": m["auto"]["median_distance"]}
        for _, lam, m in results[:20]]


def tune(args):
    from training.amo.corpus import mixtrain

    model = mixtrain.load_model(args.model)
    corpus = mixtrain.load_corpus(args.corpus)
    report = {"model": str(args.model), "grid": LAMBDA_GRID, "echo_target": ECHO_TARGET}
    splits = {}
    for split, limit in (("validation", args.tune_scenes), ("test", args.test_scenes)):
        indices = mixtrain.split_indices(corpus, split)[:limit]
        splits[split] = score_split(model, corpus, indices, args.workers, args.top_counts)
    lam, val_metrics, top = tune_lambdas(splits["validation"])
    report["lambda"] = lam
    report["validation"] = {"network_only": metrics(splits["validation"], lambda r, m: baseline(r[m])),
                            "rescored": val_metrics, "top_grid": top}
    report["test"] = {"network_only": metrics(splits["test"], lambda r, m: baseline(r[m])),
                      "rescored": metrics(splits["test"], lambda r, m: choose(r[m], lam, m != "auto"))}
    Path(args.out).write_text(json.dumps(report, indent=1))
    Path(args.out).with_suffix(".rows.json").write_text(json.dumps(splits))
    print(json.dumps({k: report[k] for k in ("lambda",)} | {"test": report["test"]}, indent=2))
    return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tune")
    t.add_argument("corpus")
    t.add_argument("model")
    t.add_argument("--tune-scenes", type=int, default=200)
    t.add_argument("--test-scenes", type=int, default=200)
    t.add_argument("--top-counts", type=int, default=3)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--out", default="mix-rescore.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    return tune(args)


if __name__ == "__main__":
    raise SystemExit(main())
