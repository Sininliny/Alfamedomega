#!/usr/bin/env python3
"""Perceptual weights for the value loss: how audible each (family, coordinate) is.

The value heads are trained on per-(family, coordinate) z-scores (``train.TargetStats``),
so an unweighted loss spends the same effort on every coordinate. The resynthesis
evaluation showed that this is wrong in both directions: families that depend on
pitch, partial and decay coordinates are the worst re-rendered, while some coordinates
(fx_phaser) barely change the sound at all.

This module measures, with the native runtime, the spectral effect of moving one
coordinate by ``delta`` standard deviations around real training instances:

    s[t, c] = mean over examples and both directions of  D(render(v ± delta e_c), render(v)) / delta

where ``D`` is the resynthesis distance (``resynth.distance``: unit-RMS stereo
multi-resolution log-STFT L1). Within each family the sensitivities are divided by
their mean over learnable coordinates, clipped to ``[--floor, --cap]`` and normalised
again to mean 1, so the weights redistribute the loss inside a family without changing
how much each family counts overall. ``train.py --perceptual-weights`` multiplies the
value-loss mask by them.

    python -m training.amo.corpus.sensitivity CORPUS_DIR sensitivity.npz --per-type 32
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "amo.corpus-perceptual-weights/1"
VALUE_TENSORS = ("parameters", "routing", "modulation", "assets")


def value_keys():
    keys = ["type_id", "id"]
    for name in VALUE_TENSORS:
        for head in ("canonical", "instance"):
            keys += [f"{head}_{name}", f"{head}_{name}_mask"]
    return keys


def probe(job):
    """Distances for every probed coordinate of one example.

    ``job`` = (type_id, performance, base values, [(tensor, coordinate, minus, plus), ...]);
    returns [(tensor, coordinate, distance minus, distance plus), ...] with NaN where a
    perturbation is clipped to no change or the render fails.
    """
    from training.amo.corpus import resynth

    type_id, performance, base, probes = job
    reference = resynth.render(resynth.assemble(type_id, base), performance)
    reference = None if reference is None else resynth.log_spectra(reference)
    out = []
    for name, c, minus, plus in probes:
        distances = []
        for value in (minus, plus):
            if reference is None or not np.isfinite(value) or np.isclose(value, base[name][c]):
                distances.append(np.nan)
                continue
            values = {k: v.copy() for k, v in base.items()}
            values[name][c] = value
            audio = resynth.render(resynth.assemble(type_id, values), performance)
            d = resynth.distance(None if audio is None else resynth.log_spectra(audio), reference)
            distances.append(d if np.isfinite(d) else np.nan)
        out.append((name, c, *distances))
    return out


def main(argv=None):
    from training.amo.corpus import resynth
    from training.amo.corpus.reader import load_split
    from training.amo.corpus.train import fit_statistics

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path, help="sensitivity .npz (a .json summary is written next to it)")
    parser.add_argument("--per-type", type=int, default=32, help="training instances probed per family")
    parser.add_argument("--delta", type=float, default=0.25, help="perturbation in family standard deviations")
    parser.add_argument("--floor", type=float, default=0.05, help="smallest relative weight")
    parser.add_argument("--cap", type=float, default=8.0, help="largest relative weight")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args(argv)
    began = time.time()

    raw = load_split(args.corpus, "train", "originals", value_keys())
    types = int(raw["type_id"].max()) + 1
    stats = fit_statistics({**raw, "X_audio": np.zeros((1, 1)), "conditions": np.zeros((1, 1))}, types)
    widths = {name: raw[f"canonical_{name}"].shape[1] for name in VALUE_TENSORS}
    rows = resynth.split_rows(args.corpus, "train")
    rng = np.random.default_rng(args.seed)

    jobs = []
    for t in range(types):
        candidates = np.flatnonzero(raw["type_id"] == t)
        if candidates.size == 0:
            continue
        _, bounds = resynth.template(t)
        for index in rng.permutation(candidates)[:args.per_type]:
            base = {name: raw[f"instance_{name}"][index].astype(np.float64) for name in VALUE_TENSORS}
            probes = []
            for name in VALUE_TENSORS:
                st = stats[name]
                mask = raw[f"instance_{name}_mask"][index] & st.learnable[t]
                if not mask.any():
                    continue
                z = st.encode(np.array([t]), base[name][None], mask[None])[0].astype(np.float64)
                for c in np.flatnonzero(mask):
                    if name == "assets":
                        low, high = -1.0, 1.0
                    else:
                        low, high = bounds[name][c] if c < len(bounds[name]) else (-np.inf, np.inf)
                    side = []
                    for sign in (-1.0, 1.0):
                        shifted = z.copy()
                        shifted[c] += sign * args.delta
                        value = st.decode(np.array([t]), shifted[None])[0][c]
                        side.append(float(np.clip(value, low, high)))
                    probes.append((name, int(c), side[0], side[1]))
            if probes:
                jobs.append((t, rows[str(raw["id"][index])]["performance"], base, probes))
    renders = sum(1 + 2 * len(job[3]) for job in jobs)
    print(f"probing {len(jobs)} instances, {renders} renders, {args.workers} workers", flush=True)

    total = {name: np.zeros((types, widths[name])) for name in VALUE_TENSORS}
    count = {name: np.zeros((types, widths[name])) for name in VALUE_TENSORS}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job, result in zip(jobs, pool.map(probe, jobs, chunksize=2)):
            for name, c, minus, plus in result:
                for d in (minus, plus):
                    if np.isfinite(d):
                        total[name][job[0], c] += d / args.delta
                        count[name][job[0], c] += 1
            done += 1
            if done % max(1, len(jobs) // 20) == 0:
                elapsed = time.time() - began
                print(f"  {done}/{len(jobs)} instances, {elapsed / 60:.1f} min", flush=True)

    sensitivity = {name: np.where(count[name] > 0, total[name] / np.maximum(count[name], 1), np.nan)
                   for name in VALUE_TENSORS}
    weights = {name: np.ones((types, widths[name]), np.float32) for name in VALUE_TENSORS}
    names = json.loads((args.corpus / "build-plan.json").read_text()).get("types", [])
    summary = []
    for t in range(types):
        learn = [(name, c) for name in VALUE_TENSORS for c in np.flatnonzero(stats[name].learnable[t])]
        measured = [(name, c, sensitivity[name][t, c]) for name, c in learn if np.isfinite(sensitivity[name][t, c])]
        if not measured:
            continue
        mean = np.mean([s for _, _, s in measured])
        if mean <= 0:
            continue
        relative = {(name, c): float(np.clip(s / mean, args.floor, args.cap)) for name, c, s in measured}
        norm = np.mean(list(relative.values()))
        for (name, c), r in relative.items():
            weights[name][t, c] = r / norm
        ranked = sorted(relative.items(), key=lambda item: -item[1])
        summary.append({
            "family": names[t] if t < len(names) else f"type{t}",
            "learnable": len(learn), "measured": len(measured),
            "mean_sensitivity": float(mean),
            "weights": {f"{name}[{c}]": round(r / norm, 3) for (name, c), r in ranked}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, schema=SCHEMA, delta=args.delta, floor=args.floor, cap=args.cap,
             **{f"weight_{n}": w for n, w in weights.items()},
             **{f"sensitivity_{n}": s for n, s in sensitivity.items()},
             **{f"count_{n}": c for n, c in count.items()})
    report = {"schema": SCHEMA, "corpus": str(args.corpus), "per_type": args.per_type, "delta": args.delta,
              "floor": args.floor, "cap": args.cap, "instances": len(jobs), "renders": renders,
              "minutes": (time.time() - began) / 60, "families": summary}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    for f in summary:
        top = ", ".join(f"{k} {v}" for k, v in list(f["weights"].items())[:3])
        low = ", ".join(f"{k} {v}" for k, v in list(f["weights"].items())[-2:])
        print(f"  {f['family']:<24} mean s {f['mean_sensitivity']:.3f} | top {top} | lowest {low}")
    print(f"wrote {args.output} in {(time.time() - began) / 60:.1f} min")


if __name__ == "__main__":
    main()
