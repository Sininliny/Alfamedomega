#!/usr/bin/env python3
"""Evaluate an exported corpus encoder (``model.pt``) on a held-out split.

Reports the same metrics as training validation plus a family confusion matrix,
per-family accuracy and end-to-end parameter error, and accuracy per
performance case and sample rate, so it shows which families the encoder
cannot tell apart. The test split is never seen by ``train.py``.

    python -m training.amo.corpus.evaluate CORPUS_DIR runs/corpus-encoder/model.pt --split test --out report.json
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus.reader import load_split  # noqa: E402
from training.amo.corpus.train import (  # noqa: E402
    HEADS, MODEL_SCHEMA, VALUE_TENSORS, Standardizer, TargetStats, build_model,
    corpus_keys, encode_split, evaluate, parse, split_values, take, to_device)


def family_names(root, types):
    plan = root / "build-plan.json"
    names = json.loads(plan.read_text()).get("types", []) if plan.exists() else []
    return [names[t] if t < len(names) else f"type{t}" for t in range(types)]


def resynthesis(model, data, ids, truth, predicted, stats, widths, names, args, amp_dtype):
    """Render predictions for a per-family sample and score them against the rendered target.

    Candidates per example, each rendered with the example's own performance plan:
    ``predicted`` (predicted family and its instance values), ``true_family`` (values
    predicted with the true family given), ``family_centre`` (true family, every
    coordinate at its training mean: what the value head must beat).
    """
    import torch
    from training.amo.corpus import resynth

    rng = np.random.default_rng(20261027)
    chosen = np.concatenate([rng.permutation(np.flatnonzero(truth == t))[:args.resynth]
                             for t in range(len(names)) if (truth == t).any()])
    batch = take(data, torch.from_numpy(chosen).to(data["x"].device))

    def instance_values(type_ids, z):
        parts = {}
        offset = 0
        for name in VALUE_TENSORS:
            width = widths[name]
            parts[name] = stats[name].decode(type_ids, z[:, offset:offset + width].astype(np.float64))
            offset += width
        return parts

    head = list(HEADS).index("instance")
    with torch.no_grad(), torch.autocast(batch["x"].device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        free = model(batch["x"], batch["conditions"], None)["values"][:, head].float().cpu().numpy()
        given = model(batch["x"], batch["conditions"], batch["type_id"])["values"][:, head].float().cpu().numpy()
    true_types, predicted_types = truth[chosen], predicted[chosen]
    candidates = {"predicted": (predicted_types, instance_values(predicted_types, free)),
                  "true_family": (true_types, instance_values(true_types, given)),
                  "family_centre": (true_types, instance_values(true_types, np.zeros_like(given)))}
    rows = resynth.split_rows(args.corpus, args.split)
    jobs = []
    for i, index in enumerate(chosen):
        row = rows[str(ids[index])]
        target = json.loads((Path(args.corpus) / row["instance_json"]).read_text())
        jobs.append((target, row["performance"],
                     {name: (int(t[i]), {k: v[i] for k, v in values.items()})
                      for name, (t, values) in candidates.items()}))
    scores = resynth.score_all(jobs, args.workers)
    by = {name: np.array([s[name] for s in scores]) for name in candidates}
    right = predicted_types == true_types
    correct_median = resynth.summarise(by["predicted"][right])["median"] or 0.0
    families = []
    for t in np.unique(true_types):
        rows_t = true_types == t
        wrong = rows_t & ~right
        families.append({
            "family": names[t], "examples": int(rows_t.sum()),
            **{f"{name}_median": resynth.summarise(by[name][rows_t])["median"] for name in candidates},
            "wrong_family": int(wrong.sum()),
            "wrong_family_median": resynth.summarise(by["predicted"][wrong])["median"] if wrong.any() else None,
            "wrong_but_close": int((wrong & (by["predicted"] <= correct_median)).sum())})
    wrong = ~right
    result = {
        "examples": len(chosen), "distance": "mean |log STFT| L1, unit-RMS stereo, FFT 2048/1024/512/256",
        "candidates": {name: resynth.summarise(values) for name, values in by.items()},
        "predicted_when_family_right": resynth.summarise(by["predicted"][right]),
        "predicted_when_family_wrong": resynth.summarise(by["predicted"][wrong]),
        "wrong_family_but_at_most_correct_median": int((wrong & (by["predicted"] <= correct_median)).sum()),
        "wrong_family_total": int(wrong.sum()),
        "families": sorted(families, key=lambda f: -(f["predicted_median"] or 0)),
    }
    def fmt(value):
        return "  n/a" if value is None else f"{value:.3f}"

    print(f"  resynthesis ({len(chosen)} examples; {result['distance']}):")
    for name, summary in result["candidates"].items():
        print(f"    {name:<14} median {fmt(summary['median'])} mean {fmt(summary['mean'])} failed {summary['failed']}")
    print(f"    family right: median {fmt(result['predicted_when_family_right']['median'])};"
          f" family wrong: median {fmt(result['predicted_when_family_wrong']['median'])};"
          f" wrong family but within the correct median: {result['wrong_family_but_at_most_correct_median']}"
          f"/{result['wrong_family_total']}")
    for f in result["families"]:
        print(f"    {f['family']:<24} predicted {fmt(f['predicted_median'])} true-family {fmt(f['true_family_median'])}"
              f" centre {fmt(f['family_centre_median'])} | wrong family {f['wrong_family']}"
              f" (close {f['wrong_but_close']})")
    return result


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("model", type=Path, help="model.pt written by train.py")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--protocol", default="originals", choices=("originals", "unseen_topology"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--resynth", type=int, default=0,
                        help="render the predicted construction for up to N examples per family (0 = off)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="render processes")
    args = parser.parse_args(argv)

    package = torch.load(args.model, map_location="cpu", weights_only=False)
    if package.get("schema") != MODEL_SCHEMA:
        raise SystemExit(f"{args.model} is not an {MODEL_SCHEMA} package")
    stats = {"X_audio": Standardizer(**package["stats"]["X_audio"]),
             "conditions": Standardizer(**package["stats"]["conditions"])}
    for name in VALUE_TENSORS:
        stats[name] = TargetStats(**package["stats"][name])
    widths, types = package["widths"], package["types"]
    operations = max(package["operation_vocabulary"].values())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = build_model(package["config"], widths, types, operations)
    model.load_state_dict(package["model"])
    model.to(device).eval()

    raw = load_split(args.corpus, args.split, args.protocol, corpus_keys() + ["id"])
    conditions = raw["conditions"]
    ids = raw["id"]
    data = to_device(encode_split(raw, stats), device)
    del raw
    defaults = parse([str(args.corpus), "unused"])
    weights = {"type": defaults.type_weight, "ops": defaults.ops_weight, "presence": defaults.presence_weight,
               "canonical": defaults.canonical_weight, "instance": defaults.instance_weight}
    metrics = {k.replace("val_", f"{args.split}_", 1): v
               for k, v in evaluate(model, data, widths, weights, args.batch_size, amp_dtype).items()}

    count = int(data["type_id"].shape[0])
    truth = data["type_id"].cpu().numpy()
    predicted = np.empty(count, np.int64)
    graph_ok = np.empty(count, bool)
    e2e_sum = np.zeros(count)
    e2e_weight = np.zeros(count)
    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            batch = take(data, slice(start, start + args.batch_size))
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                free = model(batch["x"], batch["conditions"], None)
            rows = slice(start, start + batch["type_id"].shape[0])
            predicted[rows] = free["type_used"].cpu().numpy()
            graph_ok[rows] = (free["ops"].argmax(2) == batch["node_ops"]).all(1).cpu().numpy()
            values = split_values(free["values"][:, 0].float(), widths)
            right = (free["type_used"] == batch["type_id"]).float()[:, None]
            for name in VALUE_TENSORS:
                w = batch[f"canonical_{name}_w"]
                z = batch[f"canonical_{name}_z"]
                error = (values[name] - z).abs()
                error = right * error + (1 - right) * torch.maximum(error, z.abs())
                e2e_sum[rows] += (error * w).sum(1).cpu().numpy()
                e2e_weight[rows] += w.sum(1).cpu().numpy()

    names = family_names(Path(args.corpus), types)
    confusion = np.zeros((types, types), np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    families = []
    for t in range(types):
        rows = truth == t
        if not rows.any():
            continue
        wrong = confusion[t].copy()
        wrong[t] = 0
        top = [(names[j], int(wrong[j])) for j in np.argsort(-wrong)[:3] if wrong[j]]
        families.append({"family": names[t], "examples": int(rows.sum()),
                         "type_acc": float(confusion[t, t] / rows.sum()),
                         "graph_acc": float(graph_ok[rows].mean()),
                         "mae_z_canonical_e2e": float(e2e_sum[rows].sum() / max(e2e_weight[rows].sum(), 1.0)),
                         "confused_with": top})
    # conditions[:, 8] = case / 3 (held, staccato, two_notes, vibrato); conditions[:, 0] = sr / 24000.
    by_case = {}
    for case, label in enumerate(("held", "staccato", "two_notes", "vibrato")):
        rows = np.isclose(conditions[:, 8], case / 3.0)
        if rows.any():
            by_case[label] = float((predicted[rows] == truth[rows]).mean())
    by_rate = {}
    for rate in sorted(set(np.round(conditions[:, 0] * 24000).astype(int).tolist())):
        rows = np.round(conditions[:, 0] * 24000).astype(int) == rate
        by_rate[str(rate)] = float((predicted[rows] == truth[rows]).mean())
    pairs = [(names[i], names[j], int(confusion[i, j]) + int(confusion[j, i]))
             for i in range(types) for j in range(i + 1, types) if confusion[i, j] + confusion[j, i]]
    pairs.sort(key=lambda p: -p[2])
    report = {"model": str(args.model), "epoch": package.get("epoch"), "split": args.split,
              "examples": count, "metrics": metrics, "type_acc_by_case": by_case,
              "type_acc_by_sample_rate": by_rate, "most_confused_pairs": pairs[:15],
              "families": sorted(families, key=lambda f: f["type_acc"]),
              "family_names": names, "confusion": confusion.tolist()}
    if args.resynth:
        report["resynthesis"] = resynthesis(model, data, ids, truth, predicted, stats, widths, names,
                                            args, amp_dtype)

    print(f"{args.split}: {count} examples, model epoch {package.get('epoch')}")
    print("  " + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items()
                             if k.endswith(("_total", "_acc", "_e2e", "_canonical", "_instance"))))
    print("  type acc by case: " + ", ".join(f"{k} {v:.3f}" for k, v in by_case.items())
          + " | by rate: " + ", ".join(f"{k} {v:.3f}" for k, v in by_rate.items()))
    print("  per family (worst first):")
    for f in report["families"]:
        confused = ", ".join(f"{n} {c}" for n, c in f["confused_with"])
        print(f"    {f['family']:<24} type {f['type_acc']:.3f} graph {f['graph_acc']:.3f}"
              f" e2e {f['mae_z_canonical_e2e']:.3f}  confused with: {confused}")
    print("  most confused pairs: " + "; ".join(f"{a} <-> {b} {c}" for a, b, c in pairs[:10]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
