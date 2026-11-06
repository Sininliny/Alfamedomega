#!/usr/bin/env python3
"""Compare original constructions under identical per-family probe conditions."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from .prepare import NativeRuntime, LIB, inputs_for, audible, features, NAMES, dump
import numpy as np


def family_audit(job):
    root, ti = job
    root = Path(root)
    paths = sorted((root / "shards").glob(f"{ti:02d}-*/labels/*-canonical.json"))
    performance = {
        "version": "amo.performance-plan/2",
        "sample_rate": 24000,
        "frames": 36000,
        "midi_note": 60,
        "tuning_cents": 0.0,
        "velocity": 0.7,
        "note_off_seconds": 0.75,
        "events": "held",
        "vibrato_rate": 5.0,
        "vibrato_cents": 0.0,
        "input_seed": 20261014,
        "input_kind": "noise_bursts" if ti >= 24 else "none",
        "role": "paired_fx" if ti >= 24 else "instrument",
    }
    source, midi = inputs_for(performance)
    responses = []
    identities = []
    minimum = 1.0
    for path in paths:
        doc = json.loads(path.read_text())
        with NativeRuntime(doc, 24000, LIB) as native:
            out = audible(doc, native.render(audio=source, midi=midi))
        minimum = min(minimum, float(np.sqrt(np.mean(out * out))))
        # Omit duration and absolute RMS so a mere gain change cannot inflate this count.
        response = features(out, 24000)[1:-1]
        responses.append(response)
        identities.append(
            hashlib.sha256(
                np.rint(response * 1000).astype(np.int32).tobytes()
            ).hexdigest()
        )
    matrix = np.asarray(responses, np.float32)
    # Bound the calculation to one 512-original family at a time.
    squared = np.sum(matrix * matrix, axis=1)
    distances = (
        np.maximum(squared[:, None] + squared[None, :] - 2 * (matrix @ matrix.T), 0)
        / matrix.shape[1]
    )
    np.fill_diagonal(distances, np.inf)
    nearest = np.sqrt(distances.min(axis=1))
    return {
        "type_id": ti,
        "structure": NAMES[ti],
        "originals": len(paths),
        "distinct_relative_response_fingerprints_1e3": len(set(identities)),
        "nearest_response_rms_distance_median": float(np.median(nearest)),
        "nearest_response_rms_distance_p05": float(np.quantile(nearest, 0.05)),
        "minimum_standardized_probe_rms": minimum,
        "probe_conditions": performance,
    }


def main(root, workers):
    outcomes = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed(
            [pool.submit(family_audit, (str(root), ti)) for ti in range(32)]
        ):
            result = future.result()
            outcomes.append(result)
            print(
                "audited",
                result["structure"],
                result["distinct_relative_response_fingerprints_1e3"],
                "/",
                result["originals"],
                flush=True,
            )
    result = {
        "protocol": "same performance and same excitation per graph family; no absolute gain or duration in fingerprint",
        "originals": sum(r["originals"] for r in outcomes),
        "distinct_original_responses_1e3_within_families": sum(
            r["distinct_relative_response_fingerprints_1e3"] for r in outcomes
        ),
        "families": sorted(outcomes, key=lambda r: r["type_id"]),
        "interpretation": "A deterministic acoustic diversity diagnostic, not a statistical effective sample size or a perceptual independence proof.",
    }
    dump(root / "diversity-audit.json", result)
    # Supply a separate composition-generalization protocol; do not mix it with standard splits.
    held = [6, 21, 28, 30]
    dump(
        root / "unseen-topology-protocol.json",
        {
            "held_out_type_ids": held,
            "held_out_structures": [NAMES[i] for i in held],
            "train": "standard train originals of other 28 families",
            "validation": "standard validation originals of other 28 families",
            "test": "standard test originals of other 28 families",
            "topology_test": "all originals and variants of the four withheld families",
            "warning": "Choose one evaluation protocol before fitting a model; ordinary training has already seen these families.",
        },
    )
    print(json.dumps({k: v for k, v in result.items() if k != "families"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args.root.resolve(), args.workers)
