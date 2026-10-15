"""Reproducible controlled train → GRU export → native DSP integration proof.

Every dimension, target, clock and allowance below belongs to this test fixture;
none defines a product tier, musical default or universal learned audio model.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .native import NativeRuntime, render_json
from .pipeline import train_runtime_manifest
from .runtime_model import INPUT_CATEGORIES
from .smoke import write_wav


def controlled_program():
    return {
        "schema": "amo.tensor/1", "ontology": "amo.math/1",
        "description": "Controlled runtime-model proof fixture; not a synthesis default.",
        "tensors": {
            "activation": {"axes": ["primitive"], "shape": [1], "values": [1.]},
            "parameters": {"axes": ["parameter"], "shape": [2], "values": [0., 0.]},
            "routing": {"axes": ["coefficient"], "shape": [2], "values": [1., .2]},
            "modulation": {"axes": ["relationship"], "shape": [0], "values": []}},
        "instances": [{"primitive": "math.sine", "activation": 0,
                       "parameters": {"offset": 0, "count": 2},
                       "clock": {"period_samples": 1, "phase_samples": 0},
                       "output_between_updates": "zero", "state_size": 1,
                       "initial_state": [0.], "assets": []}],
        "projections": [{"source": "midi", "source_element": index,
                         "parameter_element": index, "weights": "routing",
                         "weight_element": index, "missing_source": "zero"}
                        for index in range(2)],
        "parameter_locks": [], "shared_initial_state": [],
        "outputMix": [[0, 0, 1.]],
        "budget": {"active_primitives": 1, "state_elements": 1,
                   "parameter_elements": 2, "routing_elements": 2,
                   "modulation_elements": 0, "output_elements": 1,
                   "stored_tensor_elements": 5, "projection_terms": 2}}


def run(directory: Path, dll_path=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    baseline = controlled_program()
    target = copy.deepcopy(baseline)
    target["tensors"]["routing"]["values"][1] = .4
    (directory / "runtime-target.json").write_text(json.dumps(target, indent=2), encoding="utf-8")
    widths = {name: 0 for name in INPUT_CATEGORIES}
    widths["midi"] = 2
    sequence = []
    for velocity in (.2, .8, .4):
        inputs = {name: [] for name in INPUT_CATEGORIES}
        inputs["midi"] = [200., velocity]
        sequence.append({"inputs": inputs, "target_json": "runtime-target.json"})
    manifest = {
        "ontology": "amo.math/1", "input_widths": widths,
        "schemas": {name: {"axes": tensor["axes"], "padded_shape": tensor["shape"]}
                    for name, tensor in baseline["tensors"].items()},
        "hidden_width": 8, "available_ram_bytes": 100000000, "seed": 7,
        "steps": 180, "learning_rate": .03, "mask_weight": .1,
        "clock": {"period_samples": 7, "phase_samples": 2}, "mask_threshold": .5,
        "bounds": {"activation": [[0., 1.]], "parameters": [[0., 0.], [0., 0.]],
                   "routing": [[0., 1.], [0., 1.]], "modulation": []},
        "sequences": [sequence]}
    manifest_path = directory / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model_path = directory / "trained-runtime.json"
    training = train_runtime_manifest(manifest_path, model_path)
    learned = copy.deepcopy(baseline)
    learned["runtimeModel"] = json.loads(model_path.read_text(encoding="utf-8"))
    program_path = directory / "trained-runtime.amo.json"
    program_path.write_text(json.dumps(learned, indent=2), encoding="utf-8")
    (directory / "runtime-baseline.amo.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    rate = 16000
    midi = np.repeat(np.array([[200., .4]], dtype=np.float32), rate, axis=0)
    with NativeRuntime(learned, rate, dll_path) as runtime:
        actual = runtime.render(midi=midi)
        runtime.reset()
        reset = runtime.render(midi=midi)
    original = render_json(baseline, rate, sample_rate=rate, midi=midi, dll_path=dll_path)
    if not np.isfinite(actual).all() or not np.array_equal(actual, reset):
        raise AssertionError("Native recurrent DSP/model output must be finite and reset exactly")
    difference = float(np.max(np.abs(actual - original)))
    if difference <= .05 or training["final_loss"] >= training["initial_loss"]:
        raise AssertionError("Controlled training must reduce loss and affect actual native audio")
    refused = copy.deepcopy(learned)
    refused["runtimeModel"]["budget"]["available_ram_bytes"] = 1
    try:
        with NativeRuntime(refused, rate, dll_path):
            pass
    except ValueError as error:
        rejection = str(error)
    else:
        raise AssertionError("Native compiler must reject insufficient model memory allowance")
    write_wav(directory / "trained-runtime.wav", actual, rate)
    write_wav(directory / "runtime-baseline.wav", original, rate)
    report = {
        "fixture": "controlled runtime compiler experiment; not pretrained audio history",
        "training": training, "frames": rate, "sample_rate": rate,
        "learned_peak": float(np.abs(actual).max()),
        "baseline_peak": float(np.abs(original).max()), "max_audio_difference": difference,
        "reset_exact": True, "reset_max_difference": 0., "low_ram_rejected": rejection,
        "program": str(program_path), "audio": str(directory / "trained-runtime.wav")}
    (directory / "runtime-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dll", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.dll), indent=2))
