"""Explicit small native-rendered fixture exercising train/import/compile/render.

This test fixture covers two sinusoidal mathematical operations and several
harmonic balances. Its model is not a universal WAV recognizer or trained history.
"""
import argparse
import copy
import json
from pathlib import Path
import wave
import numpy as np
from .native import render_json
from .pipeline import train_manifest, infer_wav, export_prediction


def fixture(harmonic_amplitude):
    return {
        "schema": "amo.tensor/1", "ontology": "amo.math/1",
        "tensors": {
            "activation": {"axes": ["primitive"], "shape": [2], "values": [1., 1.]},
            "parameters": {"axes": ["parameter"], "shape": [4], "values": [200., 1., 400., harmonic_amplitude]},
            "routing": {"axes": ["coefficient"], "shape": [0], "values": []},
            "modulation": {"axes": ["relationship"], "shape": [0], "values": []}},
        "instances": [{"primitive": "math.sine", "activation": index,
                       "parameters": {"offset": index * 2, "count": 2},
                       "clock": {"period_samples": 1, "phase_samples": 0},
                       "output_between_updates": "zero", "state_size": 1, "initial_state": [0], "assets": []}
                      for index in range(2)],
        "projections": [], "parameter_locks": [1, 0, 1, 0], "shared_initial_state": [],
        "outputMix": [[0, 0, .3], [1, 0, .3]],
        "budget": {"active_primitives": 2, "state_elements": 2, "parameter_elements": 4,
                   "routing_elements": 0, "modulation_elements": 0, "output_elements": 2,
                   "stored_tensor_elements": 6, "projection_terms": 0}}


def write_wav(path, audio, rate):
    data = np.asarray(audio)
    if data.ndim == 1:
        data = data[:, None]
    if np.max(np.abs(data)) > 1:
        raise ValueError("Fixture WAV would clip")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(data.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.rint(data * 32767).astype("<i2").tobytes())


def run(directory: Path, dll_path=None):
    directory.mkdir(parents=True, exist_ok=True)
    rate = 16000
    examples = []
    for group, canonical_value in enumerate((.15, .35, .55)):
        canonical = fixture(canonical_value)
        canonical_name = f"canonical-{group}.json"
        (directory / canonical_name).write_text(json.dumps(canonical), encoding="utf-8")
        for variation, delta in enumerate((-.02, 0., .02)):
            instance = fixture(canonical_value + delta)
            name = f"group-{group}-variant-{variation}"
            (directory / f"{name}.json").write_text(json.dumps(instance), encoding="utf-8")
            raw = render_json(instance, rate, sample_rate=rate, dll_path=dll_path)
            mix = .3 * raw.sum(axis=1)
            write_wav(directory / f"{name}.wav", mix, rate)
            examples.append({"id": name, "canonical_id": str(group), "perturbation_id": str(variation),
                             "normalized_wav": f"{name}.wav", "canonical_json": canonical_name,
                             "instance_json": f"{name}.json", "normalization": {
                                 "policy_id": "controlled-render-pitch-relative-feature-fixture",
                                 "removed": ["pitch", "loudness", "recording_condition"],
                                 "preserved": ["timbre", "articulation", "temporal_behavior"]}})
    template = fixture(.35)
    manifest = {"ontology": "amo.math/1", "schemas": {name: {"axes": tensor["axes"], "padded_shape": tensor["shape"]}
                for name, tensor in template["tensors"].items()}, "examples": examples,
                "analysis": {"frames": 4, "harmonic_ratios": [1., 2., 3.], "minimum_fundamental_hz": 100.,
                             "maximum_fundamental_hz": 300., "recording_policy": "controlled_render"},
                "learn": {"hidden_width": 24, "steps": 300, "learning_rate": .02,
                          "primary_weight": 2., "secondary_weight": 1., "mask_weight": .1,
                          "available_ram_bytes": 100000000, "seed": 7}}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model_path = directory / "dual-model.pt"
    metadata = train_manifest(manifest_path, model_path)
    prediction = infer_wav(model_path, directory / "group-1-variant-1.wav", mask_threshold=.5)
    exported = export_prediction(template, prediction, head="primary_canonical", bounds={
        "activation": [(0, 1), (0, 1)], "parameters": [(200, 200), (0, 1), (400, 400), (0, 1)],
        "routing": [], "modulation": []})
    (directory / "learned.amo.json").write_text(json.dumps(exported, indent=2), encoding="utf-8")
    result = render_json(exported, rate, sample_rate=rate, dll_path=dll_path)
    write_wav(directory / "learned.wav", .3 * result.sum(axis=1), rate)
    report = {"examples": len(examples), "initial_loss": metadata["losses"][0], "final_loss": metadata["losses"][-1],
              "recovered_harmonic_amplitude": exported["tensors"]["parameters"]["values"][3],
              "canonical_harmonic_amplitude": .35, "peak": float(np.abs(result).max()),
              "model": str(model_path), "compiled_preset": str(directory / "learned.amo.json"),
              "audio": str(directory / "learned.wav")}
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dll", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.dll), indent=2))
