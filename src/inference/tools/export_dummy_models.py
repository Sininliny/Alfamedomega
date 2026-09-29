#!/usr/bin/env python3
"""Export tiny dummy ONNX models that follow the Alfamedomega model contract.

The real models come from training/alfamedomega_train.py. Until they exist, this script
writes one folder per tier with the same graph names, tensor names and shapes (at toy widths),
plus the shared-folder manifest.json, so the inference host can be built and tested.

The dummy graphs are deterministic and carry one exact pass-through that the latency tests rely
on: every expert output's channel 0 equals input feature 0 (pitch) of the same voice and frame.

Usage: export_dummy_models.py <models_dir> [--tiers lite standard pro studio]
Requires: numpy, onnx (no torch). Contract: src/inference/MODEL_CONTRACT.md
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
MODEL_VERSION = "0.0.0-dummy"

NUM_KNOBS = 32
NUM_METHODS = 9
NUM_NOTE_FEATURES = 8
NUM_GLOBAL_MOD = 4

METHODS = [
    "subtractive", "wavetable", "additive", "spectral", "fm",
    "physical", "granular", "sampling", "hybrid",
]

# Output names are the dotted field names of alfa::dsp::VoiceFrame (src/dsp/include/alfa/dsp/
# ControlFrame.h). Widths per tier follow its TierLimits; OSC is always 1024 harmonics.
# Toy widths. Real widths come from the tier spec; the host reads them from the graphs.
TIERS = {
    #            embed, context, hidden, partials, fm ops, grain layers
    "lite":     dict(E=16, C=16, H=16, partials=16, fm_ops=2, grain_layers=2),
    "standard": dict(E=16, C=16, H=24, partials=32, fm_ops=4, grain_layers=4),
    "pro":      dict(E=24, C=24, H=32, partials=48, fm_ops=6, grain_layers=6),
    "studio":   dict(E=32, C=32, H=48, partials=64, fm_ops=8, grain_layers=8),
}


def expert_outputs(method, t):
    """Which control-frame fields each method's expert writes, and at what width for this tier."""
    n, p, g = t["fm_ops"], t["partials"], t["grain_layers"]
    osc = [("osc.re", 1024), ("osc.im", 1024)]
    noise = [("noise.band", 64)]
    fm = [("fm.ratio", n), ("fm.level", n), ("fm.index", n * n)]
    fields = {
        "subtractive": osc + noise,
        "wavetable": osc,
        "additive": osc,
        "spectral": osc + noise,
        "fm": fm,
        "physical": [("partials.ratio", p), ("partials.amp", p)] + noise,
        "granular": [("grain.source", 2 * g), ("grain.weight", 2 * g), ("grain.position", g),
                     ("grain.rate", g), ("grain.gain", g)],
        "sampling": [("sampler.rate", 1), ("sampler.start", 1), ("sampler.gain", 1),
                     ("sampler.eq", 64)] + osc,
        "hybrid": osc + noise + fm,
    }[method]
    return fields + [("mix", 6), ("attack_tau", 1)]


def weight(rng, name, rows, cols, scale=0.1):
    return numpy_helper.from_array((rng.standard_normal((rows, cols)) * scale).astype(np.float32), name)


def save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="alfamedomega-dummy-export")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def patch_encoder(rng, t):
    E = t["E"]
    nodes = [
        helper.make_node("Concat", ["knobs", "method"], ["x"], axis=1),
        helper.make_node("MatMul", ["x", "W"], ["y"]),
        helper.make_node("Tanh", ["y"], ["embedding"]),
    ]
    return helper.make_graph(nodes, "patch_encoder",
                             [vi("knobs", [1, NUM_KNOBS]), vi("method", [1, NUM_METHODS])],
                             [vi("embedding", [1, E])],
                             [weight(rng, "W", NUM_KNOBS + NUM_METHODS, E)])


def instance_step(rng, t):
    E, C = t["E"], t["C"]
    nodes = [
        helper.make_node("Concat", ["embedding", "global_mod", "state"], ["x"], axis=1),
        helper.make_node("MatMul", ["x", "W"], ["y"]),
        helper.make_node("Tanh", ["y"], ["context"]),
        helper.make_node("Identity", ["context"], ["state_out"]),
    ]
    return helper.make_graph(nodes, "instance_step",
                             [vi("embedding", [1, E]), vi("global_mod", [1, NUM_GLOBAL_MOD]),
                              vi("state", [1, C])],
                             [vi("context", [1, C]), vi("state_out", [1, C])],
                             [weight(rng, "W", E + NUM_GLOBAL_MOD + C, C)])


def voice_core(rng, t):
    C, H = t["C"], t["H"]
    learned = H - NUM_NOTE_FEATURES
    nodes = [
        helper.make_node("MatMul", ["features", "Wf"], ["a"]),
        helper.make_node("MatMul", ["context", "Wc"], ["b"]),     # [1, H-8], broadcasts over voices
        helper.make_node("MatMul", ["state", "Ws"], ["s"]),
        helper.make_node("Add", ["a", "b"], ["ab"]),
        helper.make_node("Add", ["ab", "s"], ["abs"]),
        helper.make_node("Tanh", ["abs"], ["learned"]),
        # The raw note features pass straight through as the first 8 hidden units.
        helper.make_node("Concat", ["features", "learned"], ["hidden"], axis=1),
        helper.make_node("Identity", ["hidden"], ["state_out"]),
    ]
    return helper.make_graph(nodes, "voice_core",
                             [vi("features", ["voices", NUM_NOTE_FEATURES]), vi("context", [1, C]),
                              vi("state", ["voices", H])],
                             [vi("hidden", ["voices", H]), vi("state_out", ["voices", H])],
                             [weight(rng, "Wf", NUM_NOTE_FEATURES, learned),
                              weight(rng, "Wc", C, learned),
                              weight(rng, "Ws", H, learned)])


def expert(rng, t, method):
    H = t["H"]
    nodes, outputs, inits = [], [], []
    for field, size in expert_outputs(method, t):
        w = (rng.standard_normal((H, size)) * 0.1).astype(np.float32)
        w[:, 0] = 0.0
        w[0, 0] = 1.0  # channel 0 = hidden[0] = pitch feature, exactly
        inits.append(numpy_helper.from_array(w, f"W_{field}"))
        nodes.append(helper.make_node("MatMul", ["hidden", f"W_{field}"], [field]))
        outputs.append(vi(field, ["voices", size]))
    return helper.make_graph(nodes, f"expert_{method}", [vi("hidden", ["voices", H])], outputs, inits)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def export(models_dir, tiers):
    models_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for tier in tiers:
        t = TIERS[tier]
        rng = np.random.default_rng(1234 + list(TIERS).index(tier))
        folder = f"{tier}-{MODEL_VERSION}"
        out = models_dir / folder
        out.mkdir(exist_ok=True)
        graphs = {
            "patch_encoder.onnx": patch_encoder(rng, t),
            "instance_step.onnx": instance_step(rng, t),
            "voice_core.onnx": voice_core(rng, t),
        }
        for m in METHODS:
            graphs[f"expert_{m}.onnx"] = expert(rng, t, m)
        files = {}
        for name, graph in graphs.items():
            save(graph, out / name)
            files[name] = sha256(out / name)
        entries.append({"tier": tier, "version": MODEL_VERSION, "folder": folder,
                        "opset": OPSET, "frame_rate_hz": 250, "files": files})
    manifest = {"schema": 1, "models": entries}
    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("models_dir", type=pathlib.Path)
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=list(TIERS))
    a = p.parse_args()
    export(a.models_dir, a.tiers)


if __name__ == "__main__":
    main()
