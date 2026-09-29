#!/usr/bin/env python3
"""Export tiny dummy ONNX models that follow the Alfamedomega model contract.

The real models come from training/alfamedomega_train.py. Until they exist, this script
writes one folder per tier with the same graph names, tensor names and shapes (at toy widths),
plus a manifest.json in the control center's model-store format, so the inference host can be
built and tested. It follows the training export (training/alfamedomega_train.py export()):
optional inputs such as hybrid_parts, f0 and modal_state, expert_hybrid, audio_encoder and
grain_bank.npy.

The dummy graphs are deterministic and carry exact probes the tests rely on. The main one: every
expert field output's channel 0 equals input feature 0 (pitch) of the same voice and frame. The
others are listed above expert_outputs().

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
# ControlFrame.h). Toy widths: real widths come from the tier spec; the host reads them from the
# graphs. partials, fm_ops and grain_layers match the DSP's TierLimits. Hi (instance state) is
# deliberately different from C (context), as in training.
TIERS = {
    #            embed, context, inst. state, hidden, partials, fm ops, grain layers
    "lite":     dict(E=16, C=16, Hi=24, H=16, partials=16, fm_ops=2, grain_layers=2),
    "standard": dict(E=16, C=16, Hi=24, H=24, partials=32, fm_ops=4, grain_layers=4),
    "pro":      dict(E=24, C=24, Hi=32, H=32, partials=48, fm_ops=6, grain_layers=6),
    "studio":   dict(E=32, C=32, Hi=40, H=48, partials=64, fm_ops=8, grain_layers=8),
}
DEFAULT_TIER = "standard"
SAMPLE_RATE = 48000
FRAME_HZ = 250
GRAIN_BANK_ROWS = 8
GRAIN_SOURCE_LENGTH = 8192
MEL_BANDS = 64

# Experts whose f0 input training's export prunes because their builder ignores it.
NO_F0 = {"wavetable", "fm"}

# Exact probes the tests rely on (besides channel 0 = pitch in every field output):
#   mix[2] = f0 for experts that take f0;
#   physical: modal_state_out = modal_state + 1 and partials.amp[1] = modal_state[0], so it
#   counts frames since the voice's state was last cleared;
#   the patch embedding ignores method and hybrid_parts (so hybrid tests can compare against
#   each component alone), and expert_hybrid passes hidden through to both components with a
#   fixed crossfade xf = HYBRID_XF.
F0_CHANNEL = 2
HYBRID_XF = 0.25


def expert_outputs(method, t):
    """Which control-frame fields each method's expert writes, and at what width for this tier."""
    n, p, g = t["fm_ops"], t["partials"], t["grain_layers"]
    osc = [("osc.re", 1024), ("osc.im", 1024)]
    noise = [("noise.band", 64)]
    partials = [("partials.ratio", p), ("partials.amp", p)]
    fm = [("fm.ratio", n), ("fm.level", n), ("fm.index", n * n)]
    fields = {
        "subtractive": osc + noise,
        "wavetable": osc,
        "additive": partials,
        "spectral": osc + noise,
        "fm": fm,
        "physical": partials + noise,
        "granular": [("grain.source", 2 * g), ("grain.weight", 2 * g), ("grain.position", g),
                     ("grain.rate", g), ("grain.gain", g)],
        "sampling": [("sampler.rate", 1), ("sampler.start", 1), ("sampler.gain", 1),
                     ("sampler.eq", 64)] + osc,
    }[method]
    # Training's export sorts the outputs by name.
    return sorted(fields + [("mix", 6), ("attack_tau", 1)])


def weight(rng, name, rows, cols, scale=0.1):
    return numpy_helper.from_array((rng.standard_normal((rows, cols)) * scale).astype(np.float32), name)


def const(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


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
    w = (rng.standard_normal((NUM_KNOBS + 3 * NUM_METHODS, E)) * 0.1).astype(np.float32)
    w[NUM_KNOBS:] = 0.0  # method and hybrid_parts do not move the dummy embedding
    nodes = [
        helper.make_node("Concat", ["knobs", "method", "hybrid_parts"], ["x"], axis=1),
        helper.make_node("MatMul", ["x", "W"], ["y"]),
        helper.make_node("Tanh", ["y"], ["embedding"]),
    ]
    return helper.make_graph(nodes, "patch_encoder",
                             [vi("knobs", [1, NUM_KNOBS]), vi("method", [1, NUM_METHODS]),
                              vi("hybrid_parts", [1, 2 * NUM_METHODS])],
                             [vi("embedding", [1, E])],
                             [numpy_helper.from_array(w, "W")])


def audio_encoder(rng, t):
    E = t["E"]
    nodes = [
        helper.make_node("ReduceMean", ["logmel"], ["m"], axes=[2], keepdims=0),  # [1, 64]
        helper.make_node("MatMul", ["m", "W"], ["y"]),
        helper.make_node("Tanh", ["y"], ["e_audio"]),
    ]
    return helper.make_graph(nodes, "audio_encoder", [vi("logmel", [1, MEL_BANDS, "frames"])],
                             [vi("e_audio", [1, E])], [weight(rng, "W", MEL_BANDS, E, 0.05)])


def instance_step(rng, t):
    E, C, Hi = t["E"], t["C"], t["Hi"]
    nodes = [
        helper.make_node("Concat", ["embedding", "global_mod", "state"], ["x"], axis=1),
        helper.make_node("MatMul", ["x", "Wc"], ["c"]),
        helper.make_node("Tanh", ["c"], ["context"]),
        helper.make_node("MatMul", ["x", "Ws"], ["s"]),
        helper.make_node("Tanh", ["s"], ["state_out"]),
    ]
    return helper.make_graph(nodes, "instance_step",
                             [vi("embedding", [1, E]), vi("global_mod", [1, NUM_GLOBAL_MOD]),
                              vi("state", [1, Hi])],
                             [vi("context", [1, C]), vi("state_out", [1, Hi])],
                             [weight(rng, "Wc", E + NUM_GLOBAL_MOD + Hi, C),
                              weight(rng, "Ws", E + NUM_GLOBAL_MOD + Hi, Hi)])


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
                             [vi("features", ["V", NUM_NOTE_FEATURES]), vi("context", [1, C]),
                              vi("state", ["V", H])],
                             [vi("hidden", ["V", H]), vi("state_out", ["V", H])],
                             [weight(rng, "Wf", NUM_NOTE_FEATURES, learned),
                              weight(rng, "Wc", C, learned),
                              weight(rng, "Ws", H, learned)])


def expert(rng, t, method):
    H, M = t["H"], t["partials"]
    takes_f0 = method not in NO_F0
    modal = method == "physical"
    inputs = [vi("hidden", ["V", H])]
    nodes, outputs, inits = [], [], []
    if takes_f0:
        inputs.append(vi("f0", ["V"]))
        nodes.append(helper.make_node("Unsqueeze", ["f0", "axis1"], ["f0col"]))
        inits.append(const("axis1", np.array([1], dtype=np.int64)))
    if modal:
        inputs.append(vi("modal_state", ["V", M]))
    for field, size in expert_outputs(method, t):
        w = (rng.standard_normal((H, size)) * 0.1).astype(np.float32)
        w[:, 0] = 0.0
        w[0, 0] = 1.0  # channel 0 = hidden[0] = pitch feature, exactly
        extra = []
        if field == "mix" and takes_f0:
            w[:, F0_CHANNEL] = 0.0
            e = np.zeros((1, size), np.float32)
            e[0, F0_CHANNEL] = 1.0
            inits.append(const("E_f0", e))
            nodes.append(helper.make_node("MatMul", ["f0col", "E_f0"], ["mix_f0"]))
            extra.append("mix_f0")
        if field == "partials.amp" and modal:
            w[:, 1] = 0.0
            e = np.zeros((M, size), np.float32)
            e[0, 1] = 1.0
            inits.append(const("E_modal", e))
            nodes.append(helper.make_node("MatMul", ["modal_state", "E_modal"], ["amp_modal"]))
            extra.append("amp_modal")
        inits.append(numpy_helper.from_array(w, f"W_{field}"))
        base = f"{field}_base" if extra else field
        nodes.append(helper.make_node("MatMul", ["hidden", f"W_{field}"], [base]))
        if extra:
            nodes.append(helper.make_node("Add", [base, extra[0]], [field]))
        outputs.append(vi(field, ["V", size]))
    if modal:
        inits.append(const("one", np.array([1.0], np.float32)))
        nodes.append(helper.make_node("Add", ["modal_state", "one"], ["modal_state_out"]))
        outputs.append(vi("modal_state_out", ["V", M]))
    return helper.make_graph(nodes, f"expert_{method}", inputs, outputs, inits)


def expert_hybrid(t):
    H = t["H"]
    nodes = [
        helper.make_node("Identity", ["hidden"], ["hidden_a"]),
        helper.make_node("Identity", ["hidden"], ["hidden_b"]),
        # xf = HYBRID_XF for every voice, computed from hidden so it carries the voices dimension.
        helper.make_node("MatMul", ["hidden", "Z"], ["z"]),            # [V, 1] of zeros
        helper.make_node("Squeeze", ["z", "axis1"], ["zv"]),
        helper.make_node("Add", ["zv", "xf0"], ["xf"]),
    ]
    return helper.make_graph(nodes, "expert_hybrid", [vi("hidden", ["V", H])],
                             [vi("hidden_a", ["V", H]), vi("hidden_b", ["V", H]), vi("xf", ["V"])],
                             [const("Z", np.zeros((H, 1), np.float32)),
                              const("axis1", np.array([1], dtype=np.int64)),
                              const("xf0", np.array([HYBRID_XF], np.float32))])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def logmel_reference(x):
    """Independent numpy version of training's logmel(), for the host's LogMel test."""
    n_fft, hop = 2048, 512
    xp = np.pad(x.astype(np.float64), n_fft // 2, mode="reflect")
    frames = 1 + len(x) // hop
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    spec = np.stack([np.abs(np.fft.rfft(xp[i * hop:i * hop + n_fft] * window)) for i in range(frames)], 1)

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    pts = 700.0 * (10 ** (np.linspace(hz_to_mel(20.0), hz_to_mel(16000.0), MEL_BANDS + 2) / 2595.0) - 1)
    f = np.arange(n_fft // 2 + 1) * SAMPLE_RATE / n_fft
    lo, c, hi = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    fb = np.maximum(0.0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)))
    return np.log(fb @ spec + 1e-5).astype(np.float32)


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
            "audio_encoder.onnx": audio_encoder(rng, t),
            "instance_step.onnx": instance_step(rng, t),
            "voice_core.onnx": voice_core(rng, t),
        }
        for m in METHODS:
            if m != "hybrid":
                graphs[f"expert_{m}.onnx"] = expert(rng, t, m)
        graphs["expert_hybrid.onnx"] = expert_hybrid(t)
        for name, graph in graphs.items():
            save(graph, out / name)
        bank = (rng.standard_normal((GRAIN_BANK_ROWS, GRAIN_SOURCE_LENGTH)) * 0.1).astype(np.float32)
        np.save(out / "grain_bank.npy", bank)
        files = [{"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(out.iterdir()) if p.is_file()]
        entries.append({"id": f"{tier}-{MODEL_VERSION}", "tier": tier, "version": MODEL_VERSION,
                        "dir": folder, "files": files, "bytes": sum(f["bytes"] for f in files),
                        "sr": SAMPLE_RATE, "frame_hz": FRAME_HZ, "hop": SAMPLE_RATE // FRAME_HZ})
    # The control center's model-store manifest (docs/control-center/model-store.md).
    manifest = {"schema": 1, "default_tier": DEFAULT_TIER, "models": entries}
    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Not a model: a golden log-mel spectrogram for the host's LogMel test.
    n = np.arange(24000)
    x = (0.5 * np.sin(2 * np.pi * 440.0 * n / SAMPLE_RATE)
         + 0.1 * np.random.default_rng(7).standard_normal(len(n))).astype(np.float32)
    np.save(models_dir / "logmel_input.npy", x[None, :])
    np.save(models_dir / "logmel_reference.npy", logmel_reference(x))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("models_dir", type=pathlib.Path)
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=list(TIERS))
    a = p.parse_args()
    export(a.models_dir, a.tiers)


if __name__ == "__main__":
    main()
