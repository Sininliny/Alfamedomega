#!/usr/bin/env python3
"""Synthetic multi-instrument scenes for the mix -> presets model (MIX-DESIGN.md §3).

A scene is a 4-8 s phrase played by 1-8 instruments. Every instrument is a corpus
construction (``prepare.make_document``) plus an insert FX chain that is written
into the same AMO document, so each L0 instrument label is one renderable preset.
The scene mix goes through a bus FX document (its own FX preset).

Hierarchical labels ("ladder"), all exact by construction:

- L0: layered presets merged, note echo kept as an FX of its instrument;
- L1: the layers of a layered preset are separate instruments;
- L2: every note-echo tap is also a separate instrument (delayed, transposed notes).

Each scene is rendered as *atoms* (one layer x one echo tap). A level's instrument
is a sum of atoms, so every level reproduces the same pre-bus mix exactly.

Instruments are monophonic: corpus constructions read MIDI row 0 (not the
voice_indexed layout), so that is what the plugin can play back from them.

Example::

    python -m training.amo.corpus.mixscene build DEST --scenes 2000 --workers 4
    python -m training.amo.corpus.mixscene listen DEST --count 4 --out wavs/
"""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus import prepare  # noqa: E402
from training.amo.native import NativeRuntime  # noqa: E402

SCHEMA = "amo.mix-scene/1"
SEED = 2026110601
SAMPLE_RATE = 24000
MAX_INSTRUMENTS = 8  # Sinin 2026-11-06: up to 8 instruments, 8 slots
COUNT_WEIGHTS = np.array([0.10, 0.18, 0.20, 0.17, 0.13, 0.10, 0.07, 0.05])
SECONDS = (4.0, 8.0)

# Mix features: hop 512 at 24 kHz (21.3 ms), Mid 128 + Side 32 log-frequency bands.
N_FFT, HOP = 2048, 512
MID_BANDS, SIDE_BANDS = 128, 32
F_LOW, F_HIGH = 30.0, 12000.0
MAX_FRAMES = int(np.ceil(SECONDS[1] * SAMPLE_RATE / HOP)) + 1
# Atom stems (auxiliary separation targets) at a coarser grid.
STEM_HOP, STEM_BANDS = 2048, 64
MAX_STEM_FRAMES = int(np.ceil(SECONDS[1] * SAMPLE_RATE / STEM_HOP)) + 1
LOG_FLOOR = 1e-10

PITCHED = [n for n in prepare.NAMES[:24] if n not in (
    "noise_bandpass", "noise_modal", "impulse_modal4", "impulse_modal12")]
PERCUSSIVE = ["noise_bandpass", "noise_modal", "impulse_modal4", "impulse_modal12"]
ROLES = {  # register (MIDI notes) of each pitched role; corpus constructions stay below Nyquist here
    "bass": (28, 50),
    "pad": (45, 70),
    "lead": (55, 76),
    "arp": (52, 76),
    "perc": (38, 74),
}
ROLE_WEIGHTS = {"bass": 0.22, "pad": 0.18, "lead": 0.22, "arp": 0.16, "perc": 0.22}
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "pentatonic": [0, 3, 5, 7, 10],
}
INSERT_FX = ["lowpass", "highpass", "bandpass", "tanh", "phaser", "fir", "echo"]
INSERT_FX_WEIGHTS = np.array([0.20, 0.10, 0.10, 0.17, 0.10, 0.08, 0.25])
ECHO_TRANSPOSE = [0, 0, 3, 4, 5, 7, 12, 12, -12, 19, 24]
BUS_FX = ["lowpass", "highpass", "tanh", "width", "reverb", "echo"]
BUS_FX_PROBABILITY = {"lowpass": 0.25, "highpass": 0.15, "tanh": 0.2, "width": 0.3,
                      "reverb": 0.7, "echo": 0.3}
GATE_GAP = 0.006  # seconds of gate-off before a legato note so envelopes retrigger
PEAK = 0.85


# --------------------------------------------------------------------- documents


class Editor:
    """Append nodes and weights to a finished AMO document, keeping budgets right."""

    def __init__(self, doc):
        self.doc = copy.deepcopy(doc)
        t = self.doc["tensors"]
        for name in ("activation", "parameters", "routing", "modulation"):
            t.setdefault(name, {"axes": [{"activation": "primitive", "parameters": "parameter",
                                          "routing": "coefficient", "modulation": "relationship"}[name]],
                                "shape": [0], "values": [], "presence": []})
        self.doc.setdefault("budget", empty_budget())

    def _append(self, name, value):
        tensor = self.doc["tensors"][name]
        tensor["values"].append(float(value))
        tensor["presence"].append(True)
        tensor["shape"] = [len(tensor["values"])]
        return len(tensor["values"]) - 1

    def node(self, op, parameters, state_size=0, outputs=1, assets=None, initial=None):
        budget = self.doc["budget"]
        index = len(self.doc["instances"])
        offset = len(self.doc["tensors"]["parameters"]["values"])
        activation = self._append("activation", 1.0)
        for value in parameters:
            self._append("parameters", value)
            self.doc.setdefault("parameter_locks", []).append(False)
        self.doc["instances"].append({
            "primitive": op,
            "instance_id": f"node-{index}",
            "activation": activation,
            "parameters": {"offset": offset, "count": len(parameters)},
            "clock": {"period_samples": 1, "phase_samples": 0},
            "output_between_updates": "hold",
            "state_size": state_size,
            "initial_state": initial if initial is not None else [0.0] * state_size,
            "assets": [prepare.asset(a) for a in (assets or [])],
            "input_channels": 0,
            "audio_bindings": [],
        })
        start = budget["output_elements"]
        budget["active_primitives"] += 1
        budget["state_elements"] += state_size
        budget["parameter_elements"] += len(parameters)
        budget["output_elements"] += outputs
        budget["stored_tensor_elements"] += 1 + len(parameters) + sum(len(a) for a in (assets or []))
        return index, start

    def bind(self, node, source, element, destination, weight):
        index = self._append("routing", weight)
        target = self.doc["instances"][node]
        target["input_channels"] = max(target["input_channels"], destination + 1)
        target["audio_bindings"].append({
            "source": source, "source_element": int(element),
            "destination_element": int(destination), "weights": "routing",
            "weight_element": index, "missing_source": "zero"})
        budget = self.doc["budget"]
        budget["routing_elements"] += 1
        budget["projection_terms"] += 1
        budget["stored_tensor_elements"] += 2

    def mono_fx(self, kind, params, sources):
        """One mono insert stage reading the weighted sum ``sources`` [(output, weight)]."""
        if kind in ("lowpass", "highpass", "bandpass"):
            node, out = self.node("filter.biquad_" + kind, [params["cutoff"], params["q"]], 4)
        elif kind == "tanh":
            node, out = self.node("nonlinear.tanh.v1",
                                  [params["drive"], params["bias"], params["gain"]], 0)
        elif kind == "phaser":
            node, out = self.node("time.phaser_one_stage", [params["coefficient"], params["wet"]], 2)
        elif kind == "fir":
            taps = list(params["taps"])
            node, out = self.node("time.causal_fir_ring.v1", [], len(taps) + 2, assets=[taps])
        elif kind == "delay":
            delay = int(params["delay"])
            node, out = self.node("time.ring_delay.v1", [delay, params["feedback"]], delay + 3)
        else:
            raise ValueError(kind)
        for source, weight in sources:
            self.bind(node, "output_history", source, 0, weight)
        return out


def empty_budget():
    return {"active_primitives": 0, "state_elements": 0, "parameter_elements": 0,
            "routing_elements": 0, "modulation_elements": 0, "output_elements": 0,
            "stored_tensor_elements": 0, "projection_terms": 0,
            "sparse_state_writes": 256, "scratch_elements": 4096}


def empty_document():
    return {"schema": "amo.tensor/1", "ontology": "amo.math/1", "tensors": {},
            "instances": [], "projections": [], "parameter_locks": [],
            "shared_initial_state": [], "outputMix": [], "budget": empty_budget()}


def transpose(doc, semitones):
    """Scale every MIDI-frequency projection so the construction sounds ``semitones`` higher."""
    if not semitones:
        return doc
    doc = copy.deepcopy(doc)
    ratio = 2.0 ** (semitones / 12.0)
    for p in doc["projections"]:
        if p["source"] == "midi" and p["source_element"] == 0:
            values = doc["tensors"][p["weights"]]["values"]
            values[p["weight_element"]] *= ratio
    return doc


def insert_chain(doc, chain):
    """Write a mono insert FX chain between the construction's gain stage and its pan node."""
    if not chain:
        return copy.deepcopy(doc)
    editor = Editor(doc)
    pans = [i for i, n in enumerate(editor.doc["instances"])
            if n["primitive"] == "spatial.constant_power_pan"]
    if len(pans) != 1:
        raise ValueError("insert FX needs exactly one pan node")
    pan = editor.doc["instances"][pans[0]]
    routing = editor.doc["tensors"]["routing"]["values"]
    current = [(b["source_element"], routing[b["weight_element"]]) for b in pan["audio_bindings"]]
    for fx in chain:
        if fx["type"] == "echo":
            wet = editor.mono_fx("delay", fx, current)
            current = current + [(wet, fx["wet"])]
        else:
            current = [(editor.mono_fx(fx["type"], fx, current), 1.0)]
    pan["audio_bindings"] = []
    pan["input_channels"] = 0
    for source, weight in current:
        editor.bind(pans[0], "output_history", source, 0, weight)
    return editor.doc


def merge(docs):
    """Parallel layers in one document: every index is offset; MIDI is shared."""
    result = empty_document()
    names = ("activation", "parameters", "routing", "modulation")
    for name in names:
        result["tensors"][name] = {"axes": copy.deepcopy(docs[0]["tensors"][name]["axes"]),
                                   "shape": [0], "values": [], "presence": []}
    for doc in docs:
        doc = copy.deepcopy(doc)
        offsets = {n: len(result["tensors"][n]["values"]) for n in names}
        outputs = result["budget"]["output_elements"]
        state = result["budget"]["state_elements"]

        def moved(source, element):
            if source == "output_history":
                return element + outputs
            if source == "state":
                return element + state
            return element

        for node in doc["instances"]:
            node["activation"] += offsets["activation"]
            node["parameters"]["offset"] += offsets["parameters"]
            node["instance_id"] = f"node-{len(result['instances'])}"
            for b in node["audio_bindings"]:
                b["source_element"] = moved(b["source"], b["source_element"])
                b["weight_element"] += offsets[b["weights"]]
            result["instances"].append(node)
        for p in doc["projections"]:
            p["source_element"] = moved(p["source"], p["source_element"])
            p["parameter_element"] += offsets["parameters"]
            p["weight_element"] += offsets[p["weights"]]
            result["projections"].append(p)
        for name in names:
            tensor = doc["tensors"][name]
            result["tensors"][name]["values"] += tensor["values"]
            result["tensors"][name]["presence"] += tensor.get("presence", [True] * len(tensor["values"]))
        result["parameter_locks"] += doc.get("parameter_locks", [])
        result["shared_initial_state"] += doc.get("shared_initial_state", [])
        result["outputMix"] += [[i + outputs, c, g] for i, c, g in doc["outputMix"]]
        for key, value in doc["budget"].items():
            if key in ("sparse_state_writes", "scratch_elements"):
                result["budget"][key] = max(result["budget"][key], value)
            else:
                result["budget"][key] += value
    for name in names:
        result["tensors"][name]["shape"] = [len(result["tensors"][name]["values"])]
    return result


def bus_document(chain):
    """Stereo bus FX preset: dual-mono inserts, width, then reverb/echo sends."""
    editor = Editor(empty_document())
    ones = [1.0]
    sides = []
    for channel in (0, 1):  # exact identity taps so the dry path is an output element
        node, out = editor.node("time.causal_fir_ring.v1", [], 3, assets=[ones])
        editor.bind(node, "audio", channel, 0, 1.0)
        sides.append(out)
    sends = []
    for fx in chain:
        kind = fx["type"]
        if kind in ("lowpass", "highpass", "tanh"):
            sides = [editor.mono_fx(kind, fx, [(s, 1.0)]) for s in sides]
        elif kind == "width":
            node, out = editor.node("spatial.stereo_width", [fx["width"]], 0, outputs=2)
            editor.bind(node, "output_history", sides[0], 0, 1.0)
            editor.bind(node, "output_history", sides[1], 1, 1.0)
            sides = [out, out + 1]
        elif kind == "reverb":
            params = []
            for delay, gain, damping in zip(fx["delays"], fx["gains"], fx["damping"]):
                params += [delay, gain, damping]
            capacity = int(max(fx["delays"])) + 2
            node, out = editor.node("time.fdn_hadamard8.v1", params, 8 * (capacity + 3), outputs=8)
            for line in range(8):
                editor.bind(node, "output_history", sides[line % 2], line, fx["input"][line])
            sends += [(out + line, line % 2, fx["wet"] / 4) for line in range(8)]
        elif kind == "echo":
            for channel in (0, 1):
                delay = int(fx["delays"][channel])
                node, out = editor.node("time.ring_delay.v1", [delay, fx["feedback"]], delay + 3)
                editor.bind(node, "output_history", sides[channel], 0, 1.0)
                sends.append((out, channel, fx["wet"]))
        else:
            raise ValueError(kind)
    editor.doc["outputMix"] = [[sides[0], 0, 1.0], [sides[1], 1, 1.0]] + [list(s) for s in sends]
    return editor.doc


# ------------------------------------------------------------------------- sampling


def sample_notes(rng, role, bpm, root, scale, seconds, source=None):
    """Monophonic note list [[onset s, duration s, MIDI note, velocity]], sorted, non-overlapping."""
    if source is not None:  # doubling: play the same stream (optionally an octave away)
        return [list(n) for n in source]
    beat = 60.0 / bpm
    low, high = ROLES[role]
    pitches = [p for p in range(low, high + 1) if (p - root) % 12 in scale]
    notes = []
    t = float(rng.choice([0.0, 0.0, beat / 2, beat]))
    index = int(rng.integers(len(pitches) // 4, max(len(pitches) // 4 + 1, 3 * len(pitches) // 4)))
    hit = int(rng.integers(len(pitches)))
    while t < seconds - 0.1:
        if role == "bass":
            step = beat * float(rng.choice([0.5, 1.0, 1.0, 2.0]))
            length = step * float(rng.uniform(0.5, 0.95))
            index = int(np.clip(index + rng.choice([-2, 0, 0, 2, 4, -4]), 0, len(pitches) - 1))
        elif role == "pad":
            step = beat * float(rng.choice([2.0, 4.0]))
            length = step * float(rng.uniform(0.8, 0.98))
            index = int(np.clip(index + rng.choice([-3, -2, 2, 3]), 0, len(pitches) - 1))
        elif role == "lead":
            step = beat * float(rng.choice([0.25, 0.5, 0.5, 1.0, 1.5]))
            length = step * float(rng.uniform(0.4, 0.95))
            index = int(np.clip(index + rng.choice([-2, -1, -1, 1, 1, 2, 3]), 0, len(pitches) - 1))
        elif role == "arp":
            step = beat * float(rng.choice([0.25, 0.5]))
            length = step * float(rng.uniform(0.3, 0.8))
            index = (index + int(rng.choice([2, 2, 3, -5]))) % len(pitches)
        else:  # perc: fixed pitch hits on a rhythm grid
            step = beat * float(rng.choice([0.25, 0.5, 1.0]))
            length = min(step * 0.5, 0.12)
            index = hit
        if rng.random() > (0.12 if role != "perc" else 0.2):
            notes.append([t, min(length, seconds - t - 0.02), int(pitches[index]),
                          float(rng.uniform(0.4, 0.95))])
        t += step
    if not notes:
        notes.append([0.0, min(beat, seconds - 0.05), int(pitches[index]), 0.8])
    return notes


def sample_insert_fx(rng):
    chain = []
    for _ in range(int(rng.choice([0, 0, 1, 1, 1, 2]))):
        kind = str(rng.choice(INSERT_FX, p=INSERT_FX_WEIGHTS / INSERT_FX_WEIGHTS.sum()))
        if any(f["type"] == kind for f in chain):
            continue
        chain.append(sample_fx(rng, kind))
    # Canonical order (INSERT_FX order, so time FX last): the chain is a set, which keeps
    # labels identifiable; the construction families already cover tanh->filter orderings.
    return sorted(chain, key=lambda f: INSERT_FX.index(f["type"]))


def sample_fx(rng, kind, bus=False):
    if kind in ("lowpass", "highpass", "bandpass"):
        span = {"lowpass": (500.0, 9000.0), "highpass": (60.0, 1500.0), "bandpass": (250.0, 5000.0)}[kind]
        return {"type": kind, "cutoff": float(np.exp(rng.uniform(*np.log(span)))),
                "q": float(np.exp(rng.uniform(np.log(0.5), np.log(1.6 if bus else 3.0))))}
    if kind == "tanh":
        return {"type": kind, "drive": float(np.exp(rng.uniform(0.0, np.log(4.0 if bus else 8.0)))),
                "bias": float(rng.uniform(-0.1, 0.1)), "gain": float(rng.uniform(0.3, 0.8))}
    if kind == "phaser":
        return {"type": kind, "coefficient": float(rng.uniform(-0.9, 0.9)), "wet": float(rng.uniform(0.2, 0.9))}
    if kind == "fir":
        taps = np.zeros(12)
        taps[0] = 0.6
        taps[1:] = rng.uniform(-0.15, 0.15, 11) * np.exp(-np.arange(1, 12) / 4.0)
        return {"type": kind, "taps": (taps / (1 + np.abs(taps).sum())).tolist()}
    if kind == "echo":
        return {"type": kind, "delay": int(rng.uniform(0.06, 0.45) * SAMPLE_RATE),
                "feedback": float(rng.uniform(0.0, 0.6)), "wet": float(rng.uniform(0.2, 0.6))}
    if kind == "width":
        return {"type": kind, "width": float(rng.uniform(0.3, 1.8))}
    if kind == "reverb":
        t60 = float(np.exp(rng.uniform(np.log(0.4), np.log(3.5))))
        size = float(rng.uniform(0.7, 1.5))
        return reverb_spec(t60, size, float(rng.uniform(0.05, 0.6)), float(rng.uniform(0.08, 0.35)))
    if kind == "echo_bus":
        base = rng.uniform(0.12, 0.5)
        return {"type": "echo", "delays": [int(base * SAMPLE_RATE), int(base * rng.uniform(0.6, 1.5) * SAMPLE_RATE)],
                "feedback": float(rng.uniform(0.1, 0.55)), "wet": float(rng.uniform(0.1, 0.35))}
    raise ValueError(kind)


REVERB_DELAYS = [601, 761, 953, 1201, 1433, 1747, 2111, 2351]  # mutually prime line lengths at size 1
REVERB_SIGNS = [1, -1, 1, -1, -1, 1, -1, 1]


def reverb_spec(t60, size, damping, wet):
    """FDN reverb fully described by (t60, size, damping, wet), so it is learnable exactly."""
    delays = [int(round(d * size)) for d in REVERB_DELAYS]
    return {"type": "reverb", "t60": t60, "size": size, "delays": delays,
            "gains": [float(10 ** (-3 * d / (SAMPLE_RATE * t60))) for d in delays],
            "damping": [damping] * 8, "input": [0.35 * s for s in REVERB_SIGNS], "wet": wet}


def sample_scene(seed, index, count_weights=None):
    """Scene specification only (no audio). Deterministic in (seed, index, count_weights)."""
    rng = np.random.default_rng([seed, index])
    weights = COUNT_WEIGHTS if count_weights is None else np.asarray(count_weights, float)
    weights = np.pad(weights, (0, MAX_INSTRUMENTS - len(weights)))[:MAX_INSTRUMENTS]
    seconds = float(rng.uniform(*SECONDS))
    bpm = float(rng.uniform(70, 160))
    root = int(rng.integers(12))
    scale_name = str(rng.choice(list(SCALES)))
    scale = SCALES[scale_name]
    natural = int(rng.choice(np.arange(1, MAX_INSTRUMENTS + 1), p=weights / weights.sum()))
    instruments = []
    atoms = 0
    for k in range(natural):
        role = str(rng.choice(list(ROLE_WEIGHTS), p=np.array(list(ROLE_WEIGHTS.values()))))
        families = PERCUSSIVE if role == "perc" else PITCHED
        layers = [{"family": str(rng.choice(families)), "u": rng.random(64).round(6).tolist(), "transpose": 0}]
        doubling = None
        hard = rng.random()
        if role != "perc" and hard < 0.12 and atoms + 2 <= MAX_INSTRUMENTS - (natural - k - 1):
            # Layered preset (one instrument at L0, two at L1): unison or octave layer.
            layers.append({"family": str(rng.choice(families)), "u": rng.random(64).round(6).tolist(),
                           "transpose": int(rng.choice([0, 12, -12]))})
        elif role != "perc" and hard < 0.22 and instruments:
            # Different timbre doubling an existing part: a real second instrument at every level.
            candidates = [i for i, inst in enumerate(instruments) if inst["role"] != "perc"]
            if candidates:
                doubling = int(rng.choice(candidates))
        note_echo = None
        if len(layers) == 1 and hard >= 0.22 and hard < 0.40:
            taps = int(rng.integers(1, 4))
            taps = min(taps, MAX_INSTRUMENTS - atoms - 1 - (natural - k - 1))
            if taps >= 1:
                beat = 60.0 / bpm
                note_echo = {"taps": taps, "delay": float(beat * rng.choice([0.25, 0.5, 0.75, 1.0])),
                             "transpose": int(rng.choice(ECHO_TRANSPOSE)),
                             "decay": float(rng.uniform(0.35, 0.75))}
        source = instruments[doubling]["notes"] if doubling is not None else None
        notes = sample_notes(rng, role, bpm, root, scale, seconds, source)
        if doubling is not None and rng.random() < 0.5:
            notes = [[a, b, int(np.clip(p + 12, 28, 76)), v] for a, b, p, v in notes]
        instruments.append({
            "role": role, "layers": layers, "insert_fx": sample_insert_fx(rng),
            "note_echo": note_echo, "doubles": doubling, "notes": notes,
            "gain_db": float(rng.uniform(-18.0, 0.0)),
        })
        atoms += len(layers) * (1 + (note_echo["taps"] if note_echo else 0))
    bus = [sample_fx(rng, "echo_bus" if kind == "echo" else kind, bus=True)
           for kind in BUS_FX if rng.random() < BUS_FX_PROBABILITY[kind]]
    order = {"lowpass": 0, "highpass": 0, "tanh": 1, "width": 2, "reverb": 3, "echo": 4}
    bus.sort(key=lambda f: order[f["type"]])
    return {"schema": SCHEMA, "seed": seed, "index": index, "sample_rate": SAMPLE_RATE,
            "seconds": seconds, "bpm": bpm, "root": root, "scale": scale_name,
            "instruments": instruments, "bus_fx": bus}


# --------------------------------------------------------------------------- ladder


def echo_notes(notes, echo, tap):
    """Notes of note-echo tap ``tap`` (1-based): delayed, transposed, inside the clip."""
    shift = echo["delay"] * tap
    semis = echo["transpose"] * tap
    out = []
    for onset, length, pitch, velocity in notes:
        pitch = pitch + semis
        if 24 <= pitch <= 88:
            out.append([onset + shift, length, pitch, velocity])
    return out


def atoms_of(scene):
    """Finest units: (instrument, layer, tap) with their note list and linear gain."""
    atoms = []
    for k, inst in enumerate(scene["instruments"]):
        taps = inst["note_echo"]["taps"] if inst["note_echo"] else 0
        for layer in range(len(inst["layers"])):
            for tap in range(taps + 1):
                notes = inst["notes"] if tap == 0 else echo_notes(inst["notes"], inst["note_echo"], tap)
                gain = 10 ** (inst["gain_db"] / 20) * (inst["note_echo"]["decay"] ** tap if tap else 1.0)
                atoms.append({"instrument": k, "layer": layer, "tap": tap, "notes": notes, "gain": gain})
    return atoms


def ladder(scene, atoms):
    """Instrument index of every atom at L0, L1 and L2."""
    levels = {"L0": [], "L1": [], "L2": []}
    seen = {"L1": {}, "L2": {}}
    for atom in atoms:
        levels["L0"].append(atom["instrument"])
        levels["L1"].append(seen["L1"].setdefault((atom["instrument"], atom["layer"]), len(seen["L1"])))
        levels["L2"].append(seen["L2"].setdefault((atom["instrument"], atom["layer"], atom["tap"]),
                                                  len(seen["L2"])))
    return levels


def layer_document(inst, layer):
    spec = inst["layers"][layer]
    doc, _, _, _ = prepare.make_document(spec["family"], np.asarray(spec["u"]))
    return insert_chain(transpose(doc, spec["transpose"]), inst["insert_fx"])


def preset_document(inst):
    """The L0 preset of one instrument: all layers (each with the insert chain) in one document."""
    docs = [layer_document(inst, i) for i in range(len(inst["layers"]))]
    return docs[0] if len(docs) == 1 else merge(docs)


def level_presets(scene, level):
    """Preset documents and note lists of every instrument at a ladder level.

    Note echo is not a primitive yet, so an L0/L1 preset with note echo carries it as
    ``note_echo`` metadata next to the document (MIX-DESIGN.md §8).
    """
    result = []
    for inst in scene["instruments"]:
        if level == "L0":
            result.append({"document": preset_document(inst), "notes": inst["notes"],
                           "note_echo": inst["note_echo"], "gain_db": inst["gain_db"]})
            continue
        for layer in range(len(inst["layers"])):
            taps = inst["note_echo"]["taps"] if inst["note_echo"] and level == "L2" else 0
            for tap in range(taps + 1):
                notes = inst["notes"] if tap == 0 else echo_notes(inst["notes"], inst["note_echo"], tap)
                gain = inst["gain_db"] + (20 * np.log10(inst["note_echo"]["decay"]) * tap if tap else 0.0)
                result.append({"document": layer_document(inst, layer), "notes": notes,
                               "note_echo": inst["note_echo"] if level == "L1" else None,
                               "gain_db": float(gain)})
    return result


# ------------------------------------------------------------------------ rendering


def midi_tensor(notes, frames, sr=SAMPLE_RATE):
    """[frequency Hz, velocity, gate] per sample for a monophonic note list."""
    midi = np.zeros((frames, 3), np.float32)
    notes = sorted(notes)
    gap = int(GATE_GAP * sr)
    frequency = 440.0 * 2 ** ((notes[0][2] - 69) / 12) if notes else 220.0
    midi[:, 0] = frequency
    for i, (onset, length, pitch, velocity) in enumerate(notes):
        start = int(round(onset * sr))
        if start >= frames:
            break
        end = int(round((onset + length) * sr))
        if i + 1 < len(notes):
            end = min(end, int(round(notes[i + 1][0] * sr)) - gap)
        end = min(max(end, start + 1), frames)
        midi[start:, 0] = 440.0 * 2 ** ((pitch - 69) / 12)
        midi[start:, 1] = velocity
        midi[start:end, 2] = 1.0
    return midi


def render_document(doc, frames, notes=None, audio=None):
    midi = midi_tensor(notes or [], frames)
    if audio is None:
        audio = np.zeros((frames, 2), np.float32)
    with NativeRuntime(doc, SAMPLE_RATE, prepare.LIB) as runtime:
        raw = runtime.render(audio=audio, midi=midi)
    out = prepare.audible(doc, raw)
    if not np.isfinite(out).all():
        raise prepare.InvalidOutput("nonfinite render")
    return out


def render_scene(scene):
    """Atom stems (pre-bus, gain applied) and the final mix; raises InvalidOutput."""
    frames = int(round(scene["seconds"] * SAMPLE_RATE))
    atoms = atoms_of(scene)
    docs = {}
    stems = []
    for atom in atoms:
        key = (atom["instrument"], atom["layer"])
        if key not in docs:
            docs[key] = layer_document(scene["instruments"][atom["instrument"]], atom["layer"])
        stem = render_document(docs[key], frames, atom["notes"]) * atom["gain"]
        if np.sqrt(np.mean(stem ** 2)) < 1e-6:
            raise prepare.InvalidOutput(f"silent atom {key}")
        stems.append(stem)
    dry = np.sum(stems, axis=0)
    peak = float(np.abs(dry).max())
    if peak < 1e-4:
        raise prepare.InvalidOutput("silent mix")
    scale = 0.5 / peak  # headroom before the bus (tanh/reverb add level)
    stems = [s * scale for s in stems]
    dry = dry * scale
    mix = render_document(bus_document(scene["bus_fx"]), frames, audio=dry.astype(np.float32)) \
        if scene["bus_fx"] else dry
    peak = float(np.abs(mix).max())
    if peak < 1e-4 or peak > 50:
        raise prepare.InvalidOutput("bus output silent or unstable")
    post = PEAK / peak
    return stems, mix * post, {"pre_bus_scale": scale, "post_bus_scale": post}


# ------------------------------------------------------------------------- features


def _bands(count, n_fft=N_FFT):
    freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)
    edges = np.geomspace(F_LOW, F_HIGH, count + 1)
    matrix = np.zeros((len(freqs), count), np.float32)
    for b in range(count):
        inside = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if not inside.any():
            inside = np.zeros_like(inside)
            inside[np.argmin(np.abs(freqs - np.sqrt(edges[b] * edges[b + 1])))] = True
        matrix[inside, b] = 1.0 / inside.sum()
    return matrix


_BAND_CACHE = {}


def band_power(signal, hop, bands):
    key = (hop, bands)
    if key not in _BAND_CACHE:
        _BAND_CACHE[key] = _bands(bands)
    padded = np.pad(signal, (N_FFT // 2, N_FFT // 2))
    count = 1 + (len(padded) - N_FFT) // hop
    frames = np.lib.stride_tricks.as_strided(
        padded, (count, N_FFT), (padded.strides[0] * hop, padded.strides[0]))
    power = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1)) ** 2 / N_FFT
    return power @ _BAND_CACHE[key]


def mix_features(mix):
    """[frames, 160] log band power of Mid (128) and Side (32), relative to the clip RMS."""
    rms = float(np.sqrt(np.mean(mix ** 2)))
    mid = (mix[:, 0] + mix[:, 1]) / (2 * rms)
    side = (mix[:, 0] - mix[:, 1]) / (2 * rms)
    return np.log10(np.concatenate([band_power(mid, HOP, MID_BANDS),
                                    band_power(side, HOP, SIDE_BANDS)], 1) + LOG_FLOOR).astype(np.float32), rms


def stem_features(stem, rms):
    """[frames, 64] coarse Mid log band power of one atom, on the mix's scale."""
    mid = (stem[:, 0] + stem[:, 1]) / (2 * rms)
    return np.log10(band_power(mid, STEM_HOP, STEM_BANDS) + LOG_FLOOR).astype(np.float32)


# --------------------------------------------------------------------------- corpus


def split_of(index):
    value = int(hashlib.sha256(f"mix-split-{index}".encode()).hexdigest()[:8], 16) / 2 ** 32
    return "train" if value < 0.75 else "validation" if value < 0.875 else "test"


def build_scene(seed, index, attempts=6, count_weights=None):
    """Sample and render scene ``index``; redraws (index, attempt) if a render is invalid."""
    for attempt in range(attempts):
        scene = sample_scene(seed + attempt * 7919, index, count_weights)
        try:
            stems, mix, scales = render_scene(scene)
        except (prepare.InvalidOutput, ValueError) as error:
            last = error
            continue
        scene["attempt"] = attempt
        scene["render"] = scales
        return scene, stems, mix
    raise RuntimeError(f"scene {index}: no valid render in {attempts} attempts ({last})")


def generate_shard(task):
    destination, seed, start, count, keep_wav, count_weights = task
    path = Path(destination) / "shards" / f"{start:07d}"
    if (path / "complete.json").exists():
        return json.loads((path / "complete.json").read_text())
    path.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    rows, mixes, lengths, stem_arrays, atom_levels, counts = [], [], [], [], [], []
    for index in range(start, start + count):
        scene, stems, mix = build_scene(seed, index, count_weights=count_weights)
        atoms = atoms_of(scene)
        levels = ladder(scene, atoms)
        features, rms = mix_features(mix)
        padded = np.full((MAX_FRAMES, MID_BANDS + SIDE_BANDS), np.log10(LOG_FLOOR), np.float32)
        padded[: len(features)] = features[:MAX_FRAMES]
        stem_block = np.full((MAX_INSTRUMENTS, MAX_STEM_FRAMES, STEM_BANDS), np.log10(LOG_FLOOR), np.float32)
        level_block = np.full((MAX_INSTRUMENTS, 3), -1, np.int8)
        for a, stem in enumerate(stems):
            s = stem_features(stem, rms)
            stem_block[a, : len(s)] = s[:MAX_STEM_FRAMES]
            level_block[a] = [levels["L0"][a], levels["L1"][a], levels["L2"][a]]
        mixes.append(padded.astype(np.float16))
        lengths.append(min(len(features), MAX_FRAMES))
        stem_arrays.append(stem_block.astype(np.float16))
        atom_levels.append(level_block)
        counts.append([max(levels[k]) + 1 for k in ("L0", "L1", "L2")])
        scene["split"] = split_of(index)
        scene["ladder"] = levels
        scene["mix_rms"] = rms
        if keep_wav:
            (path / "audio").mkdir(exist_ok=True)
            prepare.write_wav(path / "audio" / f"{index:07d}.wav", mix, SAMPLE_RATE)
        rows.append(scene)
    np.savez_compressed(
        path / "scenes.npz", index=np.arange(start, start + count), mix=np.stack(mixes),
        frames=np.asarray(lengths, np.int16), stems=np.stack(stem_arrays),
        atom_levels=np.stack(atom_levels), counts=np.asarray(counts, np.int8),
        split=np.asarray([["train", "validation", "test"].index(r["split"]) for r in rows], np.int8))
    with open(path / "scenes.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {"start": start, "count": count, "seconds": time.monotonic() - began,
               "audio_seconds": float(sum(r["seconds"] for r in rows)),
               "counts": np.asarray(counts).tolist()}
    prepare.dump(path / "complete.json", summary)
    return summary


def build(destination, scenes, per_shard, workers, seed, keep_wav, count_weights=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tasks = [(str(destination), seed, start, min(per_shard, scenes - start), keep_wav, count_weights)
             for start in range(0, scenes, per_shard)]
    began = time.monotonic()
    results = []
    with ProcessPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(generate_shard, t) for t in tasks]):
            results.append(future.result())
            print(f"shard {results[-1]['start']} done ({len(results)}/{len(tasks)})", flush=True)
    counts = np.concatenate([np.asarray(r["counts"]).reshape(-1, 3) for r in results])
    report = {
        "schema": SCHEMA, "seed": seed, "scenes": scenes, "shards": len(tasks),
        "count_weights": None if count_weights is None else list(map(float, count_weights)),
        "sample_rate": SAMPLE_RATE, "features": {"hop": HOP, "n_fft": N_FFT, "mid_bands": MID_BANDS,
                                                 "side_bands": SIDE_BANDS, "f_low": F_LOW, "f_high": F_HIGH,
                                                 "stem_hop": STEM_HOP, "stem_bands": STEM_BANDS},
        "wall_seconds": time.monotonic() - began,
        "audio_seconds": float(sum(r["audio_seconds"] for r in results)),
        "count_histogram": {level: np.bincount(counts[:, i], minlength=MAX_INSTRUMENTS + 1)[1:].tolist()
                            for i, level in enumerate(("L0", "L1", "L2"))},
        "ladder_differs": float(np.mean(counts[:, 0] != counts[:, 2])),
        "source_commit": prepare.source_commit(),
    }
    prepare.dump(destination / "report.json", report)
    return report


def listen(destination, count, out):
    """Write mix, L0 instrument stems and specs for the first ``count`` scenes of a corpus."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((Path(destination) / "report.json").read_text())
    for index in range(count):
        scene, stems, mix = build_scene(report["seed"], index, count_weights=report.get("count_weights"))
        folder = out / f"scene-{index:04d}"
        folder.mkdir(exist_ok=True)
        prepare.write_wav(folder / "mix.wav", mix, SAMPLE_RATE)
        levels = ladder(scene, atoms_of(scene))
        post = scene["render"]["post_bus_scale"]
        for k in range(len(scene["instruments"])):
            stem = sum(s for s, i in zip(stems, levels["L0"]) if i == k) * post
            prepare.write_wav(folder / f"L0-instrument-{k}-dry.wav", np.clip(stem, -0.999, 0.999), SAMPLE_RATE)
        prepare.dump(folder / "scene.json", scene)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="sample, render and featurise scenes")
    b.add_argument("destination")
    b.add_argument("--scenes", type=int, default=1000)
    b.add_argument("--per-shard", type=int, default=250)
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--keep-wav", action="store_true")
    b.add_argument("--count-weights", type=float, nargs="+",
                   help="relative weight of 1..8 natural instruments (curriculum: '1' = single instrument)")
    s = sub.add_parser("listen", help="write WAVs of the first scenes of a built corpus")
    s.add_argument("destination")
    s.add_argument("--count", type=int, default=4)
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(args.destination, args.scenes, args.per_shard, args.workers,
                               args.seed, args.keep_wav, args.count_weights), indent=2))
    else:
        listen(args.destination, args.count, args.out)


if __name__ == "__main__":
    main()
