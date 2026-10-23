#!/usr/bin/env python3
"""Auditable CPU-native, many-topology paired AMO training corpus."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from training.amo.native import NativeRuntime
from training.amo.dataset import (
    prepare_wav,
    file_digest,
)
from training.amo.analysis import extract, AnalysisProfile

SEED = 2026101402
LIB = Path(
    os.environ.get(
        "AMO_TENSOR_LIBRARY", ROOT / "build/data-release/src/tensor/libalfa_tensor_c.so"
    )
)
NAMES = [
    "additive_sparse4",
    "additive_dense12",
    "additive_inharmonic8",
    "saw_lowpass",
    "saw_bandpass",
    "saw_highpass",
    "saw_lp_tanh",
    "saw_tanh_lp",
    "wavetable_lowpass",
    "sine_ring",
    "sine_fm",
    "sine_am",
    "saw_chorus",
    "saw_phaser",
    "saw_comb",
    "noise_bandpass",
    "noise_modal",
    "impulse_modal4",
    "impulse_modal12",
    "impulse_waveguide",
    "additive_resonator",
    "fm_lowpass",
    "additive_cubic",
    "additive_feedback",
    "fx_lowpass",
    "fx_highpass",
    "fx_bandpass",
    "fx_tanh",
    "fx_phaser",
    "fx_stereo_width",
    "fx_ring_delay",
    "fx_fir",
]
LEGACY_PITCHED = [n for n in NAMES[:24] if n not in ("noise_bandpass", "noise_modal")]
PROFILE = {
    "frames": 24,
    "harmonic_ratios": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
    "minimum_fundamental_hz": 50.0,
    "maximum_fundamental_hz": 600.0,
    "recording_policy": "controlled_render",
}
MAX_NODES, MAX_PARAMS, MAX_ROUTING, MAX_MODULATION, MAX_ASSETS = 8, 80, 80, 24, 256
SOURCE_IDS = {
    "audio": 1,
    "midi": 2,
    "gesture": 3,
    "preference": 4,
    "latent": 5,
    "context": 6,
    "state": 7,
    "output_history": 8,
}
MAX_EDGES = 96


def source_commit():
    """Commit of the generating source; a git-less bundle (e.g. Kaggle) sets AMO_SOURCE_COMMIT."""
    if os.environ.get("AMO_SOURCE_COMMIT"):
        return os.environ["AMO_SOURCE_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dump(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    )


def digest_json(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def asset(values):
    return {
        "axes": ["sample"],
        "shape": [len(values)],
        "values": [float(x) for x in values],
    }


class Graph:
    def __init__(self):
        self.doc = {
            "schema": "amo.tensor/1",
            "ontology": "amo.math/1",
            "tensors": {},
            "instances": [],
            "projections": [],
            "parameter_locks": [],
            "shared_initial_state": [],
            "outputMix": [],
        }
        self.values = {
            "activation": [],
            "parameters": [],
            "routing": [],
            "modulation": [],
        }
        self.bounds = {k: [] for k in self.values}
        self.state = 0
        self.outputs = 0
        self.parameter_metadata = []
        self.perturbable = []

    def node(
        self, op, parameters, states, outputs=1, assets=None, initial=None, clock=1
    ):
        offset = len(self.values["parameters"])
        index = len(self.doc["instances"])
        node = {
            "primitive": op,
            "instance_id": f"node-{index}",
            "activation": index,
            "parameters": {"offset": offset, "count": len(parameters)},
            "clock": {"period_samples": clock, "phase_samples": 0},
            "output_between_updates": "hold",
            "state_size": states,
            "initial_state": initial if initial is not None else [0.0] * states,
            "assets": assets or [],
            "input_channels": 0,
            "audio_bindings": [],
        }
        self.doc["instances"].append(node)
        self.values["activation"].append(1.0)
        self.bounds["activation"].append([1.0, 1.0])
        for local, (value, low, high, name) in enumerate(parameters):
            self.values["parameters"].append(float(value))
            self.bounds["parameters"].append([float(low), float(high)])
            self.parameter_metadata.append(
                {
                    "index": offset + local,
                    "instance": index,
                    "primitive": op,
                    "local": local,
                    "name": name,
                }
            )
            if low < high:
                self.perturbable.append(("parameters", offset + local))
        self.state += states
        start = self.outputs
        self.outputs += outputs
        return index, offset, start

    def weight(self, value, low=None, high=None, name="routing", perturb=False):
        index = len(self.values[name])
        self.values[name].append(float(value))
        self.bounds[name].append(
            [
                float(value if low is None else low),
                float(value if high is None else high),
            ]
        )
        if perturb:
            self.perturbable.append((name, index))
        return index

    def project(
        self,
        source,
        element,
        parameter,
        value,
        low=None,
        high=None,
        mod=False,
        perturb=False,
    ):
        kind = "modulation" if mod else "routing"
        index = self.weight(value, low, high, kind, perturb)
        self.doc["projections"].append(
            {
                "source": source,
                "source_element": element,
                "parameter_element": parameter,
                "weights": kind,
                "weight_element": index,
                "missing_source": "zero",
            }
        )

    def bind(
        self,
        node,
        source,
        element,
        destination=0,
        value=1.0,
        low=None,
        high=None,
        perturb=False,
        mod=False,
    ):
        kind = "modulation" if mod else "routing"
        index = self.weight(value, low, high, kind, perturb)
        target = self.doc["instances"][node]
        target["input_channels"] = max(target["input_channels"], destination + 1)
        target["audio_bindings"].append(
            {
                "source": source,
                "source_element": element,
                "destination_element": destination,
                "weights": kind,
                "weight_element": index,
                "missing_source": "zero",
            }
        )

    def finish(self):
        axes = {
            "activation": "primitive",
            "parameters": "parameter",
            "routing": "coefficient",
            "modulation": "relationship",
        }
        for name, values in self.values.items():
            self.doc["tensors"][name] = {
                "axes": [axes[name]],
                "shape": [len(values)],
                "values": values,
                "presence": [True] * len(values),
            }
        self.doc["parameter_locks"] = [False] * len(self.values["parameters"])
        stored = (
            sum(len(v) for v in self.values.values())
            + sum(i["input_channels"] for i in self.doc["instances"])
            + sum(len(a["values"]) for i in self.doc["instances"] for a in i["assets"])
        )
        terms = len(self.doc["projections"]) + sum(
            len(i["audio_bindings"]) for i in self.doc["instances"]
        )
        self.doc["budget"] = {
            "active_primitives": len(self.doc["instances"]),
            "state_elements": self.state,
            "parameter_elements": len(self.values["parameters"]),
            "routing_elements": len(self.values["routing"]),
            "modulation_elements": len(self.values["modulation"]),
            "output_elements": self.outputs,
            "stored_tensor_elements": stored,
            "projection_terms": terms,
            "sparse_state_writes": 256,
            "scratch_elements": 4096,
        }
        return self.doc, self.bounds, self.perturbable, self.parameter_metadata


def make_document(name, u):
    # LHS coordinates are used consistently rather than randomized gain copies.
    cursor = [0]

    def pick(lo, hi, log=False):
        value = u[cursor[0] % len(u)]
        cursor[0] += 1
        return (
            float(np.exp(np.log(lo) + value * np.log(hi / lo)))
            if log
            else float(lo + (hi - lo) * value)
        )

    def p(lo, hi, label, log=False):
        return (pick(lo, hi, log), lo, hi, label)

    def fixed(v, label):
        return (v, v, v, label)

    g = Graph()
    is_fx = name.startswith("fx_")
    last = 0
    if not is_fx:
        env, ep, eo = g.node(
            "control.adsr",
            [
                fixed(0.0, "gate"),
                p(0.004, 0.30, "attack", True),
                p(0.04, 0.45, "decay", True),
                p(0.12, 0.90, "sustain"),
                p(0.055, 0.50, "release", True),
            ],
            4,
            clock=8 if name in ("saw_chorus", "saw_phaser", "saw_comb") else 1,
        )
        g.project("midi", 2, ep, 1.0)
        g.project("midi", 1, ep + 1, -0.002)
        g.project("midi", 1, ep + 2, 0.018)

    def sine(ratio=1.0, amplitude=1.0, sampled_ratio=False):
        node, offset, out = g.node(
            "math.sine",
            [fixed(0.0, "frequency base"), fixed(amplitude, "amplitude")],
            1,
        )
        g.project(
            "midi",
            0,
            offset,
            ratio,
            0.25 if sampled_ratio else ratio,
            6.0 if sampled_ratio else ratio,
        )
        return node, offset, out

    def saw():
        node, offset, out = g.node(
            "osc.polyblep_saw",
            [fixed(0.0, "frequency base"), fixed(0.6, "amplitude")],
            1,
        )
        g.project("midi", 0, offset, 1.0)
        return out

    def additive(count, inharmonic=False):
        parameters = [
            fixed(0.0, f"partial-{i}-{kind} base")
            for i in range(count)
            for kind in ("frequency", "amplitude")
        ]
        node, offset, out = g.node("osc.additive", parameters, count)
        slope = pick(0.35, 2.5)
        for i in range(count):
            ratio = 1.0 if i == 0 else pick(1.2, 11.5) if inharmonic else float(i + 1)
            amplitude = 0.3 if i == 0 else pick(0.025, 0.15) / (i + 1) ** slope
            g.project(
                "midi",
                0,
                offset + 2 * i,
                ratio,
                1.0,
                12.0,
                perturb=inharmonic and i > 0,
            )
            g.project(
                "state", 0, offset + 2 * i + 1, amplitude, 0.000001, 0.35, perturb=i > 0
            )
        return out

    def filt(source, kind="lowpass", fx=False):
        node, offset, out = g.node(
            "filter.biquad_" + kind,
            [p(180.0, 6500.0, "cutoff Hz", True), p(0.45, 2.0, "Q", True)],
            4,
        )
        g.bind(node, "audio" if fx else "output_history", source)
        if not fx:
            g.project(
                "state",
                0,
                offset,
                pick(100.0, 1000.0),
                100.0,
                1000.0,
                mod=True,
                perturb=True,
            )
        return out

    def effect(source, op, parameters, states=0, fx=False, assets=None):
        node, offset, out = g.node(op, parameters, states, assets=assets)
        g.bind(node, "audio" if fx else "output_history", source)
        return out

    def noise():
        node, offset, out = g.node(
            "excitation.white_noise", [], 2, initial=[12345.0, 54321.0]
        )
        return out

    def impulse():
        node, offset, out = g.node(
            "excitation.impulse", [fixed(0.0, "gate"), fixed(0.15, "excitation")], 1
        )
        g.project("midi", 2, offset, 1.0)
        return out

    def modal(source, count):
        params = []
        for i in range(count):
            params.extend(
                [
                    fixed(0.0, f"mode-{i} frequency base"),
                    p(0.10, 1.0, f"mode-{i} T60", True),
                    p(0.03, 0.22, f"mode-{i} gain"),
                ]
            )
        node, offset, out = g.node("physical.modal_rotation.v1", params, 2 * count)
        for i in range(count):
            g.project(
                "midi",
                0,
                offset + 3 * i,
                1.0 if i == 0 else pick(1.2, 8.0),
                1.0,
                8.0,
                perturb=i > 0,
            )
        g.bind(node, "output_history", source)
        return out

    if name.startswith("additive_"):
        count = (
            4
            if name == "additive_sparse4"
            else 12
            if name == "additive_dense12"
            else 8
            if name == "additive_inharmonic8"
            else 6
        )
        last = additive(count, name == "additive_inharmonic8")
        if name == "additive_resonator":
            node, offset, out = g.node(
                "physical.resonator",
                [
                    fixed(0.0, "frequency base"),
                    p(0.015, 0.12, "decay seconds", True),
                    p(0.05, 0.3, "gain"),
                ],
                2,
            )
            g.project("midi", 0, offset, pick(0.9, 3.0), 0.9, 3.0, perturb=True)
            g.bind(node, "output_history", last)
            last = out
        elif name == "additive_cubic":
            last = effect(last, "nonlinear.cubic_softclip", [p(1.0, 4.0, "drive")])
        elif name == "additive_feedback":
            last = effect(
                last,
                "nonlinear.feedback_tanh",
                [p(0.5, 2.5, "drive"), p(-0.5, 0.65, "feedback")],
                1,
            )
    elif name.startswith("saw_"):
        last = saw()
        if name in ("saw_lowpass", "saw_bandpass", "saw_highpass", "saw_lp_tanh"):
            last = filt(
                last,
                {"saw_bandpass": "bandpass", "saw_highpass": "highpass"}.get(
                    name, "lowpass"
                ),
            )
            if name == "saw_lp_tanh":
                last = effect(last, "nonlinear.tanh", [p(1.0, 5.0, "drive")])
        elif name == "saw_tanh_lp":
            last = filt(effect(last, "nonlinear.tanh", [p(1.0, 5.0, "drive")]))
        elif name == "saw_chorus":
            node, offset, out = g.node(
                "time.ring_delay_cubic.v1",
                [p(12.0, 100.0, "delay updates"), p(-0.3, 0.65, "feedback")],
                258,
            )
            g.bind(node, "output_history", last)
            lfo, lp, lo = g.node(
                "math.sine",
                [p(0.2, 3.0, "LFO Hz", True), fixed(1.0, "LFO amplitude")],
                1,
            )
            g.project(
                "output_history",
                lo,
                offset,
                pick(2.0, 8.0),
                2.0,
                8.0,
                mod=True,
                perturb=True,
            )
            # Fixed dry and learned wet operands into the final gain stage.
            last = out
        elif name == "saw_phaser":
            last = effect(
                last,
                "time.phaser_one_stage",
                [p(-0.8, 0.8, "allpass coefficient"), p(0.15, 0.9, "wet")],
                2,
            )
        elif name == "saw_comb":
            last = effect(
                last,
                "time.ring_allpass.v1",
                [p(6.0, 100.0, "delay updates"), p(-0.75, 0.75, "coefficient")],
                258,
            )
    elif name == "wavetable_lowpass":
        t = np.arange(256) / 256
        table = (
            np.sin(2 * np.pi * t)
            + 0.22 * np.sin(6 * np.pi * t)
            + 0.08 * np.sin(10 * np.pi * t)
        )
        node, offset, out = g.node(
            "osc.wavetable",
            [fixed(0.0, "frequency base"), p(0.25, 0.7, "amplitude")],
            1,
            assets=[asset(table / abs(table).max())],
        )
        g.project("midi", 0, offset, 1.0)
        last = filt(out)
    elif name in ("sine_ring", "sine_fm", "sine_am", "fm_lowpass"):
        mod, mp, mo = sine(pick(0.25, 6.0), sampled_ratio=True)
        carrier, cp, last = sine(1.0, 0.25 if name in ("sine_am", "sine_ring") else 0.6)
        if name in ("sine_fm", "fm_lowpass"):
            g.project(
                "output_history",
                mo,
                cp,
                pick(15.0, 400.0),
                15.0,
                400.0,
                mod=True,
                perturb=True,
            )
        else:
            if name == "sine_ring":
                g.values["parameters"][cp + 1] = 0.0
                g.bounds["parameters"][cp + 1] = [0.0, 0.0]
            g.project(
                "output_history",
                mo,
                cp + 1,
                pick(0.1, 0.4),
                0.1,
                0.4,
                mod=True,
                perturb=True,
            )
        if name == "fm_lowpass":
            last = filt(last)
    elif name in ("noise_bandpass", "noise_modal"):
        last = noise()
        last = filt(last, "bandpass") if name == "noise_bandpass" else modal(last, 4)
    elif name in ("impulse_modal4", "impulse_modal12"):
        last = modal(impulse(), 4 if name.endswith("4") else 12)
    elif name == "impulse_waveguide":
        excitation = impulse()
        node, offset, out = g.node(
            "physical.tuned_waveguide.v1",
            [
                fixed(0.0, "frequency base"),
                p(0.90, 0.996, "feedback"),
                p(0.05, 0.75, "damping"),
                p(-0.5, 0.5, "dispersion"),
            ],
            519,
        )
        g.project("midi", 0, offset, 1.0)
        g.bind(node, "output_history", excitation)
        last = out
    elif name in ("fx_lowpass", "fx_highpass", "fx_bandpass"):
        last = filt(0, name[3:], True)
    elif name == "fx_tanh":
        last = effect(
            0,
            "nonlinear.tanh.v1",
            [
                p(1.0, 10.0, "drive", True),
                p(-0.15, 0.15, "bias"),
                p(0.15, 0.7, "output gain"),
            ],
            fx=True,
        )
    elif name == "fx_phaser":
        last = effect(
            0,
            "time.phaser_one_stage",
            [p(-0.9, 0.9, "allpass coefficient"), p(0.05, 0.95, "wet")],
            2,
            fx=True,
        )
    elif name == "fx_stereo_width":
        node, offset, last = g.node(
            "spatial.stereo_width", [p(0.1, 1.9, "width")], 0, outputs=2
        )
        g.bind(node, "audio", 0, 0)
        g.bind(node, "audio", 1, 1)
    elif name == "fx_ring_delay":
        last = effect(
            0,
            "time.ring_delay_cubic.v1",
            [p(20.0, 900.0, "delay updates"), p(-0.65, 0.75, "feedback")],
            1026,
            fx=True,
        )
    elif name == "fx_fir":
        taps = np.zeros(16)
        taps[0] = 0.5
        for i in range(1, 16):
            taps[i] = pick(-0.12, 0.12) * np.exp(-i / 5)
        taps = taps / (1 + np.abs(taps).sum())
        last = effect(
            0, "time.causal_fir_ring.v1", [], 18, fx=True, assets=[asset(taps)]
        )
    else:
        raise ValueError(name)
    if not is_fx:
        # Authored output gain stage really executes in native DSP; no hidden envelope postprocess.
        source_output = last
        node, offset, last = g.node(
            "nonlinear.tanh.v1",
            [fixed(0.7, "drive"), fixed(0.0, "bias"), fixed(0.0, "gain base")],
            0,
        )
        g.bind(node, "output_history", source_output)
        if name == "saw_chorus":
            g.bind(
                node,
                "output_history",
                1,
                value=pick(0.2, 0.65),
                low=0.2,
                high=0.65,
                perturb=True,
            )
        g.project("state", 0, offset + 2, 0.25)
        pan, pp, last = g.node(
            "spatial.constant_power_pan", [p(-0.8, 0.8, "pan")], 0, outputs=2
        )
        g.bind(pan, "output_history", last - 1)
        g.doc["outputMix"] = [[last, 0, 1.0], [last + 1, 1, 1.0]]
    else:
        g.doc["outputMix"] = [
            [last, 0, 0.4],
            [last + (1 if name == "fx_stereo_width" else 0), 1, 0.4],
        ]
    result = g.finish()
    result[0]["dataset_structure"] = {
        "name": name,
        "role": "paired_fx" if is_fx else "instrument",
        "description": "producer-authored native mathematical graph; not a pretrained sound category",
    }
    return result


MAX_RESAMPLES = 8


class InvalidOutput(ValueError):
    """A sampled construction rendered silent, clipping or nonfinite audio."""


def lhs(count, dimensions, seed):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            (rng.permutation(count) + rng.random(count)) / count
            for _ in range(dimensions)
        ]
    )


def instance_for(doc, bounds, editable, seed):
    result = copy.deepcopy(doc)
    rng = np.random.default_rng(seed)
    updates = []
    for tensor, index in editable:
        low, high = bounds[tensor][index]
        old = result["tensors"][tensor]["values"][index]
        # Perturb in declared producer coordinates, preserving sign and hard bounds.
        scale = max(abs(old) * 0.055, (high - low) * 0.008)
        new = float(np.clip(old + rng.uniform(-scale, scale), low, high))
        result["tensors"][tensor]["values"][index] = new
        updates.append(
            {"tensor": tensor, "index": index, "canonical": old, "instance": new}
        )
    if not editable:
        for node in result["instances"]:
            for a in node["assets"]:
                old = np.asarray(a["values"])
                new = old * rng.uniform(0.94, 1.06, size=len(old))
                a["values"] = new.tolist()
                updates.append(
                    {"tensor": "immutable_asset", "relative_change_bound": 0.06}
                )
    return result, updates


def performance_for(type_index, original_index, case, attempt=0):
    rng = np.random.default_rng(
        SEED + type_index * 1000003 + original_index * 9176 + case * 131 + attempt * 7777777
    )
    sr = 16000 if case % 2 == 0 else 24000
    is_fx = type_index >= 24
    # One duration range for every family: clip length is an observed input
    # (X_audio[0] and conditions[1]), so family-specific ranges would let a
    # model read "physical/modal" off the duration instead of the sound. The
    # floor still covers the slowest ADSR release and modal T60 tails.
    duration = float(rng.uniform(1.2, 2.4))
    note = int([36, 47, 58, 69][case] + rng.integers(-2, 3))
    # Limit high partials while still crossing registers and tuning conditions.
    note = min(69, max(34, note))
    note_off = float(rng.uniform(0.32, 0.62))
    event_kind = ["held", "staccato", "two_notes", "vibrato"][case]
    return {
        "version": "amo.performance-plan/2",
        "sample_rate": sr,
        "frames": round(duration * sr),
        "midi_note": note,
        "tuning_cents": float(rng.uniform(-12, 12)),
        "velocity": float(rng.uniform(0.25, 0.95)),
        "note_off_seconds": note_off,
        "events": event_kind,
        "vibrato_rate": float(rng.uniform(3.0, 7.0)),
        "vibrato_cents": float(rng.uniform(4.0, 30.0)),
        "input_seed": int(rng.integers(1, 2**31)),
        "input_kind": ["two_tones", "chirp", "noise_bursts", "chord"][case]
        if is_fx
        else "none",
        "role": "paired_fx" if is_fx else "instrument",
    }


def inputs_for(performance):
    sr = performance["sample_rate"]
    n = performance["frames"]
    t = np.arange(n, dtype=np.float64) / sr
    f0 = 440 * 2 ** (
        (performance["midi_note"] - 69 + performance["tuning_cents"] / 100) / 12
    )
    frequency = np.full(n, f0)
    if performance["events"] == "vibrato":
        frequency *= 2 ** (
            performance["vibrato_cents"]
            * np.sin(2 * np.pi * performance["vibrato_rate"] * t)
            / 1200
        )
    gate = (t < performance["note_off_seconds"]).astype(np.float64)
    if performance["events"] == "staccato":
        gate = (t < 0.14).astype(np.float64)
    elif performance["events"] == "two_notes":
        gate[((t >= 0.20) & (t < 0.28)) | (t >= 0.72)] = 0
        frequency[t >= 0.28] *= 2 ** (3 / 12)
    midi = np.column_stack(
        [frequency, np.full(n, performance["velocity"]), gate]
    ).astype(np.float32)
    audio = np.zeros((n, 2), np.float32)
    if performance["role"] == "paired_fx":
        rng = np.random.default_rng(performance["input_seed"])
        kind = performance["input_kind"]
        if kind == "two_tones":
            left = 0.20 * np.sin(2 * np.pi * f0 * t) + 0.08 * np.sin(
                2 * np.pi * f0 * 2.31 * t
            )
            right = 0.17 * np.sin(2 * np.pi * f0 * t + 0.43) + 0.11 * np.sin(
                2 * np.pi * f0 * 3.17 * t + 0.8
            )
        elif kind == "chirp":
            phase = 2 * np.pi * (60 * t + (6000 - 60) / (2 * max(t[-1], 1e-3)) * t * t)
            left = 0.24 * np.sin(phase)
            right = 0.17 * np.sin(phase + 0.47) + 0.04 * np.sin(phase * 0.53)
        elif kind == "noise_bursts":
            noise = rng.standard_normal((n, 2))
            window = (0.6 + 0.4 * np.sin(2 * np.pi * 2.2 * t)) * (t < 0.75)
            audio = (0.075 * noise * window[:, None]).astype(np.float32)
            return audio, midi
        else:
            left = sum(
                0.075 * np.sin(2 * np.pi * f0 * ratio * t + phase)
                for ratio, phase in [(1, 0), (1.26, 0.3), (1.5, 0.7)]
            )
            right = sum(
                0.07 * np.sin(2 * np.pi * f0 * ratio * t + phase)
                for ratio, phase in [(1, 0.4), (1.26, 1.1), (1.5, 2.0)]
            )
        amplitude = np.minimum(t / 0.012, 1) * np.clip((0.85 - t) / 0.16, 0, 1)
        audio = np.column_stack([left * amplitude, right * amplitude]).astype(
            np.float32
        )
    return audio, midi


def audible(doc, raw):
    output = np.zeros((len(raw), 2), np.float64)
    for index, channel, gain in doc["outputMix"]:
        output[:, channel] += raw[:, index] * gain
    return output


def write_wav(path, audio, sr):
    """Quantise to stereo PCM24; write `path` unless it is None.

    Returns (decoded audio, sha256 of the WAV file bytes). The digest is computed
    from the in-memory file either way, so a corpus built with --no-wav carries
    the same audio fingerprints as one that keeps its WAV files.
    """
    if not np.isfinite(audio).all() or np.abs(audio).max() >= 1:
        raise ValueError("Invalid or clipping native audio")
    codes = np.rint(audio * 8388608).astype(np.int32)
    encoded = (
        np.column_stack(
            [
                (codes.ravel() & 255),
                ((codes.ravel() >> 8) & 255),
                ((codes.ravel() >> 16) & 255),
            ]
        )
        .astype(np.uint8)
        .tobytes()
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(encoded)
    data = buffer.getvalue()
    if path is not None:
        Path(path).write_bytes(data)
    # Match stored PCM24 exactly for feature extraction and reproducibility checks.
    return codes.astype(np.float64) / 8388608, hashlib.sha256(data).hexdigest()


def features(audio, sr):
    # Broad-spectrum features do not infer a fictional fundamental for noise or FX.
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-8:
        return np.zeros(32 * 35 + 2, np.float32)
    mono = audio[:, int(np.argmax(np.mean(audio**2, axis=0)))]
    grid = np.geomspace(30.0, 7400.0, 32)
    values = []
    for a, b in zip(np.array_split(mono, 32), np.array_split(audio, 32)):
        spectrum = np.abs(np.fft.rfft(a * np.hanning(len(a)), n=2048))
        # Normalize to complete-clip RMS, never a silent frame's peak.
        spectrum = spectrum / (max(len(a), 1) * rms)
        values.extend(
            np.log1p(
                np.interp(grid, np.fft.rfftfreq(2048, 1 / sr), spectrum) * 20
            ).tolist()
        )
        values.extend((np.sqrt(np.mean(b * b, axis=0)) / rms).tolist())
        values.append(
            float(
                np.mean(b[:, 0] * b[:, 1])
                / (np.sqrt(np.mean(b[:, 0] ** 2) * np.mean(b[:, 1] ** 2)) + 1e-16)
            )
        )
    return np.asarray(
        [len(audio) / sr, *values, float(np.log(max(rms, 1e-12)))], np.float32
    )


def graph_labels(doc, op_ids):
    result = {}
    widths = {
        "activation": MAX_NODES,
        "parameters": MAX_PARAMS,
        "routing": MAX_ROUTING,
        "modulation": MAX_MODULATION,
    }
    for name, width in widths.items():
        values = doc["tensors"][name]["values"]
        array = np.zeros(width, np.float32)
        mask = np.zeros(width, np.bool_)
        if len(values) > width:
            raise ValueError("Tensor exceeds declared shared padding extent")
        array[: len(values)] = values
        mask[: len(values)] = True
        result[name] = array
        result[name + "_mask"] = mask
    assets = [
        v for node in doc["instances"] for a in node["assets"] for v in a["values"]
    ]
    if len(assets) > MAX_ASSETS:
        raise ValueError("Assets exceed shared padding extent")
    result["assets"] = np.pad(
        np.asarray(assets, np.float32), (0, MAX_ASSETS - len(assets))
    )
    result["assets_mask"] = np.arange(MAX_ASSETS) < len(assets)
    nodes = np.zeros((MAX_NODES, 7), np.int32)
    ops = np.zeros(MAX_NODES, np.int32)
    for i, node in enumerate(doc["instances"]):
        ops[i] = op_ids[node["primitive"]]
        nodes[i] = [
            node["parameters"]["offset"],
            node["parameters"]["count"],
            node["state_size"],
            node["clock"]["period_samples"],
            node["clock"]["phase_samples"],
            node["input_channels"],
            len(node["assets"]),
        ]
    edges = []
    for p in doc["projections"]:
        edges.append(
            [
                SOURCE_IDS.get(p["source"], 0),
                p["source_element"],
                -1,
                p["parameter_element"],
                0 if p["weights"] == "routing" else 1,
                p["weight_element"],
                1 if p["source"] == "output_history" else 0,
            ]
        )
    for i, node in enumerate(doc["instances"]):
        for b in node["audio_bindings"]:
            edges.append(
                [
                    SOURCE_IDS.get(b["source"], 0),
                    b["source_element"],
                    i,
                    b["destination_element"],
                    0 if b["weights"] == "routing" else 1,
                    b["weight_element"],
                    1 if b["source"] == "output_history" else 0,
                ]
            )
    if len(edges) > MAX_EDGES:
        raise ValueError("Graph exceeds shared edge extent")
    result["node_ops"] = ops
    result["node_layout"] = nodes
    result["node_mask"] = ops != 0
    result["edges"] = np.pad(
        np.asarray(edges, np.int32).reshape(-1, 7),
        ((0, MAX_EDGES - len(edges)), (0, 0)),
    )
    result["edge_mask"] = np.arange(MAX_EDGES) < len(edges)
    return result


def probe(destination):
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    all_ops = set()
    metadata = {}
    for ti, name in enumerate(NAMES):
        timings = []
        minimum = 1.0
        maximum = 0.0
        for level in (0.03, 0.5, 0.97):
            doc, bounds, editable, fields = make_document(name, np.full(64, level))
            for node in doc["instances"]:
                all_ops.add(node["primitive"])
            metadata[name] = {
                "bounds": bounds,
                "parameters": fields,
                "role": "paired_fx" if ti >= 24 else "instrument",
            }
            perf = performance_for(ti, 5, 2)
            input_audio, midi = inputs_for(perf)
            start = time.monotonic()
            with NativeRuntime(doc, perf["sample_rate"], LIB) as r:
                raw = r.render(audio=input_audio, midi=midi)
                out = audible(doc, raw)
                r.reset()
                again = audible(doc, r.render(audio=input_audio, midi=midi))
            if not np.array_equal(out, again):
                raise ValueError("Nondeterministic render " + name)
            peak = float(np.abs(out).max())
            rms = float(np.sqrt(np.mean(out * out)))
            if not np.isfinite(out).all() or peak >= 0.95 or rms <= 1e-6:
                raise ValueError((name, level, "invalid audio", peak, rms))
            timings.append(time.monotonic() - start)
            minimum = min(minimum, rms)
            maximum = max(maximum, peak)
            # Validate masks and structural encoding even before the large run.
            graph_labels(doc, {op: i + 1 for i, op in enumerate(sorted(all_ops))})
        results.append(
            {
                "name": name,
                "boundary_probes": 3,
                "max_peak": maximum,
                "min_rms": minimum,
                "seconds_two_renders_mean": float(np.mean(timings)),
            }
        )
        print("probed", name, round(np.mean(timings), 4), "seconds", flush=True)
    op_ids = {op: i + 1 for i, op in enumerate(sorted(all_ops))}
    dump(destination / "native-probe.json", results)
    dump(destination / "operation-vocabulary.json", op_ids)
    dump(destination / "coordinate-semantics.json", metadata)
    return op_ids


def generate_shard(task):
    ti, start, count, total, root, op_ids, keep_wav = task
    name = NAMES[ti]
    root = Path(root)
    path = root / "shards" / f"{ti:02d}-{name}-{start:04d}"
    if (path / "complete.json").exists():
        return json.loads((path / "complete.json").read_text())
    if path.exists():
        raise ValueError(
            "Incomplete shard exists; inspect it rather than overwrite " + str(path)
        )
    path.mkdir(parents=True)
    if keep_wav:
        (path / "audio").mkdir()
    (path / "labels").mkdir()
    coordinates = lhs(total, 64, SEED + ti * 3109)
    split_order = np.random.default_rng(SEED + ti * 787).permutation(total)
    split_of = {
        int(v): "train"
        if i < int(total * 0.75)
        else "validation"
        if i < int(total * 0.875)
        else "test"
        for i, v in enumerate(split_order)
    }
    rows = []
    arrays = []
    legacy = []
    group_digests = {}
    minimum = 1.0
    maximum = 0.0
    samples = 0
    total_audio_seconds = 0.0
    replay_count = 0
    features_seen = set()
    waveforms_seen = set()
    f0_error = []
    start_time = time.monotonic()
    profile = AnalysisProfile(
        **{**PROFILE, "harmonic_ratios": tuple(PROFILE["harmonic_ratios"])}
    )
    resampled = []
    for original in range(start, start + count):
        # A rare construction/performance pair renders silent or clipping (e.g. a pitch-modulated
        # tuned waveguide whose integer delay steps past its single-sample impulse). Redraw the
        # performances and the point inside the same Latin-hypercube cell (stratification is
        # kept) instead of aborting the whole build; complete.json lists every redraw.
        for attempt in range(MAX_RESAMPLES + 1):
            u = coordinates[original]
            if attempt:
                jitter = np.random.default_rng(SEED + ti * 7919 + original * 131 + attempt)
                u = (np.floor(u * total) + jitter.random(u.shape)) / total
            snapshot = (len(rows), len(arrays), len(legacy), len(f0_error), set(features_seen),
                        set(waveforms_seen), minimum, maximum, samples, total_audio_seconds, replay_count)
            structural_digest = None
            try:
                doc, bounds, editable, fields = make_document(name, u)
                alternative, changes = instance_for(
                    doc, bounds, editable, SEED + ti * 9511 + original * 43
                )
                group_id = f"{ti:02d}-{original:05d}"
                canonical_path = path / "labels" / f"{group_id}-canonical.json"
                instance_path = path / "labels" / f"{group_id}-instance.json"
                dump(canonical_path, doc)
                dump(instance_path, alternative)
                canonical_sha = file_digest(canonical_path)
                instance_sha = file_digest(instance_path)
                structural_digest = digest_json(doc)
                if structural_digest in group_digests:
                    raise ValueError("Duplicate original construction")
                group_digests[structural_digest] = split_of[original]
                targets = [graph_labels(d, op_ids) for d in (doc, alternative)]
                for document in (doc, alternative):
                    for name_, tensor in document["tensors"].items():
                        assert all(
                            lo <= v <= hi
                            for v, (lo, hi) in zip(tensor["values"], bounds[name_])
                        ), (name, original, name_)
                for case in range(4):
                    performance = performance_for(ti, original, case, attempt)
                    sr = performance["sample_rate"]
                    source_audio, midi = inputs_for(performance)
                    input_file = None
                    input_sha = None
                    if ti >= 24:
                        input_file = path / "audio" / f"{group_id}-input-{case}.wav"
                        # DSP consumes precisely the stored PCM input, not an unrecorded higher-precision input.
                        source_audio, input_sha = write_wav(
                            input_file if keep_wav else None, source_audio.astype(np.float64), sr
                        )
                        source_audio = source_audio.astype(np.float32)
                        if not keep_wav:
                            input_file = None
                    source_feature = (
                        features(source_audio, sr)
                        if ti >= 24
                        else np.zeros(32 * 35 + 2, np.float32)
                    )
                    for variation, d in enumerate((doc, alternative)):
                        example_id = f"{group_id}-p{case}-d{variation}"
                        with NativeRuntime(d, sr, LIB) as runtime:
                            output = audible(d, runtime.render(audio=source_audio, midi=midi))
                            if original == start and case == 0:
                                runtime.reset()
                                replay = audible(
                                    d, runtime.render(audio=source_audio, midi=midi)
                                )
                                assert np.array_equal(output, replay), (
                                    "Native reset reproducibility failed"
                                )
                                replay_count += 1
                        peak = float(np.abs(output).max())
                        rms = float(np.sqrt(np.mean(output**2)))
                        if not np.isfinite(output).all() or peak >= 0.95 or rms < 1e-6:
                            raise InvalidOutput(
                                (name, original, case, "invalid output", peak, rms)
                            )
                        wav = path / "audio" / f"{example_id}.wav"
                        decoded, waveform_sha = write_wav(wav if keep_wav else None, output, sr)
                        output_feature = features(decoded, sr)
                        conditions = np.asarray(
                            [
                                sr / 24000.0,
                                performance["frames"] / sr,
                                performance["velocity"],
                                performance["note_off_seconds"],
                                performance["midi_note"] / 127.0,
                                performance["tuning_cents"] / 100.0,
                                performance["vibrato_rate"] / 10.0,
                                performance["vibrato_cents"] / 100.0,
                                case / 3.0,
                                ti >= 24,
                            ],
                            np.float32,
                        )
                        x = np.concatenate([output_feature, source_feature])
                        assert np.isfinite(x).all()
                        fhash = hashlib.sha256(
                            np.rint(x * 1000).astype(np.int32).tobytes()
                        ).hexdigest()
                        waveforms_seen.add(waveform_sha)
                        features_seen.add(fhash)
                        minimum = min(minimum, rms)
                        maximum = max(maximum, peak)
                        samples += 1
                        total_audio_seconds += len(output) / sr
                        row = {
                            "id": example_id,
                            "canonical_id": group_id,
                            "split": split_of[original],
                            "type_id": ti,
                            "structure_name": name,
                            "role": performance["role"],
                            "canonical_json": str(canonical_path.relative_to(root)),
                            "instance_json": str(
                                (
                                    canonical_path if variation == 0 else instance_path
                                ).relative_to(root)
                            ),
                            "normalized_wav": str(wav.relative_to(root)) if keep_wav else None,
                            "input_wav": str(input_file.relative_to(root))
                            if input_file
                            else None,
                            "performance": performance,
                            "perturbation_id": "identity"
                            if variation == 0
                            else "bounded-multicoordinate",
                            "perturbations": [] if variation == 0 else changes,
                            "normalization": {
                                "policy_id": "controlled-broad-spectrum-paired-v2",
                                "removed": ["global_gain_from_relative_spectrum"],
                                "preserved": [
                                    "duration",
                                    "envelope",
                                    "spectral_bandwidth",
                                    "stereo_relation",
                                    "input_output_gain_ratio",
                                ],
                                "recording_conditions": "exact controlled render; no inferred calibration",
                            },
                            "quality": {
                                "peak": peak,
                                "rms": rms,
                                "last_50ms_rms": float(
                                    np.sqrt(np.mean(output[-round(sr * 0.05) :] ** 2))
                                ),
                            },
                            "sha256": {
                                "wav": waveform_sha,
                                "input": input_sha,
                                "canonical": canonical_sha,
                                "instance": canonical_sha if variation == 0 else instance_sha,
                                "construction": structural_digest,
                            },
                            "feature_fingerprint_1e3": fhash,
                        }
                        # The compact legacy export is deliberately restricted to matching known structures.
                        # Broad paired features remain the primary corpus for noisy/FX/nonfundamental examples.
                        legacy_feature = None
                        if keep_wav and name in LEGACY_PITCHED:
                            legacy_feature = np.asarray(
                                extract(decoded, sr, profile), np.float32
                            )
                            signal = decoded[:, int(np.argmax(np.mean(decoded**2, axis=0)))]
                            spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
                            bins = np.fft.rfftfreq(len(signal), 1 / sr)
                            selected = np.flatnonzero((bins >= 50) & (bins <= 600))
                            estimate = bins[selected[np.argmax(spectrum[selected])]]
                            expected = 440 * 2 ** (
                                (
                                    performance["midi_note"]
                                    - 69
                                    + performance["tuning_cents"] / 100
                                )
                                / 12
                            )
                            error = float(abs(1200 * np.log2(estimate / expected)))
                            row["reference_pitch_error_cents"] = error
                            f0_error.append(error)
                            if (
                                error < 35
                                and performance["events"] != "two_notes"
                                and split_of[original] != "test"
                            ):
                                legacy_row = {
                                    k: row[k]
                                    for k in (
                                        "id",
                                        "canonical_id",
                                        "split",
                                        "canonical_json",
                                        "instance_json",
                                        "normalized_wav",
                                        "perturbation_id",
                                    )
                                }
                                legacy_row["normalization"] = {
                                    "policy_id": "controlled-voiced-reference-policy-v2",
                                    "removed": ["pitch", "loudness", "recording_condition"],
                                    "preserved": [
                                        "timbre",
                                        "articulation",
                                        "temporal_behavior",
                                    ],
                                }
                                legacy.append(legacy_row)
                        packed = {
                            "X_audio": x,
                            "conditions": conditions,
                            "id": example_id,
                            "type_id": ti,
                            "split": {"train": 0, "validation": 1, "test": 2}[
                                split_of[original]
                            ],
                            "canonical_id": group_id,
                            "fx_input_mask": ti >= 24,
                        }
                        for prefix, target in [
                            ("canonical_", targets[0]),
                            ("instance_", targets[variation]),
                        ]:
                            for key, value in target.items():
                                packed[prefix + key] = value
                        if legacy_feature is not None:
                            packed["reference_features"] = legacy_feature
                        rows.append(row)
                        arrays.append(packed)
                break
            except InvalidOutput as error:
                if attempt == MAX_RESAMPLES:
                    raise
                del rows[snapshot[0]:], arrays[snapshot[1]:], legacy[snapshot[2]:], f0_error[snapshot[3]:]
                features_seen, waveforms_seen = snapshot[4], snapshot[5]
                minimum, maximum, samples, total_audio_seconds, replay_count = snapshot[6:]
                if structural_digest is not None and structural_digest in group_digests:
                    del group_digests[structural_digest]
                resampled.append({"original": original, "attempt": attempt + 1, "reason": list(error.args[0])})
                print(f"resampling {name} {original}: {error.args[0]}", flush=True)
    # Save bounded shards, never repeatedly decompress a complete dataset inside an item loop.
    keys = set(arrays[0])
    common = set.intersection(*(set(a) for a in arrays))
    np.savez_compressed(
        path / "prepared.npz",
        **{key: np.asarray([a[key] for a in arrays]) for key in sorted(common)},
    )
    with (path / "index.jsonl").open("w") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    dump(path / "canonical-splits.json", group_digests)
    dump(path / "legacy-index.json", legacy)
    summary = {
        "shard": str(path.relative_to(root)),
        "type_id": ti,
        "structure": name,
        "canonical_groups": count,
        "examples": samples,
        "min_rms": minimum,
        "max_peak": maximum,
        "audio_seconds": total_audio_seconds,
        "reset_replay_checks": replay_count,
        "resampled": resampled,
        "unique_waveforms": len(waveforms_seen),
        "unique_features_rounded_1e3": len(features_seen),
        "split_counts": {
            s: sum(r["split"] == s for r in rows)
            for s in ("train", "validation", "test")
        },
        "seconds": time.monotonic() - start_time,
        "index_sha256": file_digest(path / "index.jsonl"),
        "prepared_sha256": file_digest(path / "prepared.npz"),
    }
    dump(path / "complete.json", summary)
    return summary


def build(destination, canonical_per_type, workers, keep_wav=True):
    if destination.exists():
        raise ValueError("Published output exists; choose a new version")
    stage = destination.with_name(destination.name + ".staging")
    if stage.exists():
        raise ValueError("Staging exists; do not overwrite an incomplete run")
    stage.mkdir(parents=True)
    (stage / "shards").mkdir()
    op_ids = probe(stage)
    plan = {
        "seed": SEED,
        "types": NAMES,
        "canonical_per_type": canonical_per_type,
        "points_per_canonical": 8,
        "workers": workers,
        "expected_examples": 32 * canonical_per_type * 8,
        "git_commit": source_commit(),
        "script_sha256": file_digest(Path(__file__)),
        "native_library_sha256": file_digest(LIB),
        "split_policy": "original-construction group; stratified 75/12.5/12.5 per topology",
        "no_optimizer_started": True,
        # False: WAV files were hashed in memory but not stored (prepared features,
        # labels and fingerprints only). Legacy WAV-learner manifests need the files.
        "wav_files": keep_wav,
    }
    dump(stage / "build-plan.json", plan)
    jobs = [
        (
            ti,
            start,
            min(64, canonical_per_type - start),
            canonical_per_type,
            str(stage),
            op_ids,
            keep_wav,
        )
        for ti in range(32)
        for start in range(0, canonical_per_type, 64)
    ]
    done = []
    began = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_shard, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                print("FAILED shard", futures[future][:3], flush=True)
                raise
            done.append(result)
            print(
                f"shards {len(done)}/{len(jobs)} examples {sum(s['examples'] for s in done)} elapsed {time.monotonic() - began:.1f}s {result['structure']}",
                flush=True,
            )
            if os.statvfs(stage).f_bavail * os.statvfs(stage).f_frsize < 1024**3:
                raise RuntimeError("Less than 1 GiB free; unpublished corpus retained")
    # SQLite provides exact global uniqueness/leakage checks without keeping audio in RAM.
    db = sqlite3.connect(stage / "catalog.sqlite")
    db.executescript(
        "CREATE TABLE examples(id TEXT PRIMARY KEY, canonical_id TEXT, split TEXT, type_id INTEGER, wav_sha TEXT, feature_sha TEXT, shard TEXT, row_index INTEGER); CREATE TABLE originals(sha TEXT PRIMARY KEY, canonical_id TEXT, split TEXT);"
    )
    legacy_manifests = []
    for ti, name in enumerate(NAMES):
        representative, bounds, _, fields = make_document(name, np.full(64, 0.5))
        schemas = {
            k: {"axes": v["axes"], "padded_shape": v["shape"]}
            for k, v in representative["tensors"].items()
        }
        dump(stage / f"{ti:02d}-template.json", representative)
        all_legacy = []
        for summary in sorted(
            (s for s in done if s["type_id"] == ti), key=lambda s: s["shard"]
        ):
            shard = stage / summary["shard"]
            identities = json.loads((shard / "canonical-splits.json").read_text())
            # Canonical IDs derive only from originals, not waveform or parameter variants.
            shard_rows = [
                json.loads(line)
                for line in (shard / "index.jsonl").read_text().splitlines()
            ]
            identity_ids = {
                row["sha256"]["construction"]: row["canonical_id"] for row in shard_rows
            }
            for sha, split in identities.items():
                db.execute(
                    "INSERT INTO originals VALUES (?,?,?)",
                    (sha, identity_ids[sha], split),
                )
            for ri, line in enumerate((shard / "index.jsonl").read_text().splitlines()):
                row = json.loads(line)
                db.execute(
                    "INSERT INTO examples VALUES (?,?,?,?,?,?,?,?)",
                    (
                        row["id"],
                        row["canonical_id"],
                        row["split"],
                        ti,
                        row["sha256"]["wav"],
                        row["feature_fingerprint_1e3"],
                        summary["shard"],
                        ri,
                    ),
                )
            all_legacy.extend(json.loads((shard / "legacy-index.json").read_text()))
        if all_legacy and any(r["split"] == "validation" for r in all_legacy):
            legacy = {
                "schema": "amo.training-manifest/1",
                "category": "wav_preset_learning",
                "ontology": "amo.math/1",
                "schemas": schemas,
                "analysis": PROFILE,
                "learn": {
                    "hidden_width": 128,
                    "steps": 3000,
                    "learning_rate": 0.001,
                    "primary_weight": 2.0,
                    "secondary_weight": 1.0,
                    "mask_weight": 0.1,
                    "available_ram_bytes": 8 * 1024**3,
                    "seed": SEED,
                    "batch_size": 128,
                    "device": "cpu",
                },
                "require_validation": True,
                "examples": all_legacy,
                "reconstruction_bounds": bounds,
                "reconstruction_mask_threshold": 0.5,
                "limitations": "Known structural template; voiced-filtered subset only, not topology or FX recognition.",
            }
            manifest = stage / f"{ti:02d}-legacy-manifest.json"
            dump(manifest, legacy)
            # Real repository validation checks group splits, axes/padding and memory.
            _, _, _, audit, _ = prepare_wav(manifest, extract_features=False)
            dump(stage / f"{ti:02d}-legacy-preflight.json", audit)
            legacy_manifests.append(
                {"path": manifest.name, "examples": len(all_legacy)}
            )
    db.commit()
    leakage = db.execute(
        "SELECT wav_sha FROM examples GROUP BY wav_sha HAVING COUNT(DISTINCT split)>1"
    ).fetchone()
    if leakage:
        raise ValueError("Exact audio crossed split boundary")
    counts = dict(db.execute("SELECT split,COUNT(*) FROM examples GROUP BY split"))
    unique_audio = db.execute(
        "SELECT COUNT(DISTINCT wav_sha) FROM examples"
    ).fetchone()[0]
    unique_features = db.execute(
        "SELECT COUNT(DISTINCT feature_sha) FROM examples"
    ).fetchone()[0]
    groups = db.execute("SELECT COUNT(*) FROM originals").fetchone()[0]
    db.close()
    report = {
        **plan,
        "actual_examples": sum(s["examples"] for s in done),
        "actual_canonical_groups": groups,
        "counts": counts,
        "structures": 32,
        "native_operations": len(op_ids),
        "native_operations_used": sorted(op_ids),
        "unique_waveforms": unique_audio,
        "unique_features_rounded_1e3": unique_features,
        "max_peak": max(s["max_peak"] for s in done),
        "min_rms": min(s["min_rms"] for s in done),
        "audio_hours": sum(s["audio_seconds"] for s in done) / 3600,
        "shards": len(done),
        "native_reset_checks": sum(s["reset_replay_checks"] for s in done),
        "cross_split_exact_waveform_leakage": False,
        "reference_learner_manifests": legacy_manifests,
        "generation_seconds": time.monotonic() - began,
        "limitations": [
            "Synthetic controlled corpus, not measured historical audio or user preferences.",
            "32 authored graph families are not coverage of every possible graph.",
            "Full paired-FX/graph corpus uses the supplied structured shard reader; existing WAV-only learner receives voiced known-template subsets.",
            "Feature content hashes measure distinguishability, not statistical independence or model quality.",
            "No pretrained model, optimization run, or held-out model quality claim.",
        ],
    }
    assert report["actual_examples"] == plan["expected_examples"]
    assert groups == canonical_per_type * 32
    dump(stage / "shards.json", sorted(done, key=lambda s: s["shard"]))
    dump(stage / "report.json", report)
    os.rename(stage, destination)
    print(json.dumps(report, indent=2), flush=True)


def validate(destination, verification_output=None):
    report = json.loads((destination / "report.json").read_text())
    summaries = json.loads((destination / "shards.json").read_text())
    vocabulary = json.loads((destination / "operation-vocabulary.json").read_text())
    checked_hashes = 0
    replayed = 0
    max_error = 0.0
    fx_pairs = 0
    cache_items = 0
    groups = {}
    audio_splits = {}
    legacy_checks = []
    for si, summary in enumerate(summaries):
        shard = destination / summary["shard"]
        assert file_digest(shard / "index.jsonl") == summary["index_sha256"]
        assert file_digest(shard / "prepared.npz") == summary["prepared_sha256"]
        rows = [
            json.loads(line)
            for line in (shard / "index.jsonl").read_text().splitlines()
        ]
        # Load each bounded shard once, not once per item.
        with np.load(shard / "prepared.npz", allow_pickle=False) as archive:
            cache = {k: archive[k] for k in archive.files}
        assert len(cache["id"]) == len(rows) == summary["examples"]
        assert (
            np.isfinite(cache["X_audio"]).all()
            and np.isfinite(cache["conditions"]).all()
        )
        for prefix in ("canonical_", "instance_"):
            for name in ("activation", "parameters", "routing", "modulation", "assets"):
                assert np.isfinite(cache[prefix + name]).all()
                assert np.all(
                    cache[prefix + name][~cache[prefix + name + "_mask"]] == 0
                ), "Padding must not become present zero"
        known_labels = {}
        selected = {}
        for ri, row in enumerate(rows):
            assert str(cache["id"][ri]) == row["id"]
            group = row["canonical_id"]
            split = row["split"]
            assert groups.get(group, split) == split
            groups[group] = split
            digest = row["sha256"]["wav"]
            assert audio_splits.get(digest, split) == split
            audio_splits[digest] = split
            if row["normalized_wav"]:
                assert file_digest(destination / row["normalized_wav"]) == digest
                checked_hashes += 1
            if row["role"] == "paired_fx":
                fx_pairs += 1
            if row["input_wav"]:
                assert (
                    file_digest(destination / row["input_wav"])
                    == row["sha256"]["input"]
                )
                checked_hashes += 1
            for prefix, key in [
                ("canonical_", "canonical_json"),
                ("instance_", "instance_json"),
            ]:
                file = destination / row[key]
                if row[key] not in known_labels:
                    assert (
                        file_digest(file)
                        == row["sha256"][
                            "canonical" if prefix == "canonical_" else "instance"
                        ]
                    )
                    document = json.loads(file.read_text())
                    known_labels[row[key]] = graph_labels(document, vocabulary)
                    checked_hashes += 1
                expected = known_labels[row[key]]
                for name in expected:
                    assert np.array_equal(cache[prefix + name][ri], expected[name]), (
                        "Incorrect target packing",
                        row["id"],
                        name,
                    )
            selected.setdefault((split, row["perturbation_id"]), ri)
            cache_items += 1
        for ri in selected.values():
            row = rows[ri]
            perf = row["performance"]
            _, midi = inputs_for(perf)
            source = np.zeros((perf["frames"], 2), np.float32)
            if row["role"] == "paired_fx" and not row["input_wav"]:
                # --no-wav corpus: regenerate the deterministic input and require
                # the exact PCM24 file bytes that were fingerprinted.
                source, input_sha = write_wav(None, inputs_for(perf)[0].astype(np.float64), perf["sample_rate"])
                assert input_sha == row["sha256"]["input"], ("Input replay differs", row["id"])
                source = source.astype(np.float32)
                checked_hashes += 1
            if row["input_wav"]:
                from training.amo.analysis import read_wav

                source, rate = read_wav(destination / row["input_wav"])
                source = source.astype(np.float32)
                assert rate == perf["sample_rate"]
            doc = json.loads((destination / row["instance_json"]).read_text())
            with NativeRuntime(doc, perf["sample_rate"], LIB) as runtime:
                output = audible(doc, runtime.render(audio=source, midi=midi))
            if row["normalized_wav"]:
                from training.amo.analysis import read_wav

                recorded, rate = read_wav(destination / row["normalized_wav"])
            else:
                recorded, replay_sha = write_wav(None, output, perf["sample_rate"])
                rate = perf["sample_rate"]
                assert replay_sha == row["sha256"]["wav"], ("Output replay differs", row["id"])
                checked_hashes += 1
            assert rate == perf["sample_rate"]
            error = float(np.abs(recorded - output).max())
            max_error = max(max_error, error)
            assert error <= 6.1e-8, (
                "Replay exceeds PCM24 quantization tolerance",
                row["id"],
                error,
            )
            recomputed = np.concatenate(
                [
                    features(recorded, rate),
                    features(source, rate)
                    if row["role"] == "paired_fx"
                    else np.zeros(32 * 35 + 2, np.float32),
                ]
            )
            assert np.allclose(
                cache["X_audio"][ri], recomputed, atol=2e-6, rtol=1e-5
            ), "Cached features do not match decoded audio"
            replayed += 1
        if si % 16 == 0:
            print(
                f"verified shards {si + 1}/{len(summaries)}, points {cache_items}",
                flush=True,
            )
    for item in report["reference_learner_manifests"]:
        # Actual feature extraction + label validation for a representative train/validation subset.
        manifest_path = destination / item["path"]
        manifest = json.loads(manifest_path.read_text())
        selected = []
        for split in ("train", "validation"):
            candidates = [r for r in manifest["examples"] if r["split"] == split]
            selected.extend(candidates[:8])
        manifest["examples"] = selected
        local = destination / (".verify-" + item["path"])
        dump(local, manifest)
        _, _, splits, audit, _ = prepare_wav(local)
        local.unlink()
        assert len(splits["train"][0]) > 0 and len(splits["validation"][0]) > 0
        legacy_checks.append(
            {"manifest": item["path"], "feature_extraction_examples": audit["examples"]}
        )
    assert (
        cache_items == report["actual_examples"]
        and len(groups) == report["actual_canonical_groups"]
    )
    result = {
        "points_verified": cache_items,
        "original_groups_verified": len(groups),
        "content_hashes_verified": checked_hashes,
        "fx_input_output_points_verified": fx_pairs,
        "native_replay_examples": replayed,
        "max_native_replay_sample_error": max_error,
        "padding_and_presence_verified": True,
        "operation_and_edge_labels_verified": True,
        "cached_features_recomputed": True,
        "cross_split_group_or_exact_waveform_leakage": False,
        "reference_learner_checks": legacy_checks,
        "no_optimizer_started": True,
    }
    dump(verification_output or destination / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


class ShardReader:
    """Bounded CPU reader; topology IDs are metadata/targets, not audio input features."""

    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.split = {"train": 0, "validation": 1, "test": 2}[split]
        self.shards = json.loads((self.root / "shards.json").read_text())

    def batches(self, batch_size=64):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for shard in self.shards:
            with np.load(
                self.root / shard["shard"] / "prepared.npz", allow_pickle=False
            ) as archive:
                values = {key: archive[key] for key in archive.files}
            indices = np.flatnonzero(values["split"] == self.split)
            for start in range(0, len(indices), batch_size):
                rows = indices[start : start + batch_size]
                yield {key: value[rows] for key, value in values.items()}


if __name__ == "__main__":
    # Generation is CPU-only and single-threaded per worker. Set this only for the
    # CLI: importing ShardReader from a GPU trainer must not hide its devices.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "validate", "probe"])
    parser.add_argument("destination", type=Path)
    parser.add_argument("--canonical-per-type", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--no-wav",
        action="store_true",
        help="Do not store WAV files (~0.5 GB instead of ~30 GB); features, labels and "
        "audio fingerprints are kept, legacy WAV-learner manifests are skipped",
    )
    parser.add_argument(
        "--verification-output",
        type=Path,
        help="Write verification separately without changing an existing corpus",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= (os.cpu_count() or 1):
        parser.error(f"Use 1–{os.cpu_count()} CPU workers")
    if args.canonical_per_type < 8 or args.canonical_per_type % 8:
        parser.error("Canonical count must be a positive multiple of 8, at least 8")
    if args.action == "probe":
        probe(args.destination.resolve())
    elif args.action == "build":
        build(
            args.destination.resolve(),
            args.canonical_per_type,
            args.workers,
            keep_wav=not args.no_wav,
        )
    else:
        validate(args.destination.resolve(), args.verification_output)
