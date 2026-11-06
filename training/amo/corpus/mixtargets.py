#!/usr/bin/env python3
"""Fixed-size slot targets for mix scenes, and the inverse: model outputs -> scene spec.

Pure numpy (no torch) so the round trip is testable anywhere. One *unit* is an
instrument at one ladder level (MIX-DESIGN.md §3.3). Its slot target holds:

- the preset: family + u coordinates of up to two layers (``prepare.make_document``
  rebuilds the construction exactly from them), the second layer's transpose,
  the insert FX set and their parameters, the note echo;
- the performance: gain and a piano roll (onset / sustain / velocity) of its notes;
- an auxiliary stem: coarse log band power of the unit's atoms (``scenes.npz``).

``decode_scene`` turns slot predictions back into a ``mixscene`` scene spec, so a
prediction is rendered and scored with the same code that made the data.
"""

from functools import lru_cache

import numpy as np

from training.amo.corpus import mixscene, prepare

FAMILIES = list(prepare.NAMES[:24])
NO_LAYER = len(FAMILIES)  # family2 class meaning "no second layer"
LAYER_TRANSPOSE = [0, 12, -12]
ECHO_TRANSPOSE = sorted(set(mixscene.ECHO_TRANSPOSE))
INSERT_FX = list(mixscene.INSERT_FX)
BUS_FX = list(mixscene.BUS_FX)
U_DIMS = 64
PITCH_LOW, PITCH_HIGH = 16, 100
PITCHES = PITCH_HIGH - PITCH_LOW + 1
LEVELS = ("L0", "L1", "L2")

# name -> (low, high, log). Values are normalised to [0, 1] on these ranges.
INSERT_PARAMS = {
    "lowpass": [("cutoff", 500.0, 9000.0, True), ("q", 0.5, 3.0, True)],
    "highpass": [("cutoff", 60.0, 1500.0, True), ("q", 0.5, 3.0, True)],
    "bandpass": [("cutoff", 250.0, 5000.0, True), ("q", 0.5, 3.0, True)],
    "tanh": [("drive", 1.0, 8.0, True), ("bias", -0.1, 0.1, False), ("gain", 0.3, 0.8, False)],
    "phaser": [("coefficient", -0.9, 0.9, False), ("wet", 0.2, 0.9, False)],
    "fir": [(f"tap{i}", -0.2, 0.6, False) for i in range(12)],
    "echo": [("delay", 0.06, 0.45, False), ("feedback", 0.0, 0.6, False), ("wet", 0.2, 0.6, False)],
}
BUS_PARAMS = {
    "lowpass": [("cutoff", 500.0, 9000.0, True), ("q", 0.5, 1.6, True)],
    "highpass": [("cutoff", 60.0, 1500.0, True), ("q", 0.5, 1.6, True)],
    "tanh": [("drive", 1.0, 4.0, True), ("bias", -0.1, 0.1, False), ("gain", 0.3, 0.8, False)],
    "width": [("width", 0.3, 1.8, False)],
    "reverb": [("t60", 0.4, 3.5, True), ("size", 0.7, 1.5, False), ("damping", 0.05, 0.6, False),
               ("wet", 0.08, 0.35, False)],
    "echo": [("delay_left", 0.12, 0.5, False), ("delay_right", 0.07, 0.75, False),
             ("feedback", 0.1, 0.55, False), ("wet", 0.1, 0.35, False)],
}
ECHO_PARAMS = [("delay", 0.05, 1.0, False), ("decay", 0.35, 0.75, False)]
GAIN_RANGE = (-48.0, 0.0)


def _offsets(table):
    offsets, width = {}, 0
    for name in table:
        offsets[name] = width
        width += len(table[name])
    return offsets, width


INSERT_OFFSET, INSERT_WIDTH = _offsets(INSERT_PARAMS)
BUS_OFFSET, BUS_WIDTH = _offsets(BUS_PARAMS)


def normalise(value, low, high, log):
    if log:
        value, low, high = np.log(max(value, 1e-9)), np.log(low), np.log(high)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def denormalise(x, low, high, log):
    x = float(np.clip(x, 0.0, 1.0))
    return float(np.exp(np.log(low) + x * np.log(high / low))) if log else float(low + x * (high - low))


@lru_cache(maxsize=None)
def used_dims(family):
    """Mask of u coordinates that change the construction of ``family``."""
    base = np.full(U_DIMS, 0.5)
    reference = prepare.make_document(family, base)[0]
    mask = np.zeros(U_DIMS, bool)
    for i in range(U_DIMS):
        probe = base.copy()
        probe[i] = 0.9
        mask[i] = prepare.make_document(family, probe)[0] != reference
    return mask


def level_units(scene, level):
    """Units of one ladder level, in the order ``mixscene.ladder`` numbers them."""
    units = []
    for inst in scene["instruments"]:
        echo = inst["note_echo"]
        if level == "L0":
            units.append({"layers": inst["layers"], "insert_fx": inst["insert_fx"], "note_echo": echo,
                          "notes": inst["notes"], "gain_db": inst["gain_db"]})
            continue
        for layer in inst["layers"]:
            taps = echo["taps"] if echo and level == "L2" else 0
            # A layer of its own plays its transpose as notes: a construction whose MIDI
            # frequency projections are scaled by 2^(s/12) playing n sounds like it playing n+s.
            shift = layer["transpose"]
            layer = {**layer, "transpose": 0}
            for tap in range(taps + 1):
                notes = inst["notes"] if tap == 0 else mixscene.echo_notes(inst["notes"], echo, tap)
                notes = [[a, b, p + shift, v] for a, b, p, v in notes]
                gain = inst["gain_db"] + (20 * np.log10(echo["decay"]) * tap if tap else 0.0)
                units.append({"layers": [layer],
                              "insert_fx": inst["insert_fx"], "note_echo": echo if level == "L1" else None,
                              "notes": notes, "gain_db": float(gain)})
    return units


ROLL_CHANNELS = ("onset", "sustain", "velocity", "offset", "length")
MAX_LENGTH = 4.0  # seconds; note length is a regression target at the onset frame


def piano_roll(notes, frames):
    """[frames, PITCHES, 5] onset / sustain / velocity / sub-frame onset offset / length.

    Offset (fraction of a hop) and length (seconds / MAX_LENGTH) are read at the onset
    frame, so decoded timing is exact rather than snapped to the 21 ms frame grid.
    """
    roll = np.zeros((frames, PITCHES, len(ROLL_CHANNELS)), np.float32)
    rate = mixscene.SAMPLE_RATE / mixscene.HOP
    for onset, length, pitch, velocity in notes:
        p = pitch - PITCH_LOW
        position = onset * rate
        start = int(np.floor(position + 0.5))
        if not 0 <= p < PITCHES or start >= frames:
            continue
        end = max(start + 1, int(round((onset + length) * rate)))
        roll[start, p, 0] = 1.0
        roll[start:end, p, 1] = 1.0
        roll[start:end, p, 2] = velocity
        roll[start, p, 3] = position - start + 0.5  # in [0, 1)
        roll[start, p, 4] = min(length / MAX_LENGTH, 1.0)
    return roll


def unit_target(unit, frames):
    first = unit["layers"][0]
    second = unit["layers"][1] if len(unit["layers"]) > 1 else None
    insert_present = np.zeros(len(INSERT_FX), np.float32)
    insert_values = np.zeros(INSERT_WIDTH, np.float32)
    for fx in unit["insert_fx"]:
        insert_present[INSERT_FX.index(fx["type"])] = 1
        for j, (name, low, high, log) in enumerate(INSERT_PARAMS[fx["type"]]):
            raw = fx["taps"][j] if fx["type"] == "fir" else fx[name] / (
                mixscene.SAMPLE_RATE if fx["type"] == "echo" and name == "delay" else 1)
            insert_values[INSERT_OFFSET[fx["type"]] + j] = normalise(raw, low, high, log)
    echo = unit["note_echo"]
    return {
        "family1": FAMILIES.index(first["family"]),
        "u1": np.asarray(first["u"], np.float32),
        "u1_mask": used_dims(first["family"]),
        "family2": NO_LAYER if second is None else FAMILIES.index(second["family"]),
        "u2": np.asarray(second["u"] if second else np.zeros(U_DIMS), np.float32),
        "u2_mask": used_dims(second["family"]) if second else np.zeros(U_DIMS, bool),
        "transpose2": LAYER_TRANSPOSE.index(second["transpose"]) if second else 0,
        "insert_present": insert_present,
        "insert_values": insert_values,
        "echo_present": float(echo is not None),
        "echo_taps": (echo["taps"] - 1) if echo else 0,
        "echo_transpose": ECHO_TRANSPOSE.index(echo["transpose"]) if echo else 0,
        "echo_values": np.asarray([normalise(echo[n], lo, hi, lg) for n, lo, hi, lg in ECHO_PARAMS]
                                  if echo else [0.0] * len(ECHO_PARAMS), np.float32),
        "gain": normalise(unit["gain_db"], *GAIN_RANGE, False),
        "roll": piano_roll(unit["notes"], frames),
    }


def bus_target(chain):
    present = np.zeros(len(BUS_FX), np.float32)
    values = np.zeros(BUS_WIDTH, np.float32)
    for fx in chain:
        present[BUS_FX.index(fx["type"])] = 1
        if fx["type"] == "reverb":
            raw = [fx["t60"], fx["size"], fx["damping"][0], fx["wet"]]
        elif fx["type"] == "echo":
            raw = [fx["delays"][0] / mixscene.SAMPLE_RATE, fx["delays"][1] / mixscene.SAMPLE_RATE,
                   fx["feedback"], fx["wet"]]
        else:
            raw = [fx[name] for name, *_ in BUS_PARAMS[fx["type"]]]
        for j, (value, (_, low, high, log)) in enumerate(zip(raw, BUS_PARAMS[fx["type"]])):
            values[BUS_OFFSET[fx["type"]] + j] = normalise(value, low, high, log)
    return present, values


def unit_stems(stems, atom_levels, level):
    """[slots, stem frames, bands] log power of each unit = log sum of its atoms' power."""
    column = LEVELS.index(level)
    floor = np.log10(mixscene.LOG_FLOOR)
    out = np.full((mixscene.MAX_INSTRUMENTS,) + stems.shape[1:], floor, np.float32)
    for unit in range(mixscene.MAX_INSTRUMENTS):
        rows = atom_levels[:, column] == unit
        if rows.any():
            out[unit] = np.log10(np.sum(10.0 ** stems[rows].astype(np.float64), 0) + mixscene.LOG_FLOOR)
    return out


def scene_targets(scene, level, frames=mixscene.MAX_FRAMES):
    """Stacked slot targets (MAX_INSTRUMENTS slots, padded) of one scene at one level."""
    units = [unit_target(u, frames) for u in level_units(scene, level)]
    slots = mixscene.MAX_INSTRUMENTS
    out = {}
    for key, value in units[0].items():
        array = np.asarray(value)
        out[key] = np.zeros((slots,) + array.shape, array.dtype)
        for i, unit in enumerate(units):
            out[key][i] = unit[key]
    out["exists"] = (np.arange(slots) < len(units)).astype(np.float32)
    out["bus_present"], out["bus_values"] = bus_target(scene["bus_fx"])
    out["count"] = len(units)
    return out


# --------------------------------------------------------------------------- decode


def roll_notes(roll, threshold=0.5):
    """Monophonic note list from one slot's roll [frames, PITCHES, 5] (probabilities/values)."""
    rate = mixscene.SAMPLE_RATE / mixscene.HOP
    onset, velocity, offset, length = roll[..., 0], roll[..., 2], roll[..., 3], roll[..., 4]
    notes = []
    for t in range(len(onset)):
        p = int(np.argmax(onset[t]))
        if onset[t, p] < threshold:
            continue
        start = (t + float(np.clip(offset[t, p], 0, 1)) - 0.5) / rate
        notes.append([max(start, 0.0), max(float(length[t, p]) * MAX_LENGTH, 1.0 / rate), p + PITCH_LOW,
                      float(np.clip(velocity[t, p], 0.05, 1.0))])
    return notes


def decode_unit(pred):
    """One slot's prediction (numpy, probabilities already applied) -> scene instrument spec."""
    layers = [{"family": FAMILIES[int(pred["family1"])], "u": np.clip(pred["u1"], 0, 1).tolist(),
               "transpose": 0}]
    if int(pred["family2"]) != NO_LAYER:
        layers.append({"family": FAMILIES[int(pred["family2"])], "u": np.clip(pred["u2"], 0, 1).tolist(),
                       "transpose": LAYER_TRANSPOSE[int(pred["transpose2"])]})
    chain = []
    for index, kind in enumerate(INSERT_FX):
        if pred["insert_present"][index] < 0.5:
            continue
        values = [denormalise(pred["insert_values"][INSERT_OFFSET[kind] + j], lo, hi, lg)
                  for j, (_, lo, hi, lg) in enumerate(INSERT_PARAMS[kind])]
        if kind == "fir":
            chain.append({"type": kind, "taps": values})
        elif kind == "echo":
            chain.append({"type": kind, "delay": int(round(values[0] * mixscene.SAMPLE_RATE)),
                          "feedback": values[1], "wet": values[2]})
        else:
            chain.append({"type": kind, **{n: v for (n, *_), v in zip(INSERT_PARAMS[kind], values)}})
    echo = None
    if pred["echo_present"] >= 0.5:
        delay, decay = [denormalise(x, lo, hi, lg) for x, (_, lo, hi, lg) in zip(pred["echo_values"], ECHO_PARAMS)]
        echo = {"taps": int(pred["echo_taps"]) + 1, "delay": delay,
                "transpose": ECHO_TRANSPOSE[int(pred["echo_transpose"])], "decay": decay}
    notes = roll_notes(pred["roll"])
    return {"role": "decoded", "layers": layers, "insert_fx": chain, "note_echo": echo, "doubles": None,
            "notes": notes, "gain_db": denormalise(pred["gain"], *GAIN_RANGE, False)}


def decode_bus(present, values):
    chain = []
    for index, kind in enumerate(BUS_FX):
        if present[index] < 0.5:
            continue
        v = [denormalise(values[BUS_OFFSET[kind] + j], lo, hi, lg)
             for j, (_, lo, hi, lg) in enumerate(BUS_PARAMS[kind])]
        if kind == "reverb":
            chain.append(mixscene.reverb_spec(*v))
        elif kind == "echo":
            chain.append({"type": kind, "delays": [int(round(v[0] * mixscene.SAMPLE_RATE)), int(round(v[1] * mixscene.SAMPLE_RATE))],
                          "feedback": v[2], "wet": v[3]})
        else:
            chain.append({"type": kind, **{n: x for (n, *_), x in zip(BUS_PARAMS[kind], v)}})
    return chain


def decode_scene(units, bus_present, bus_values, seconds):
    """Scene spec that ``mixscene.render_scene`` renders, from decoded existing slots."""
    instruments = [decode_unit(u) for u in units]
    return {"schema": mixscene.SCHEMA, "seed": None, "index": None, "sample_rate": mixscene.SAMPLE_RATE,
            "seconds": float(seconds), "instruments": [i for i in instruments if i["notes"]],
            "bus_fx": decode_bus(bus_present, bus_values)}


def target_as_prediction(targets, slot):
    """A slot target in prediction form (used by tests and as an oracle baseline)."""
    return {"family1": targets["family1"][slot], "u1": targets["u1"][slot],
            "family2": targets["family2"][slot], "u2": targets["u2"][slot],
            "transpose2": targets["transpose2"][slot], "insert_present": targets["insert_present"][slot],
            "insert_values": targets["insert_values"][slot], "echo_present": targets["echo_present"][slot],
            "echo_taps": targets["echo_taps"][slot], "echo_transpose": targets["echo_transpose"][slot],
            "echo_values": targets["echo_values"][slot], "gain": targets["gain"][slot],
            "roll": targets["roll"][slot]}
