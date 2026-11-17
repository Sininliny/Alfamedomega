#!/usr/bin/env python3
"""Mix -> one preset per instrument, for AMO Centre (MIX-DESIGN.md §5, §15).

Input: a multi-instrument mixed WAV (any length, any rate). Output: one folder with

    01 <family>.amo.json   one polyphonic instrument preset per instrument (voice_indexed MIDI)
    01 <family>.mid        that instrument's notes (note echo baked in as extra notes, see below)
    bus.amo.json           the bus FX chain as an FX preset (only when the mix has bus FX)
    report.json            what was decided, per window and per instrument

Instrument count: ``--count auto`` (default) or ``--count N`` (1..8, exactly N instruments).

Note echo (a synth whose pitched echo must stay one instrument, §2) is one instrument with
``note_echo`` metadata. The runtime has no note-echo primitive yet, so the ``.mid`` file carries
the echo taps as extra notes (velocity scaled by the decay) unless ``--echo-metadata-only``.

Long mixes are cut into windows of at most 8 s (the generator's longest scene). Each window is
decomposed on its own; instruments are linked across windows by preset distance (same family,
nearby construction coordinates; Hungarian matching), notes are concatenated with the window
offset, and each instrument keeps the construction of the window where it is loudest.

    python -m training.amo.corpus.mixexport MODEL.pt song.wav --out exported/ [--count 3]
        [--rescore mix-rescore.json] [--transcriber model.pt --separator model.pt]

``--rescore`` uses the tuned λ of mixinfer (render every candidate, pick the lowest objective);
without it the network's own choice is used. The writer half (``write_export``) needs no torch.
"""

import argparse
import copy
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus import mixscene, prepare, wav_to_preset  # noqa: E402
from training.amo.native import NativeRuntime  # noqa: E402

SCHEMA = "amo.mix-export/1"
WINDOW = mixscene.SECONDS[1]  # s
MIN_WINDOW = 1.0  # a shorter tail is folded into the previous window
LINK_COST = 1.5  # above this, two windows' instruments are different instruments
PPQ = 480
TEMPO_US = 500000  # 120 BPM: 1 s = 960 ticks; DAWs re-time from the file's tempo map


# ------------------------------------------------------------------------- MIDI


def _varlen(value):
    value = int(value)
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(out))


def midi_bytes(notes, name, ppq=PPQ, tempo_us=TEMPO_US):
    """Standard MIDI file (format 1: tempo track + one note track, channel 1) for [[onset s, length s, note, velocity]]."""
    ticks_per_second = ppq * 1e6 / tempo_us
    events = []
    for onset, length, pitch, velocity in notes:
        pitch = int(np.clip(round(pitch), 0, 127))
        start = int(round(max(onset, 0.0) * ticks_per_second))
        end = max(start + 1, int(round((max(onset, 0.0) + length) * ticks_per_second)))
        v = int(np.clip(round(velocity * 127), 1, 127))
        events.append((end, 0, bytes([0x80, pitch, 0])))  # note-offs first at equal ticks
        events.append((start, 1, bytes([0x90, pitch, v])))
    events.sort(key=lambda e: (e[0], e[1]))
    label = name.encode("utf-8")[:127]
    track = bytearray(b"\x00\xff\x03" + _varlen(len(label)) + label)
    now = 0
    for tick, _, data in events:
        track += _varlen(tick - now) + data
        now = tick
    track += b"\x00\xff\x2f\x00"
    tempo = b"\x00\xff\x51\x03" + tempo_us.to_bytes(3, "big") + b"\x00\xff\x58\x04\x04\x02\x18\x08" + b"\x00\xff\x2f\x00"
    chunk = lambda body: b"MTrk" + struct.pack(">I", len(body)) + bytes(body)
    return b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + chunk(tempo) + chunk(track)


def played_notes(inst, bake_echo=True):
    """The notes a DAW must play for this instrument: stored notes plus (optionally) the echo taps."""
    notes = [list(n) for n in inst["notes"]]
    echo = inst.get("note_echo")
    if bake_echo and echo:
        for tap in range(1, int(echo["taps"]) + 1):
            for onset, length, pitch, velocity in mixscene.echo_notes(inst["notes"], echo, tap):
                notes.append([onset, length, pitch, float(np.clip(velocity * echo["decay"] ** tap, 0.02, 1.0))])
    return sorted(notes)


# ----------------------------------------------------------------------- presets


def instrument_preset(inst, name, voices, source):
    """Polyphonic instrument preset of one decoded instrument (all layers, insert chain, gain)."""
    mono = mixscene.preset_document(inst)
    gain = 0.5 * 10 ** (float(inst.get("gain_db", 0.0)) / 20)
    families = "+".join(layer["family"] for layer in inst["layers"])
    preset = wav_to_preset.make_preset(mono, name, voices, gain=gain, source=source, guess=families)
    preset["preset"]["category"] = "Mix import"
    preset["description"] = (f"Instrument {name} decomposed from {source} by the AMO mix decomposer"
                             f" ({families})")
    preset["authoring"] = {"kind": "mix to presets (mix decomposer guess)", "families": families,
                           "source": source, "semantic_mappings": False,
                           "note_echo": inst.get("note_echo"), "insert_fx": [fx["type"] for fx in inst["insert_fx"]]}
    return preset


def tail_seconds(chain):
    tail = 0.1
    for fx in chain:
        if fx["type"] == "reverb":
            tail = max(tail, float(fx.get("t60", 2.0)) * 1.5)
        elif fx["type"] == "echo":
            delay = max(fx["delays"]) / mixscene.SAMPLE_RATE
            tail = max(tail, delay * np.log(1e-3) / np.log(max(abs(fx["feedback"]), 1e-3)))
    return float(np.clip(tail, 0.1, 30.0))


def bus_preset(chain, name, source):
    doc = mixscene.bus_document(chain)
    doc["preferred_mode"] = 1
    doc["performance"] = {"mode": "fx"}
    doc["tail_seconds"] = tail_seconds(chain)
    doc["preset"] = {"schema": "amo.preset/1", "id": wav_to_preset.uuid.uuid4().__str__(), "name": name,
                     "category": "Mix import"}
    doc["description"] = f"Bus FX of {source}: " + ", ".join(fx["type"] for fx in chain)
    doc["authoring"] = {"kind": "mix to presets (bus FX guess)", "source": source, "semantic_mappings": False,
                        "chain": [fx["type"] for fx in chain]}
    with NativeRuntime(doc, 48000.0, prepare.LIB):
        pass  # the native compiler must accept it, or nothing is written
    return doc


def write_export(scene, out, source, voices=8, bake_echo=True, extra=None, preview=False, target=None):
    """Write the folder for one decoded scene (``mixscene`` spec, ``seconds`` = full length). No torch."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "source": source, "seconds": scene["seconds"], "instruments": [],
              "bus": None, **(extra or {})}
    used = set()
    for k, inst in enumerate(scene["instruments"], 1):
        stem = f"{k:02d} {inst['layers'][0]['family']}"
        while stem in used:
            stem += "+"
        used.add(stem)
        entry = {"index": k, "families": [l["family"] for l in inst["layers"]],
                 "transpose": [l["transpose"] for l in inst["layers"]], "insert_fx": inst["insert_fx"],
                 "note_echo": inst.get("note_echo"), "gain_db": inst.get("gain_db"),
                 "stored_notes": len(inst["notes"]), "warnings": []}
        try:
            preset = instrument_preset(inst, stem, voices, source)
            if not wav_to_preset.playable(preset):
                entry["warnings"].append("some voice rows do not render (idle voices or low notes)")
            path = out / f"{stem}.amo.json"
            path.write_text(json.dumps(preset, indent=1), encoding="utf-8")
            entry["preset"] = path.name
        except Exception as error:  # noqa: BLE001 - the MIDI is still useful on its own
            entry["error"] = f"preset: {error}"
        notes = played_notes(inst, bake_echo)
        (out / f"{stem}.mid").write_bytes(midi_bytes(notes, stem))
        entry["midi"] = f"{stem}.mid"
        entry["midi_notes"] = len(notes)
        entry["echo_in_midi"] = bool(bake_echo and inst.get("note_echo"))
        report["instruments"].append(entry)
    if scene.get("bus_fx"):
        try:
            doc = bus_preset(scene["bus_fx"], "bus", source)
            (out / "bus.amo.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
            report["bus"] = {"preset": "bus.amo.json", "chain": scene["bus_fx"]}
        except Exception as error:  # noqa: BLE001
            report["bus"] = {"error": str(error), "chain": scene["bus_fx"]}
    if preview:
        report["preview"] = write_preview(scene, out, target)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return report


def write_preview(scene, out, target=None):
    """preview.wav (the whole decoded scene) and one <stem>.preview.wav per instrument, 24 kHz."""
    from training.amo.corpus import resynth

    stems, mix, _ = mixscene.render_scene(scene)
    atoms = mixscene.atoms_of(scene)
    peak = float(np.abs(mix).max()) or 1.0
    prepare.write_wav(out / "preview.wav", mix * (0.9 / peak), mixscene.SAMPLE_RATE)
    for k in range(len(scene["instruments"])):
        own = sum(s for s, a in zip(stems, atoms) if a["instrument"] == k)
        prepare.write_wav(out / f"{k + 1:02d}.preview.wav", np.asarray(own) * (0.9 / peak), mixscene.SAMPLE_RATE)
    result = {"mix": "preview.wav", "instruments": [f"{k + 1:02d}.preview.wav"
                                                   for k in range(len(scene["instruments"]))]}
    if target is not None:  # the log-spectral distance of mixinfer / resynth between input and preview
        frames = min(len(target), len(mix))
        result["distance"] = float(resynth.distance(resynth.log_spectra(mix[:frames]),
                                                    resynth.log_spectra(target[:frames])))
    return result


# ----------------------------------------------------------------------- windows


def windows(frames, rate=mixscene.SAMPLE_RATE, length=WINDOW):
    """[start, end) sample ranges of ≤ ``length`` s; a tail under MIN_WINDOW is evened out with the previous one."""
    size = int(length * rate)
    starts = list(range(0, frames, size))
    spans = [[s, min(s + size, frames)] for s in starts]
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < MIN_WINDOW * rate:
        end = spans.pop()[1]  # split the last two windows evenly (each stays ≤ length)
        middle = (spans[-1][0] + end) // 2
        spans[-1][1] = middle
        spans.append([middle, end])
    return [tuple(s) for s in spans]


def link_cost(a, b):
    """Preset distance between two decoded instruments of neighbouring windows."""
    la, lb = a["layers"][0], b["layers"][0]
    cost = 0.0 if la["family"] == lb["family"] else 1.0
    cost += float(np.mean(np.abs(np.asarray(la["u"]) - np.asarray(lb["u"]))))
    cost += 0.3 * (len(a["layers"]) != len(b["layers"]))
    cost += 0.02 * abs(a["gain_db"] - b["gain_db"])
    return cost


def link_windows(parts, fixed=None, limit=mixscene.MAX_INSTRUMENTS):
    """Join per-window scenes [(offset s, scene)] into tracks: one instrument list over the whole mix."""
    from scipy.optimize import linear_sum_assignment

    tracks = []  # {"members": [(offset, inst)], "last": inst}
    for offset, scene in parts:
        current = scene["instruments"]
        if tracks and current:
            cost = np.array([[link_cost(t["last"], inst) for inst in current] for t in tracks])
            rows, cols = linear_sum_assignment(cost)
            taken = set()
            for r, c in zip(rows, cols):
                if fixed is not None or cost[r, c] <= LINK_COST:
                    tracks[r]["members"].append((offset, current[c]))
                    tracks[r]["last"] = current[c]
                    taken.add(c)
            rest = [c for c in range(len(current)) if c not in taken]
        else:
            rest = list(range(len(current)))
        for c in rest:
            if len(tracks) < (fixed or limit):
                tracks.append({"members": [(offset, current[c])], "last": current[c]})
            else:  # no room: join the nearest track
                best = min(range(len(tracks)), key=lambda t: link_cost(tracks[t]["last"], current[c]))
                tracks[best]["members"].append((offset, current[c]))
    instruments = []
    for track in tracks:
        loudest = max(track["members"], key=lambda m: m[1]["gain_db"] + 10 * np.log10(len(m[1]["notes"]) + 1))
        inst = copy.deepcopy(loudest[1])
        inst["notes"] = sorted([n[0] + offset, *n[1:]] for offset, member in track["members"]
                               for n in member["notes"])
        inst["windows"] = len(track["members"])
        instruments.append(inst)
    return instruments


def bus_of(parts):
    """The bus chain of the window whose chain is the most common (by FX types)."""
    chains = [scene["bus_fx"] for _, scene in parts]
    keys = [tuple(fx["type"] for fx in c) for c in chains]
    best = max(set(keys), key=keys.count)
    return chains[keys.index(best)]


# --------------------------------------------------------------------- inference


def decompose_window(model, audio, fixed=None, lam=None):
    """One window (≤ 8 s, 24 kHz stereo) -> decoded scene and provenance. Needs torch."""
    from training.amo.corpus import mixinfer, mixtargets as T, mixtrain, resynth

    raw, _ = mixscene.mix_features(audio)
    frames = min(len(raw), mixscene.MAX_FRAMES)
    features = np.full((mixscene.MAX_FRAMES, raw.shape[1]), np.log10(mixscene.LOG_FLOOR), np.float32)
    features[:frames] = raw[:frames]  # padded like the corpus (mixscene.generate_shard)
    seconds = len(audio) / mixscene.SAMPLE_RATE
    if lam is None:
        units, bus_present, bus_values = mixtrain.predict(model, features, frames, fixed or 0)
        scene = T.decode_scene(units, bus_present, bus_values, seconds)
        if fixed is None:  # still fold pitched echo copies into their source (§2)
            scene["instruments"] = mixinfer.merge_sequence(scene["instruments"])[-1]
        return scene, {"mode": "network"}
    target = resynth.log_spectra(audio)
    found = mixinfer.candidate_scenes(model, features, frames, seconds, fixed)
    rows = []
    for scene, info in found:
        rows.append({**mixinfer.cost_terms(scene), "D": mixinfer.render_distance((scene, target)), **info})
    best = mixinfer.choose(rows, lam, fixed is not None)
    if best is None:
        raise ValueError("no candidate renders")
    return found[rows.index(best)][0], {"mode": "rescored", "candidates": len(rows),
                                        "chosen": {k: best[k] for k in ("K", "N", "F", "B", "D", "merges")}}


def decompose(model, audio, fixed=None, lam=None):
    """Whole mix (24 kHz stereo) -> one scene over its full length, plus per-window notes."""
    parts, info = [], []
    for start, end in windows(len(audio)):
        piece = audio[start:end]
        if np.sqrt(np.mean(piece ** 2)) < 1e-5:
            info.append({"start": start / mixscene.SAMPLE_RATE, "silent": True})
            continue
        scene, provenance = decompose_window(model, piece, fixed, lam)
        parts.append((start / mixscene.SAMPLE_RATE, scene))
        info.append({"start": start / mixscene.SAMPLE_RATE, "seconds": (end - start) / mixscene.SAMPLE_RATE,
                     "instruments": len(scene["instruments"]), **provenance})
    if not parts:
        raise ValueError("the input is silent")
    instruments = parts[0][1]["instruments"] if len(parts) == 1 else link_windows(parts, fixed)
    if fixed is not None and len(instruments) > fixed:
        instruments = instruments[:fixed]
    scene = {"schema": mixscene.SCHEMA, "seed": None, "index": None, "sample_rate": mixscene.SAMPLE_RATE,
             "seconds": len(audio) / mixscene.SAMPLE_RATE, "instruments": instruments, "bus_fx": bus_of(parts)}
    return scene, info


def read_mix(path):
    from training.amo.corpus import resynth_wav

    audio, rate = resynth_wav.read_wav(path)
    audio = resynth_wav.resample(audio, rate, mixscene.SAMPLE_RATE)
    peak = float(np.abs(audio).max())
    if peak < 1e-6:
        raise SystemExit(f"{path}: silent")
    return (audio * (mixscene.PEAK / peak)).astype(np.float64), rate


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="mix decomposer model.pt (mixtrain)")
    parser.add_argument("input", type=Path, help="mixed WAV")
    parser.add_argument("--out", type=Path, default=None, help="export folder (default: <input stem> presets/)")
    parser.add_argument("--count", default="auto", help="auto, or the exact number of instruments 1..8")
    parser.add_argument("--rescore", type=Path, default=None, help="mixinfer tune output (λ); renders candidates")
    parser.add_argument("--voices", type=int, default=8)
    parser.add_argument("--echo-metadata-only", action="store_true",
                        help="do not write note-echo taps into the MIDI files")
    parser.add_argument("--preview", action="store_true",
                        help="also render preview.wav and one preview per instrument, with the distance to the input")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    fixed = None if args.count == "auto" else int(args.count)
    if fixed is not None and not 1 <= fixed <= mixscene.MAX_INSTRUMENTS:
        raise SystemExit(f"--count must be auto or 1..{mixscene.MAX_INSTRUMENTS}")
    lam = json.loads(args.rescore.read_text())["lambda"] if args.rescore else None
    import torch
    from training.amo.corpus import mixtrain

    torch.set_num_threads(max(1, args.threads))
    model = mixtrain.load_model(args.model)
    audio, rate = read_mix(args.input)
    scene, info = decompose(model, audio, fixed, lam)
    out = args.out or args.input.with_name(f"{args.input.stem} presets")
    report = write_export(scene, out, args.input.name, args.voices, not args.echo_metadata_only,
                          {"count": args.count, "input_rate": rate, "lambda": lam, "windows": info,
                           "model": str(args.model)}, args.preview, audio)
    for entry in report["instruments"]:
        print(f"{entry.get('preset', entry.get('error'))}  {entry['midi']}  notes {entry['midi_notes']}"
              + (f"  echo x{entry['note_echo']['taps']}" if entry["note_echo"] else ""), flush=True)
    if report.get("preview", {}).get("distance") is not None:
        print("distance to input:", round(report["preview"]["distance"], 3), flush=True)
    print("bus:", report["bus"] and report["bus"].get("preset", report["bus"].get("error")), flush=True)
    print("RESULT " + json.dumps({"out": str(out), "instruments": len(report["instruments"])}), flush=True)
    return 0 if all("preset" in e for e in report["instruments"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
