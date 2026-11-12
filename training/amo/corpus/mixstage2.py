#!/usr/bin/env python3
"""Two-stage mix decomposition (MIX-DESIGN.md §12): separate, then let the one-shot encoder guess each part.

Stage 1 (``mixtrain``) gives K slots with notes, coarse stem energies and bus FX. Stage 2:

1. soft-mask the mix STFT with the slots' stem energies -> one audio track per slot;
2. cut up to three clean reference notes out of each track;
3. let the corpus encoder (run 3) guess a construction from each reference note, render every
   guess with the slot's full note list, fit the gain, keep the one closest to the track;
4. assemble the scene (guessed constructions + the slot's time FX + predicted bus) and render.

``evaluate`` measures the ceiling first (true atom stems and true notes in place of steps 1-2)
and then the real pipeline, next to the network-only baseline.

    python -m training.amo.corpus.mixstage2 evaluate EVAL_CORPUS MIX_MODEL ENCODER_MODEL --scenes 64
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus import mixscene, prepare, resynth  # noqa: E402

SR = mixscene.SAMPLE_RATE
NOTE_RANGE = (34, 69)          # the encoder's training range (resynth_wav.NOTE_RANGE)
CLIP_SECONDS = (1.2, 2.4)      # the encoder's clip lengths
NOTE_OFF_RANGE = (0.32, 0.62)  # the encoder's held-note lengths
REFERENCES = 3
TIME_FX = ("echo",)            # insert FX the encoder cannot absorb from one note


# ----------------------------------------------------------------- separation


def _stft(channel, n_fft=mixscene.N_FFT, hop=mixscene.HOP):
    padded = np.pad(channel, (n_fft // 2, n_fft // 2))
    count = 1 + (len(padded) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(padded, (count, n_fft), (padded.strides[0] * hop, padded.strides[0]))
    return np.fft.rfft(frames * np.hanning(n_fft), axis=1)


def _istft(spectrum, length, n_fft=mixscene.N_FFT, hop=mixscene.HOP):
    window = np.hanning(n_fft)
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1) * window
    out = np.zeros(length + n_fft)
    norm = np.zeros(length + n_fft)
    for i, frame in enumerate(frames):
        out[i * hop:i * hop + n_fft] += frame
        norm[i * hop:i * hop + n_fft] += window ** 2
    out = out / np.maximum(norm, 1e-8)
    return out[n_fft // 2:n_fft // 2 + length]


def band_of_bin(n_fft=mixscene.N_FFT, bands=mixscene.STEM_BANDS):
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    edges = np.geomspace(mixscene.F_LOW, mixscene.F_HIGH, bands + 1)
    return np.clip(np.searchsorted(edges, freqs, side="right") - 1, 0, bands - 1)


def separate(mix, stems_log):
    """Soft-mask separation: ``stems_log`` [K, stem frames, 64] log10 power -> K stereo tracks summing to mix."""
    k = len(stems_log)
    spectra = [_stft(c) for c in mix.T]
    frames = spectra[0].shape[0]
    stem_times = np.arange(stems_log.shape[1]) * mixscene.STEM_HOP
    times = np.arange(frames) * mixscene.HOP
    power = 10.0 ** np.asarray(stems_log, np.float64)
    power = np.stack([np.stack([np.interp(times, stem_times, power[s, :, b]) for b in range(power.shape[2])], 1)
                      for s in range(k)])  # K, frames, bands
    masks = power[:, :, band_of_bin()]  # K, frames, bins
    masks = masks / (masks.sum(0, keepdims=True) + 1e-12)
    return [np.stack([_istft(spec * masks[s], len(mix)) for spec in spectra], 1) for s in range(k)]


# ----------------------------------------------------------- reference notes


def reference_notes(notes, seconds, count=REFERENCES):
    """Indices of up to ``count`` notes best suited to a one-note guess (in range, long gap)."""
    order = sorted(range(len(notes)), key=lambda i: notes[i][0])
    scored = []
    for rank, i in enumerate(order):
        onset, _, pitch, _ = notes[i]
        following = notes[order[rank + 1]][0] if rank + 1 < len(order) else seconds
        gap = following - onset
        inside = NOTE_RANGE[0] <= pitch <= NOTE_RANGE[1]
        scored.append((inside, min(gap, CLIP_SECONDS[1]), i, gap))
    scored.sort(reverse=True)
    return [(i, gap) for _, _, i, gap in scored[:count]]


def note_clip(track, note, gap):
    """The encoder input for one reference note: clip audio and performance plan."""
    from training.amo.corpus import resynth_wav

    onset, length, pitch, velocity = note
    start = max(0, int(round((onset - 0.005) * SR)))
    stop = min(len(track), start + int(round(min(gap, CLIP_SECONDS[1]) * SR)))
    frames = int(np.clip(stop - start, CLIP_SECONDS[0] * SR, CLIP_SECONDS[1] * SR))
    audio = np.zeros((frames, 2))
    audio[:stop - start] = track[start:stop][:frames]
    perf = resynth_wav.performance(int(pitch), 0.0, float(np.clip(velocity, 0.05, 1.0)),
                                   float(np.clip(length, *NOTE_OFF_RANGE)), frames)
    return audio, perf


def encode_track(encoder, track, notes, seconds):
    """Candidate constructions for one track: one guess per reference note, plus a second family."""
    from training.amo.corpus import resynth_wav

    candidates = []
    for n, (i, gap) in enumerate(reference_notes(notes, seconds)):
        audio, perf = note_clip(track, notes[i], gap)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-7:
            continue
        audio = audio * min(np.exp(encoder.typical_log_rms) / rms, 0.9 / float(np.abs(audio).max()))
        x, conditions, _ = resynth_wav.model_input(audio, perf)
        conditions[1] = perf["frames"] / perf["sample_rate"]
        type_id, values, probabilities = encoder.guess(x, conditions)
        candidates.append({"type_id": type_id, "document": resynth.assemble(type_id, values), "note": i,
                           "probability": float(probabilities[type_id])})
        if n == 0:
            second, values2, _ = encoder.guess(x, conditions, exclude=(type_id,))
            candidates.append({"type_id": second, "document": resynth.assemble(second, values2), "note": i,
                               "probability": float(probabilities[second])})
    return candidates


# ------------------------------------------------------------------ choice


def render_part(document, notes, time_fx, frames):
    inst = {"layers": [{"family": None, "u": None, "transpose": 0, "document": document}],
            "insert_fx": time_fx, "notes": notes}
    return mixscene.render_document(mixscene.layer_document(inst, 0), frames, notes)


def score_candidate(job):
    """(distance to the track, linear gain) of one candidate construction."""
    document, notes, time_fx, track = job
    try:
        audio = render_part(document, notes, time_fx, len(track))
    except Exception:  # noqa: BLE001
        return float("inf"), 0.0
    rms, target_rms = float(np.sqrt(np.mean(audio ** 2))), float(np.sqrt(np.mean(track ** 2)))
    if rms < 1e-9 or target_rms < 1e-9:
        return float("inf"), 0.0
    return resynth.distance(resynth.log_spectra(audio), resynth.log_spectra(track)), target_rms / rms


def assemble_scene(parts, bus_fx, seconds):
    """Scene spec from chosen parts: {document, notes, gain (linear), time_fx, note_echo}."""
    instruments = []
    for p in parts:
        if p is None or not p["notes"] or p["gain"] <= 0:
            continue
        instruments.append({"role": "decoded", "layers": [{"family": None, "u": None, "transpose": 0,
                                                           "document": p["document"]}],
                            "insert_fx": p["time_fx"], "note_echo": p.get("note_echo"), "doubles": None,
                            "notes": p["notes"], "gain_db": float(20 * np.log10(p["gain"]))})
    return {"schema": mixscene.SCHEMA, "seed": None, "index": None, "sample_rate": SR, "seconds": float(seconds),
            "instruments": instruments, "bus_fx": bus_fx}


def mix_distance(scene, target):
    if not scene["instruments"]:
        return float("inf")
    try:
        return resynth.distance(resynth.log_spectra(mixscene.render_scene(scene)[1]), target)
    except Exception:  # noqa: BLE001
        return float("inf")


def choose_parts(encoder, tracks, slots, pool):
    """For each track (with its slot's notes / time FX), the best encoder construction and its gain."""
    jobs, owners, candidates = [], [], []
    for s, (track, slot) in enumerate(zip(tracks, slots)):
        found = encode_track(encoder, track, slot["notes"], len(track) / SR) if slot["notes"] else []
        candidates.append(found)
        for c in found:
            jobs.append((c["document"], slot["notes"], slot["time_fx"], track))
            owners.append(s)
    scores = list(pool.map(score_candidate, jobs)) if jobs else []
    parts = [None] * len(tracks)
    best = [float("inf")] * len(tracks)
    per_slot = {}
    for s, (d, g) in zip(owners, scores):
        per_slot.setdefault(s, []).append((d, g))
    for s, found in enumerate(candidates):
        for c, (d, g) in zip(found, per_slot.get(s, [])):
            if d < best[s]:
                best[s] = d
                parts[s] = {**slots[s], "document": c["document"], "gain": g, "family": prepare.NAMES[c["type_id"]],
                            "track_distance": d}
    return parts


# --------------------------------------------------------------- evaluation


def oracle_slots(scene):
    """True L2 atoms as slots: notes, the instrument's time FX, the atom's own stem comes from render."""
    slots = []
    for atom in mixscene.atoms_of(scene):
        inst = scene["instruments"][atom["instrument"]]
        slots.append({"notes": atom["notes"], "time_fx": [f for f in inst["insert_fx"] if f["type"] in TIME_FX],
                      "note_echo": None})
    return slots


def evaluate_scene(encoder, model, corpus, index, pool):
    from training.amo.corpus import mixtargets as T, mixtrain

    scene = corpus["scenes"][index]
    stems, mix, scales = mixscene.render_scene(scene)
    target = resynth.log_spectra(mix)
    row = {"index": index, "L0": int(max(scene["ladder"]["L0"]) + 1), "atoms": len(stems)}

    # Ceiling: true atom stems (pre-bus, on the mix scale) and true notes.
    slots = oracle_slots(scene)
    tracks = [s * scales["post_bus_scale"] for s in stems]
    parts = choose_parts(encoder, tracks, slots, pool)
    row["oracle_stems"] = {"D": mix_distance(assemble_scene(parts, scene["bus_fx"], scene["seconds"]), target),
                           "track_D": [p["track_distance"] if p else None for p in parts]}

    # Real pipeline: auto count, mask separation with the predicted stems, predicted notes.
    features, frames = corpus["mix"][index].astype(np.float32), int(corpus["frames"][index])
    out = mixtrain.forward(model, features, frames, 0)
    k = int(out["count"][0].argmax()) + 1
    units, bus_present, bus_values = mixtrain.slot_units(out, frames, k)
    stems_log = np.stack([u["stem"][:mixscene.MAX_STEM_FRAMES] for u in units])
    decoded = T.decode_scene(units, bus_present, bus_values, scene["seconds"])
    row["network_only"] = {"K": len(decoded["instruments"]), "D": mix_distance(decoded, target)}
    instruments = [T.decode_unit(u) for u in units]
    tracks = separate(mix, stems_log)
    slots = [{"notes": i["notes"], "time_fx": [f for f in i["insert_fx"] if f["type"] in TIME_FX],
              "note_echo": i["note_echo"]} for i in instruments]
    parts = choose_parts(encoder, tracks, slots, pool)
    scene_two = assemble_scene(parts, T.decode_bus(bus_present, bus_values), scene["seconds"])
    row["two_stage"] = {"K": len(scene_two["instruments"]), "D": mix_distance(scene_two, target),
                        "track_D": [p["track_distance"] if p else None for p in parts]}
    # Control: the network's own constructions, gains refitted on the same separated tracks.
    own = []
    for inst, track in zip(instruments, tracks):
        if not inst["notes"]:
            own.append(None)
            continue
        doc = mixscene.preset_document(inst)  # layers + its whole insert chain
        d, g = score_candidate((doc, inst["notes"], [], track))
        own.append({"notes": inst["notes"], "time_fx": [], "note_echo": inst["note_echo"], "document": doc,
                    "gain": g, "track_distance": d})
    row["network_refit"] = {"D": mix_distance(assemble_scene(own, T.decode_bus(bus_present, bus_values),
                                                             scene["seconds"]), target)}
    return row


def summarise(rows):
    out = {"scenes": len(rows)}
    for key in ("oracle_stems", "network_only", "network_refit", "two_stage"):
        d = np.asarray([r[key]["D"] for r in rows], float)
        out[key] = {"median_D": float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else None,
                    "failed": int((~np.isfinite(d)).sum())}
        tracks = [t for r in rows for t in r[key].get("track_D", []) if t is not None and np.isfinite(t)]
        if tracks:
            out[key]["median_track_D"] = float(np.median(tracks))
    return out


def evaluate(args):
    from training.amo.corpus import mixtrain, resynth_wav

    encoder = resynth_wav.Encoder(args.encoder)
    model = mixtrain.load_model(args.model)
    corpus = mixtrain.load_corpus(args.corpus)
    indices = mixtrain.split_indices(corpus, args.split)[:args.scenes]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index in indices:
            try:
                rows.append(evaluate_scene(encoder, model, corpus, index, pool))
            except Exception as error:  # noqa: BLE001 - one bad scene must not end the run
                print(f"scene {index}: {error!r}", flush=True)
                continue
            r = rows[-1]
            print(f"scene {index}: oracle {r['oracle_stems']['D']:.3f} two-stage {r['two_stage']['D']:.3f} "
                  f"network {r['network_only']['D']:.3f} refit {r['network_refit']['D']:.3f}", flush=True)
            if len(rows) % 16 == 0:
                Path(args.out).write_text(json.dumps({"summary": summarise(rows), "rows": rows}, indent=1))
    result = {"summary": summarise(rows), "rows": rows}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=2))
    return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("corpus")
    e.add_argument("model")
    e.add_argument("encoder")
    e.add_argument("--split", default="test")
    e.add_argument("--scenes", type=int, default=64)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--out", default="mix-stage2.json")
    return parser.parse_args(argv)


def main(argv=None):
    return evaluate(parse(argv))


if __name__ == "__main__":
    raise SystemExit(main())
