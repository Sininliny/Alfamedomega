#!/usr/bin/env python3
"""Listen to the corpus encoder: guess a sound's construction and re-render it with our DSP.

Two modes:

- ``demo``: draws a random construction from one of our instrument families, renders it,
  hides everything but the audio and the performance, lets the encoder guess the family
  and every parameter, and renders the guess. This is exactly what the model was trained
  for, so it shows what the model can do.
- ``wav``: the same for your own WAV files (a single held note works best). The model has
  only ever heard our own 24 instrument structures, so real recordings come out as the
  nearest sound those structures can make, not a copy.

Every result folder holds ``original.wav`` (what the model heard), ``resynth.wav`` (our DSP
playing the guess), ``construction.json`` (the guessed AMO program, editable) and
``guess.json`` (top families with probabilities and the spectral distance).

    python -m training.amo.corpus.resynth_wav demo MODEL OUT --count 8
    python -m training.amo.corpus.resynth_wav wav MODEL OUT input1.wav input2.wav --note auto
"""

import argparse
import json
import os
from pathlib import Path
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus import prepare, resynth  # noqa: E402
from training.amo.corpus.train import (  # noqa: E402
    HEADS, MODEL_SCHEMA, VALUE_TENSORS, Standardizer, TargetStats, build_model)

INSTRUMENT_TYPES = 24          # families 24.. are FX and need a paired input signal
SAMPLE_RATE = 24000
MIN_SECONDS, MAX_SECONDS = 1.2, 2.4
NOTE_RANGE = (34, 69)           # MIDI notes seen in training


# ------------------------------------------------------------------------- model


class Encoder:
    def __init__(self, path):
        import torch

        self.torch = torch
        package = torch.load(path, map_location="cpu", weights_only=False)
        if package.get("schema") != MODEL_SCHEMA:
            raise SystemExit(f"{path} is not an {MODEL_SCHEMA} package")
        self.stats = {"X_audio": Standardizer(**package["stats"]["X_audio"]),
                      "conditions": Standardizer(**package["stats"]["conditions"])}
        for name in VALUE_TENSORS:
            self.stats[name] = TargetStats(**package["stats"][name])
        self.widths = package["widths"]
        self.types = package["types"]
        operations = max(package["operation_vocabulary"].values())
        self.model = build_model(package["config"], self.widths, self.types, operations)
        self.model.load_state_dict(package["model"])
        self.model.eval()
        # Mean log RMS of the training renders (last of the output-feature block).
        self.typical_log_rms = float(self.stats["X_audio"].mean[32 * 35 + 1])

    def guess(self, x, conditions, instruments_only=True, exclude=()):
        torch = self.torch
        xs = torch.from_numpy(self.stats["X_audio"](x[None]))
        cs = torch.from_numpy(self.stats["conditions"](conditions[None]))
        with torch.no_grad():
            h = self.model.tokens(xs, cs)
            logits = self.model.type_head(h)
            if instruments_only:
                logits[:, INSTRUMENT_TYPES:] = -1e9
            for t in exclude:
                logits[:, t] = -1e9
            probabilities = torch.softmax(logits.float(), 1)[0].numpy()
            type_id = int(probabilities.argmax())
            values = self.model(xs, cs, torch.tensor([type_id]))["values"][0, list(HEADS).index("instance")]
        z = values.float().numpy().astype(np.float64)
        decoded, offset = {}, 0
        for name in VALUE_TENSORS:
            width = self.widths[name]
            decoded[name] = self.stats[name].decode(np.array([type_id]), z[None, offset:offset + width])[0]
            offset += width
        return type_id, decoded, probabilities


# ------------------------------------------------------------------------ audio


def read_wav(path):
    """PCM 16/24/32-bit WAV as float64 [frames, 2] and its sample rate."""
    with wave.open(str(path), "rb") as w:
        channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        data = w.readframes(frames)
    if width == 2:
        audio = np.frombuffer(data, "<i2").astype(np.float64) / 32768
    elif width == 3:
        raw = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        codes = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        audio = np.where(codes >= 1 << 23, codes - (1 << 24), codes).astype(np.float64) / 8388608
    elif width == 4:
        audio = np.frombuffer(data, "<i4").astype(np.float64) / 2147483648
    else:
        raise SystemExit(f"{path}: only 16/24/32-bit PCM WAV is supported (export it as PCM WAV)")
    audio = audio.reshape(-1, channels)
    audio = np.repeat(audio, 2, axis=1) if channels == 1 else audio[:, :2]
    return audio, rate


def resample(audio, rate, target=SAMPLE_RATE):
    if rate == target:
        return audio
    if rate > target:  # crude anti-alias: moving average over the decimation ratio
        k = max(1, int(round(rate / target)))
        kernel = np.ones(k) / k
        audio = np.column_stack([np.convolve(c, kernel, "same") for c in audio.T])
    t_old = np.arange(len(audio)) / rate
    t_new = np.arange(int(len(audio) * target / rate)) / target
    return np.column_stack([np.interp(t_new, t_old, c) for c in audio.T])


def estimate_pitch(audio, sr):
    """MIDI note (float) of the first loud half second by autocorrelation, or None."""
    mono = audio.mean(1)
    start = int(np.argmax(np.abs(mono) > 0.1 * np.abs(mono).max()))
    segment = mono[start + int(0.03 * sr): start + int(0.5 * sr)]
    if len(segment) < sr // 10:
        return None
    segment = segment - segment.mean()
    spectrum = np.fft.rfft(segment * np.hanning(len(segment)), n=2 * len(segment))
    corr = np.fft.irfft(np.abs(spectrum) ** 2)[:len(segment)]
    low, high = int(sr / 1000), int(sr / 25)   # 25 Hz floor: sub basses sit at C1 (32.7 Hz)
    if high >= len(corr) or corr[0] <= 0:
        return None
    window = corr[low:high]
    # Only interior local maxima are periods; the window's left edge is just the decay from
    # lag 0 and would read as ~1 kHz for a sub bass whose period is longer than the window.
    peaks = np.flatnonzero((window[1:-1] >= window[:-2]) & (window[1:-1] >= window[2:])) + 1
    if not len(peaks):
        return None
    best = float(window[peaks].max())
    if best < 0.3 * corr[0]:
        return None
    # Peaks at whole multiples of the period are nearly as tall as the first one, so the
    # highest peak often lands on a subharmonic. Take the earliest strong one.
    lag = low + int(peaks[window[peaks] >= 0.85 * best][0])
    return 69 + 12 * np.log2(sr / lag / 440)


def estimate_note_off(audio, sr):
    """Time where the envelope first stays below a third of its peak after the peak."""
    env = np.sqrt(np.convolve(audio.mean(1) ** 2, np.ones(int(0.02 * sr)) / int(0.02 * sr), "same"))
    peak = int(np.argmax(env))
    below = np.flatnonzero(env[peak:] < env[peak] / 3)
    return (peak + below[0]) / sr if below.size else len(audio) / sr * 0.5


def performance(note, cents, velocity, note_off, frames):
    return {"version": "amo.performance-plan/2", "sample_rate": SAMPLE_RATE, "frames": frames,
            "midi_note": int(note), "tuning_cents": float(cents), "velocity": float(velocity),
            "note_off_seconds": float(note_off), "events": "held", "vibrato_rate": 5.0,
            "vibrato_cents": 17.0, "input_seed": 1, "input_kind": "none", "role": "instrument"}


def conditions_for(perf):
    return np.asarray([perf["sample_rate"] / 24000.0, perf["frames"] / perf["sample_rate"], perf["velocity"],
                       perf["note_off_seconds"], perf["midi_note"] / 127.0, perf["tuning_cents"] / 100.0,
                       perf["vibrato_rate"] / 10.0, perf["vibrato_cents"] / 100.0,
                       ["held", "staccato", "two_notes", "vibrato"].index(perf["events"]) / 3.0, 0.0],
                      np.float32)


def model_input(audio, perf):
    peak = float(np.abs(audio).max())
    if peak >= 0.95:  # the corpus never clips; keep the input in the same range
        audio = audio * (0.9 / peak)
    decoded, _ = prepare.write_wav(None, audio, perf["sample_rate"])
    x = np.concatenate([prepare.features(decoded, perf["sample_rate"]), np.zeros(32 * 35 + 2, np.float32)])
    return x.astype(np.float32), conditions_for(perf), decoded


def save(folder, original, resynthesised, doc, info, sr):
    folder.mkdir(parents=True, exist_ok=True)
    for name, audio in (("original", original), ("resynth", resynthesised)):
        if audio is None:
            continue
        peak = float(np.abs(audio).max())
        if peak > 1e-6:
            prepare.write_wav(folder / f"{name}.wav", audio * (0.5 / peak), sr)
    (folder / "construction.json").write_text(json.dumps(doc, indent=2))
    (folder / "guess.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def describe(encoder, type_id, probabilities, original, rendered):
    names = prepare.NAMES
    top = np.argsort(-probabilities)[:3]
    distance = resynth.distance(None if rendered is None else resynth.log_spectra(rendered),
                                resynth.log_spectra(original))
    return {"guessed_family": names[type_id],
            "top_families": [{"family": names[t], "probability": round(float(probabilities[t]), 3)} for t in top],
            "spectral_distance": round(distance, 3) if np.isfinite(distance) else None,
            "distance_note": "multi-resolution log-STFT distance; test-set median 0.18, family-average baseline 0.51"}


# ------------------------------------------------------------------------- modes


def run_demo(encoder, out, count, seed):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(count):
        t = int(rng.integers(0, INSTRUMENT_TYPES))
        for _ in range(8):
            doc, _, _, _ = prepare.make_document(prepare.NAMES[t], rng.random(64))
            perf = prepare.performance_for(t, int(rng.integers(0, 1 << 20)), int(rng.integers(0, 4)))
            target = resynth.render(doc, perf)
            if target is not None and 1e-6 < np.sqrt(np.mean(target ** 2)) and np.abs(target).max() < 0.95:
                break
        else:
            continue
        x, conditions, decoded = model_input(target, perf)
        conditions[1] = perf["frames"] / perf["sample_rate"]
        guessed, values, probabilities = encoder.guess(x, conditions)
        guess_doc = resynth.assemble(guessed, values)
        rendered = resynth.render(guess_doc, perf)
        info = {"true_family": prepare.NAMES[t], **describe(encoder, guessed, probabilities, decoded, rendered),
                "performance": perf}
        folder = out / f"demo-{i + 1:02d}-{prepare.NAMES[t]}"
        save(folder, decoded, rendered, guess_doc, info, perf["sample_rate"])
        results.append(info)
        mark = "OK " if guessed == t else "   "
        print(f"{mark}{folder.name}: guessed {info['guessed_family']}, distance {info['spectral_distance']}", flush=True)
    return results


def guess_wav(encoder, path, note_arg="auto", velocity=0.7, exclude=()):
    """Guess one WAV: returns construction, original, resynth audio and an info dict."""
    audio, rate = read_wav(path)
    audio = resample(audio, rate)
    threshold = 0.01 * np.abs(audio).max()
    loud = np.flatnonzero(np.abs(audio).max(1) > threshold)
    if loud.size == 0:
        raise ValueError("the file is silent")
    audio = audio[max(0, loud[0] - int(0.005 * SAMPLE_RATE)):]
    frames = int(np.clip(len(audio), MIN_SECONDS * SAMPLE_RATE, MAX_SECONDS * SAMPLE_RATE))
    audio = np.pad(audio, ((0, max(0, frames - len(audio))), (0, 0)))[:frames]
    warnings = []
    if note_arg == "auto":
        pitch = estimate_pitch(audio, SAMPLE_RATE)
        if pitch is None:
            pitch = 57.0
            warnings.append("no clear pitch found; assumed A3 (57). Pass --note to set it.")
    else:
        pitch = float(note_arg)
    note = int(round(pitch))
    cents = float(np.clip((pitch - note) * 100, -12, 12))
    if not NOTE_RANGE[0] <= note <= NOTE_RANGE[1]:
        warnings.append(f"note {note} is outside the trained range {NOTE_RANGE}; expect a weaker guess")
    note_off = float(np.clip(estimate_note_off(audio, SAMPLE_RATE), 0.32, 0.62))
    perf = performance(note, cents, velocity, note_off, frames)
    # Level is a model input (log RMS, last output feature): bring the file to the training median,
    # since a user's sample level says nothing about the construction.
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms > 1e-9:
        audio = audio * min(np.exp(encoder.typical_log_rms) / rms, 0.9 / float(np.abs(audio).max()))
    x, conditions, decoded = model_input(audio, perf)
    guessed, values, probabilities = encoder.guess(x, conditions, exclude=exclude)
    construction = resynth.assemble(guessed, values)
    rendered = resynth.render(construction, perf)
    info = {"input": str(path), **describe(encoder, guessed, probabilities, decoded, rendered),
            "assumed_performance": perf, "warnings": warnings}
    return {"construction": construction, "original": decoded, "resynth": rendered, "info": info}


def run_wav(encoder, out, paths, note_arg, velocity):
    results = []
    for path in paths:
        try:
            guess = guess_wav(encoder, path, note_arg, velocity)
        except ValueError as error:
            print(f"skip {path}: {error}")
            continue
        info = guess["info"]
        folder = out / Path(path).stem
        if folder.exists():
            folder = out / f"{Path(path).stem}-{len(results) + 1}"
        save(folder, guess["original"], guess["resynth"], guess["construction"], info, SAMPLE_RATE)
        results.append(info)
        print(f"{folder.name}: note {info['assumed_performance']['midi_note']}, guessed {info['guessed_family']},"
              f" distance {info['spectral_distance']}" + "".join(f"\n  warning: {w}" for w in info["warnings"]),
              flush=True)
    return results


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    demo = modes.add_parser("demo", help="random sounds from our own structures")
    demo.add_argument("model", type=Path)
    demo.add_argument("out", type=Path)
    demo.add_argument("--count", type=int, default=8)
    demo.add_argument("--seed", type=int, default=None)
    wav = modes.add_parser("wav", help="your own WAV files (one held note each works best)")
    wav.add_argument("model", type=Path)
    wav.add_argument("out", type=Path)
    wav.add_argument("inputs", nargs="+", type=Path)
    wav.add_argument("--note", default="auto", help="MIDI note of the sound (60 = C4), or auto")
    wav.add_argument("--velocity", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for torch (fewer = quieter)")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(max(1, args.threads))
    encoder = Encoder(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "demo":
        results = run_demo(encoder, args.out, args.count, args.seed)
    else:
        results = run_wav(encoder, args.out, args.inputs, args.note, args.velocity)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"results in {args.out}")


if __name__ == "__main__":
    main()
