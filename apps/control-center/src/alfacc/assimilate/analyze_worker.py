#!/usr/bin/env python3
"""Turn rendered notes into timbre features. Runs in the training Python (needs NumPy only).

    python analyze_worker.py --renders renders.json --notes notes.json --out features.jsonl
                             [--descriptor preference/alfapref/descriptor.py] [--delete-audio]

Standalone on purpose: the control center itself stays standard-library only, and a packaged
control center runs this file with the Python from Settings, like the preference learner.
For each render it writes one JSON line: level, attack, release, pitch tracking, spectral
centroid and flatness, and, when preference/ is available, its 134-number timbre descriptor
(logmel-stats-v1), so harvested presets live in the same space as the user's libraries.
"""
import argparse
import importlib.util
import json
import os
import struct
import sys

import numpy as np

FRAME_S = 0.01
SILENT_DB = -60.0


def read_wav(path):
    """Mono float64 samples and the sample rate. PCM 16/24/32 and float 32/64."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            if fmt[0] == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE: real tag in the GUID
                fmt = (struct.unpack("<H", body[24:26])[0],) + fmt[1:]
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError(f"{path} has no fmt or data chunk")
    tag, channels, sr, _, _, bits = fmt
    if tag == 3:
        x = np.frombuffer(pcm, dtype="<f4" if bits == 32 else "<f8").astype(np.float64)
    elif bits == 16:
        x = np.frombuffer(pcm, dtype="<i2") / 32768.0
    elif bits == 24:
        b = np.frombuffer(pcm, dtype=np.uint8)[: len(pcm) // 3 * 3].reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v & 0x800000, v - 0x1000000, v) / 8388608.0
    elif bits == 32:
        x = np.frombuffer(pcm, dtype="<i4") / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported {bits}-bit audio")
    x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x, sr


def db(v):
    return float(20 * np.log10(max(float(v), 1e-12)))


def envelope(x, sr):
    n = max(1, int(FRAME_S * sr))
    frames = len(x) // n
    if frames == 0:
        return np.array([np.sqrt(np.mean(x ** 2)) if len(x) else 0.0])
    return np.sqrt(np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1))


def pitch(x, sr, lo=20.0, hi=5000.0):
    """(f0 Hz, confidence 0..1) by normalised autocorrelation, preferring the shortest strong lag."""
    n = min(len(x), 8192)
    if n < 512:
        return None, 0.0
    seg = x[:n] * np.hanning(n)
    spec = np.fft.rfft(seg, 2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
    if ac[0] <= 1e-12:
        return None, 0.0
    ac = ac / ac[0]
    # Normalise for the shrinking overlap so long lags aren't penalised.
    ac = ac / np.maximum(1e-9, (n - np.arange(n)) / n)
    lmin, lmax = max(2, int(sr / hi)), min(n - 2, int(sr / lo))
    if lmax <= lmin:
        return None, 0.0
    region = ac[lmin:lmax]
    best = float(region.max())
    if best < 0.3:
        return None, best
    # First local peak that reaches 90% of the best one: avoids picking a sub-octave.
    for i in range(1, len(region) - 1):
        if region[i] >= 0.9 * best and region[i] >= region[i - 1] and region[i] >= region[i + 1]:
            lag = lmin + i
            a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
            shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
            return sr / (lag + shift), min(1.0, float(region[i]))
    return None, best


def analyse(x, sr, note):
    off = int(round(note["dur_s"] * sr))
    held = x[:max(1, min(off, len(x)))]
    env = envelope(x, sr)
    env_held = env[: max(1, int(note["dur_s"] / FRAME_S))]
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    out = {"peak_db": round(db(peak), 2), "seconds": round(len(x) / sr, 3)}
    if db(peak) < SILENT_DB:
        out.update(audible=False, pitched=False)
        return out
    top = float(env_held.max()) if len(env_held) else float(env.max())
    active = env_held[env_held > top * 10 ** (SILENT_DB / 20)]
    out["rms_db"] = round(db(np.sqrt(np.mean(active ** 2))) if len(active) else db(0), 2)
    reach = np.nonzero(env_held >= 0.9 * top)[0]
    out["attack_ms"] = round(float(reach[0]) * FRAME_S * 1000, 1) if len(reach) else None
    # Release: from note-off until 60 dB below the level at note-off.
    k = min(len(env) - 1, int(note["dur_s"] / FRAME_S))
    level = float(env[max(0, k - 1)])
    after = env[k:]
    below = np.nonzero(after < level * 1e-3)[0] if level > 0 else np.array([0])
    out["release_ms"] = round(float(below[0]) * FRAME_S * 1000, 1) if len(below) else None
    out["release_truncated"] = not len(below)
    # Pitch from the sustained part, after the attack.
    start = int(min(len(held) // 3, max(0.05 * sr, (reach[0] if len(reach) else 0) * FRAME_S * sr)))
    f0, conf = pitch(held[start:], sr)
    expected = 440.0 * 2 ** ((note["midi"] - 69) / 12)
    out["expected_hz"] = round(expected, 3)
    out["f0_confidence"] = round(conf, 3)
    if f0:
        cents = 1200 * np.log2(f0 / expected)
        octave = int(np.round(cents / 1200))
        out.update(f0_hz=round(f0, 3), cents=round(float(cents - 1200 * octave), 1), octave=octave)
        out["pitched"] = bool(conf >= 0.6 and abs(cents - 1200 * octave) <= 50)
    else:
        out["pitched"] = False
    # Spectral shape of the held part.
    seg = held[start:start + 16384] if len(held) - start > 1024 else held
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-12
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    out["centroid_hz"] = round(float(np.sum(freqs * mag) / np.sum(mag)), 1)
    out["flatness"] = round(float(np.exp(np.mean(np.log(mag))) / np.mean(mag)), 4)
    out["audible"] = True
    return out


def load_descriptor(path):
    if not path:
        return None, None
    spec = importlib.util.spec_from_file_location("_alfapref_descriptor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.descriptor, getattr(mod, "DESCRIPTOR_ID", "unknown")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--descriptor", default="")
    ap.add_argument("--delete-audio", action="store_true")
    a = ap.parse_args(argv)
    renders = json.load(open(a.renders, encoding="utf-8"))["renders"]
    notes = json.load(open(a.notes, encoding="utf-8"))
    describe, descriptor_id = load_descriptor(a.descriptor)
    with open(a.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(renders, 1):
            note = notes[r["note"]]
            rec = {"preset": r["preset"], "note": r["note"], "midi": note["midi"],
                   "velocity": note["velocity"], "dur_s": note["dur_s"]}
            try:
                x, sr = read_wav(r["wav"])
                rec.update(analyse(x, sr, note))
                if describe is not None and rec.get("audible"):
                    rec["descriptor_id"] = descriptor_id
                    rec["descriptor"] = [round(float(v), 5) for v in describe(x, sr)]
            except (OSError, ValueError) as e:
                rec["error"] = str(e)
            f.write(json.dumps(rec) + "\n")
            if a.delete_audio:
                try:
                    os.remove(r["wav"])
                except OSError:
                    pass
            if i % 20 == 0 or i == len(renders):
                print(f"progress {i}/{len(renders)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
