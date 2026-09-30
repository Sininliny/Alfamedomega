#!/usr/bin/env python3
"""Stands in for alfa-render-host in tests: same commands and files, synthetic instrument.

The instrument behaves like render-host/test-synth (AlfaTestSynth): program 0 is a sine on every
key, program 1 a saw that is silent outside C1..C4 (MIDI 24..60), program 2 decaying noise.
FAKE_HOST_CRASH=<preset key> makes that preset crash the host, like a misbehaving plugin.
"""
import json
import os
import sys
import wave

import numpy as np

PLUGIN = {"name": "FakeSynth", "vendor": "Alfamedomega", "format": "VST3", "uid": "VST3-FakeSynth-1",
          "version": "1.0", "category": "Instrument|Synth", "is_instrument": True, "inputs": 0,
          "outputs": 2}
PROGRAMS = ["Sine Pad", "Saw Bass C1-C4", "Noise Hit"]


def voice(program, midi, velocity, dur_s, tail_s, sr):
    n_on, n = int(dur_s * sr), int((dur_s + min(tail_s, 0.5)) * sr)
    t = np.arange(n) / sr
    f = 440.0 * 2 ** ((midi - 69) / 12)
    amp = 0.5 * velocity / 127
    env = np.minimum(1.0, t / 0.01) * np.where(t < dur_s, 1.0, np.exp(-(t - dur_s) / 0.05))
    if program == 0:
        x = np.sin(2 * np.pi * f * t)
    elif program == 1:
        if not 24 <= midi <= 60:
            return np.zeros(n_on)
        x = 2 * ((f * t) % 1.0) - 1
    else:
        x = np.random.default_rng(midi).uniform(-1, 1, n) * np.exp(-t / 0.2)
    return amp * env * x


def write(path, x, sr):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main(argv):
    if argv[0] == "formats":
        print(json.dumps(["VST3"]))
    elif argv[0] == "scan":
        files = [a for a in argv[1:] if not a.startswith("--")]
        if any(os.path.basename(f).startswith("Crash") for f in files):
            os.abort()
        plugins = [{**PLUGIN, "file": f, "programs": PROGRAMS} for f in files]
        print(json.dumps({"plugins": plugins}))
    elif argv[0] == "render":
        job = json.load(open(argv[1]))
        sr, out = job["sample_rate"], job["out_dir"]
        total = len(job["presets"]) * len(job["notes"])
        renders, i = [], 0
        for p in job["presets"]:
            if os.environ.get("FAKE_HOST_CRASH") == p["key"]:
                print("rendering", flush=True)
                os.abort()
            os.makedirs(os.path.join(out, p["key"]), exist_ok=True)
            for note in job["notes"]:
                x = voice(p.get("index", 0), note["midi"], note["velocity"], note["dur_s"], note["tail_s"], sr)
                path = os.path.join(out, p["key"], note["id"] + ".wav")
                write(path, x, sr)
                renders.append({"preset": p["key"], "note": note["id"], "wav": path,
                                "seconds": len(x) / sr, "peak": float(np.max(np.abs(x))) if len(x) else 0.0})
                i += 1
                print(f"progress {i}/{total}", flush=True)
        with open(os.path.join(out, "renders.json"), "w") as f:
            json.dump({"plugin": PLUGIN, "renders": renders}, f)
        print("result " + os.path.join(out, "renders.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
