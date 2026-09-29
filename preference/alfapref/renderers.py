"""Preview renderers: patch -> audio, used to embed patches (design.md s6.2).

A renderer is any callable `render(patch, pitch=60, velocity=0.8, seconds=1.0) -> np.ndarray`
returning 48 kHz mono audio. In the plugin it is the engine itself on the Lite tier. In Python the
real one is `render_patch` in training/alfamedomega_train.py (teacher programs, or a checkpoint;
needs torch). `SketchRenderer` is a small NumPy stand-in whose knobs have audible, monotone
effects, used by the tests and as the default.
"""
import hashlib

import numpy as np

SR = 48000
N_H = 24


class SketchRenderer:
    """Not Alfamedomega's sound. Knobs: 0 brightness, 1 even/odd balance, 2 inharmonicity,
    3 attack, 4 decay, 5 sustain, 6 noise level, 7 noise brightness, 8 vibrato, 9 FM index,
    10 FM ratio. The method shifts a few of them so methods sound different."""

    def __init__(self, sr=SR):
        self.sr = sr
        self.cache = {}

    def __call__(self, patch, pitch=60, velocity=0.8, seconds=1.0):
        key = (repr(patch["method"]), repr(patch.get("pair")), tuple(np.round(patch["knobs"], 4)), pitch,
               velocity, seconds)
        if key in self.cache:
            return self.cache[key]
        k = np.asarray(patch["knobs"], dtype=np.float64)
        m = patch["method"]
        sr, n = self.sr, int(seconds * self.sr)
        t = np.arange(n) / sr
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        slope = 0.4 + 2.6 * (1 - k[0])
        inh = 1e-4 * 10 ** (k[2] * 2.5) * (10 if m == "physical" else 1)
        h = np.arange(1, N_H + 1)[:, None]
        amp = h ** -slope * np.where(h % 2 == 0, 0.1 + 0.9 * k[1], 1.0)
        freq = f0 * h * np.sqrt(1 + inh * h * h)
        amp = amp * (freq < sr / 2 * 0.95)
        # phase of f * (1 + v sin(w t)), integrated in closed form
        v, w = 0.01 * k[8], 2 * np.pi * 5.5
        phase = 2 * np.pi * freq * (t + v * (1 - np.cos(w * t)) / w)
        if m == "fm" or k[9] > 0.8:
            ratio = 1 + np.floor(k[10] * 4)
            phase = phase + 4 * k[9] * np.sin(2 * np.pi * f0 * ratio * t)
        tone = (amp * np.sin(phase)).sum(0) / amp.sum()
        rng = np.random.default_rng(int(hashlib.sha1(repr(key).encode()).hexdigest()[:8], 16))
        noise = rng.standard_normal(n)
        a = 0.3 + 0.69 * k[7]
        noise = _one_pole(noise, 1 - a)
        nlev = 0.5 * k[6] ** 2 * (2 if m in ("granular", "sampling", "spectral") else 1)
        att = 1e-3 * 10 ** (k[3] * 2.5)
        dec = 0.1 * 10 ** (k[4] * 1.5)
        sus = 0.0 if m == "physical" else k[5]
        env = (1 - np.exp(-t / att)) * (sus + (1 - sus) * np.exp(-t / dec))
        y = velocity * env * (tone + nlev * noise / (np.abs(noise).max() + 1e-9))
        y = y.astype(np.float32)
        if len(self.cache) < 4096:
            self.cache[key] = y
        return y


def _one_pole(x, c):
    """y[n] = (1 - c) x[n] + c y[n-1], vectorised in blocks with a closed form per block."""
    y = np.empty_like(x)
    prev = 0.0
    B = 256
    powers = c ** np.arange(1, B + 1)
    for s in range(0, len(x), B):
        seg = (1 - c) * x[s:s + B]
        L = len(seg)
        # y_j = sum_i c^(j-i) seg_i + c^(j+1) prev
        conv = np.convolve(seg, c ** np.arange(L))[:L]
        y[s:s + L] = conv + powers[:L] * prev
        prev = y[s + L - 1]
    return y
