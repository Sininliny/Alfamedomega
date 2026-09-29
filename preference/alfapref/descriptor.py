"""Timbre descriptor (design.md s2): log-mel statistics, independent of the model and the tier.

The log-mel front end matches `logmel()` in training/alfamedomega_train.py (48 kHz, n_fft 2048,
hop 512, 64 mel bands from 20 Hz to 16 kHz, centred frames with reflect padding).
"""
import math

import numpy as np

DESCRIPTOR_ID = "logmel-stats-v1"
SR = 48000
N_FFT = 2048
HOP = 512
N_MELS = 64
N_SCALARS = 6
DIM = 2 * N_MELS + N_SCALARS
MAX_SECONDS = 10.0          # only the start of long files is analysed
ACTIVE_DB = 60.0            # frames more than 60 dB below the loudest frame are ignored
FLOOR_DB = 80.0             # mel values more than 80 dB below the clip's peak are clamped

_FB = None


def mel_filterbank():
    global _FB
    if _FB is None:
        hz = lambda m: 700 * (10 ** (m / 2595) - 1)
        mel = lambda f: 2595 * math.log10(1 + f / 700)
        pts = np.array([hz(m) for m in np.linspace(mel(20), mel(16000), N_MELS + 2)])
        fb = np.fft.rfftfreq(N_FFT, 1 / SR)
        W = np.zeros((N_MELS, fb.shape[0]))
        for i in range(N_MELS):
            lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
            W[i] = np.clip(np.minimum((fb - lo) / (c - lo), (hi - fb) / (hi - c)), 0, None)
        _FB = W
    return _FB


def resample(x, sr):
    """FFT resampling to 48 kHz (no SciPy needed). Good enough for summary statistics."""
    if sr == SR:
        return x
    n = int(round(len(x) * SR / sr))
    X = np.fft.rfft(x)
    m = n // 2 + 1
    Y = np.zeros(m, dtype=complex)
    k = min(m, X.shape[0])
    Y[:k] = X[:k]
    return np.fft.irfft(Y, n) * (n / len(x))


def logmel(x):
    """[N_MELS, frames] natural-log mel magnitudes of 48 kHz mono audio."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= N_FFT // 2:
        x = np.pad(x, (0, N_FFT // 2 + 1 - len(x)))
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    win = np.hanning(N_FFT + 1)[:-1]                     # periodic Hann, as torch.hann_window
    X = np.abs(np.fft.rfft(x[idx] * win, axis=1)).T
    return np.log(mel_filterbank() @ X + 1e-5)


def descriptor(x, sr=SR):
    """134-dim descriptor: gain-invariant, loudness-normalised log-mel mean (64) and std (64), then centroid,
    flatness, log attack time, decay slope, spectral flux, log active duration."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x[: int(MAX_SECONDS * sr)]
    x = resample(x, sr)
    L = logmel(x)                                        # [64, T]
    L = np.maximum(L, L.max() - FLOOR_DB * math.log(10) / 20)   # relative floor: gain-invariant
    e = np.log(np.exp(L).sum(0) + 1e-9)                  # per-frame log energy
    floor = e.max() - ACTIVE_DB * math.log(10) / 20
    act = e > floor
    if act.sum() < 2:
        act[:] = True
    La = L[:, act] - e[act].mean()
    mean, std = La.mean(1), La.std(1)
    p = np.exp(L[:, act]).mean(1)
    p = p / (p.sum() + 1e-12)
    centroid = float((np.arange(N_MELS) * p).sum() / (N_MELS - 1))
    flat = float(np.exp(np.log(p + 1e-12).mean()) / (p.mean() + 1e-12))
    fr = HOP / SR
    first = int(np.argmax(act))
    peak = int(np.argmax(e))
    attack = math.log(max(peak - first, 0) * fr + 1e-3)
    tail = e[peak:][act[peak:]]
    if len(tail) > 2:
        t = np.arange(len(tail)) * fr
        decay = float(np.polyfit(t, tail, 1)[0]) / 10.0  # nats per 100 ms
    else:
        decay = 0.0
    flux = float(np.abs(np.diff(La, axis=1)).mean()) if La.shape[1] > 1 else 0.0
    dur = math.log(act.sum() * fr + 1e-3)
    scal = np.array([centroid, flat, attack, np.clip(decay, -5, 5), flux, dur])
    return np.concatenate([mean, std, scal]).astype(np.float32)
