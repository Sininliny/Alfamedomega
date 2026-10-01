"""Explicit reference features for WAV experiments; not universal normalization.

Pitch-relative spectra and relative envelopes remove global pitch/gain for
controlled voiced renderings approximately. Unknown recording coloration is not
identifiable from arbitrary WAV alone: supply known calibration or retain that
uncertainty. This profile does not certify perceptual invariance.
"""
from dataclasses import dataclass
from pathlib import Path
import wave
import numpy as np


@dataclass(frozen=True)
class AnalysisProfile:
    frames: int
    harmonic_ratios: tuple[float, ...]
    minimum_fundamental_hz: float
    maximum_fundamental_hz: float
    recording_policy: str

    def validate(self):
        if self.frames <= 0 or not self.harmonic_ratios or any(not np.isfinite(x) or x <= 0 for x in self.harmonic_ratios):
            raise ValueError("Explicit frame count and positive spectral ratios required")
        if not 0 < self.minimum_fundamental_hz < self.maximum_fundamental_hz:
            raise ValueError("Explicit fundamental search range required")
        if self.recording_policy not in ("controlled_render", "known_calibration"):
            raise ValueError("Unknown recording conditions need an approved normalization policy")


def read_wav(path: Path, *, available_ram_bytes=None):
    from .hardware import detect
    from .contracts import ModelBudget
    available = detect(gpu_device=None).available_ram_bytes
    if available_ram_bytes is not None:
        available = min(available, available_ram_bytes)
    with wave.open(str(path), "rb") as wav:
        width = wav.getsampwidth()
        channels = wav.getnchannels()
        rate = wav.getframerate()
        # PCM decode, float audio, windows/FFT workspace and Python label storage.
        # Reject before reading the complete waveform; no hidden chunking claim.
        estimate = wav.getnframes() * channels * (width + 96)
        ModelBudget(estimate, available, 0, 0)
        data = wav.readframes(wav.getnframes())
    if width == 1:
        values = (np.frombuffer(data, np.uint8).astype(np.float64) - 128) / 128
    elif width in (2, 4):
        values = np.frombuffer(data, "<i2" if width == 2 else "<i4").astype(np.float64) / (2 ** (8 * width - 1))
    elif width == 3:
        raw = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        packed = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        signed = (packed ^ 0x800000) - 0x800000
        values = signed.astype(np.float64) / 8388608
    else:
        raise ValueError("Unsupported PCM width")
    return values.reshape(-1, channels), rate


def extract(audio, sample_rate: float, profile: AnalysisProfile, *, calibration=None):
    profile.validate()
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or not audio.size or not np.isfinite(audio).all() or not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Finite nonempty audio and positive rate required")
    if profile.recording_policy == "known_calibration":
        if calibration is None:
            raise ValueError("Known calibration profile requires an explicit correction callable")
        audio = np.asarray(calibration(audio, sample_rate), dtype=np.float64)
        if audio.ndim != 2 or not audio.size or not np.isfinite(audio).all():
            raise ValueError("Calibration produced invalid audio")
    # Channel energy is retained separately; no phase-cancelling mono average.
    channel_rms = np.sqrt(np.mean(audio ** 2, axis=0))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms <= np.finfo(np.float64).eps:
        raise ValueError("Silent WAV has no identifiable timbral spectrum")
    signal = audio[:, int(np.argmax(channel_rms))]
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    frequencies = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    region = (frequencies >= profile.minimum_fundamental_hz) & (frequencies <= profile.maximum_fundamental_hz)
    if not region.any():
        raise ValueError("WAV duration cannot resolve requested fundamental range")
    # Reference estimator uses strongest component in explicit range, not a claim
    # of fundamental detection for arbitrary missing-fundamental/polyphonic audio.
    fundamental = frequencies[np.flatnonzero(region)[np.argmax(spectrum[region])]]
    envelope = []
    spectral = []
    for frame in np.array_split(signal, profile.frames):
        if not len(frame):
            raise ValueError("Frame count exceeds number of samples")
        envelope.append(float(np.sqrt(np.mean(frame ** 2)) / rms))
        size = max(len(frame), int(sample_rate / fundamental * 8))
        magnitude = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=size))
        bins = np.fft.rfftfreq(size, 1 / sample_rate)
        denominator = float(magnitude.max())
        spectral.extend((np.interp(np.asarray(profile.harmonic_ratios) * fundamental, bins, magnitude,
                                   left=0, right=0) / max(denominator, np.finfo(float).eps)).tolist())
    # Duration retains absolute temporal scale; relative envelope retains ordered
    # articulation and global gain is removed. No time stretch is performed.
    return [len(signal) / sample_rate, *envelope, *spectral, *(channel_rms / rms).tolist()]
