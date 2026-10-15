"""Perceptual features, spectral reconstruction distance and the offline parameter metric.

NumPy only. Three pieces share one feature map phi(y) of rendered audio y:

* `spectral_distance(y, y_ref)` = ||phi(y) - phi(y_ref)||^2 + w1 * W1. phi holds
  multi-resolution log band energies, a log RMS envelope and the cumulative
  distribution of power over log frequency. None of them depends on phase, and
  the cumulative distribution (plus the explicit W1 term) moves smoothly when
  pitch moves, so a descent on pitch-like parameters points the right way even
  when spectral peaks do not overlap. This replaces sample-domain MSE.
* `Coordinates` maps declared parameters to unconstrained training coordinates u
  ("linear" keeps p, "log" uses log p) so gradient steps are scale-uniform.
* `parameter_metric(...)` measures J = d phi / d u once per rendered example by
  central finite differences and stores its rank-r factor: G = J^T J ~ U diag(S^2) U^T.
  The learner then penalises Delta u^T G Delta u, an exact quadratic in u, so
  directions that do not change the sound cost nothing and audible ones dominate.

Field names in the metric file (U [K, r], S [r], free_mask [K], u_ref [K], Jd [8, K])
match training/amo/wave_features.py.
"""
import math
from pathlib import Path

import numpy as np
from .schema import schema_is

SCHEMA = "amo.perceptual-metric/1"
FEATURE_ID = "amo.phi/1"
RESOLUTIONS = (256, 1024, 4096)
BANDS = 48
ENVELOPE_HOP = 256
GRID_POINTS = 512
SMOOTHING_OCTAVES = 1 / 12
MINIMUM_HZ = 30.0
# Measured on harmonic tones (110 Hz/8 partials, 440 Hz sine, 1 kHz/3 partials), starting
# up to 0.85 octave away: sign descent on pitch converged within 1% from 30 of 30 starts.
# Heavier band weights reintroduce local minima because log band energies saturate
# once partials stop overlapping; the cumulative terms carry the long-range pull.
W1_WEIGHT = 10.0
BAND_WEIGHT = 0.03
CDF_WEIGHT = 1.0
ENERGY_FLOOR = 1e-8
COORDINATE_KINDS = ("linear", "log")


class Coordinates:
    """Per-parameter unconstrained coordinates; absent entries default to linear."""

    def __init__(self, kinds, bounds):
        self.kinds = tuple(kinds)
        self.bounds = np.asarray(bounds, dtype=np.float64).reshape(-1, 2)
        if len(self.kinds) != len(self.bounds) or any(k not in COORDINATE_KINDS for k in self.kinds):
            raise ValueError("One linear/log coordinate kind per bounded parameter required")
        if np.isnan(self.bounds).any() or (self.bounds[:, 0] > self.bounds[:, 1]).any():
            raise ValueError("Ordered parameter bounds required")
        self.log = np.array([k == "log" for k in self.kinds], dtype=bool)
        if not np.isfinite(self.bounds[self.log]).all() or (self.bounds[self.log, 0] <= 0).any():
            raise ValueError("Log coordinates require finite, strictly positive bounds")

    @classmethod
    def from_manifest(cls, manifest, width):
        """Declared kinds and bounds; without them every coordinate is linear and unbounded."""
        kinds = manifest.get("parameter_coordinates") or ["linear"] * width
        bounds = manifest.get("reconstruction_bounds", {}).get("parameters") or [[-math.inf, math.inf]] * width
        if len(kinds) != width or len(bounds) != width:
            raise ValueError("parameter_coordinates and parameter bounds must cover every padded parameter")
        return cls(kinds, bounds)

    @property
    def identity(self):
        return not self.log.any()

    def to_u(self, p):
        p = np.asarray(p, dtype=np.float64)
        return np.where(self.log, np.log(np.maximum(p, np.where(self.log, self.bounds[:, 0], 1.0))), p)

    def to_p(self, u):
        u = np.asarray(u, dtype=np.float64)
        return np.where(self.log, np.exp(np.minimum(u, 700.0)), u)

    def u_bounds(self):
        return self.to_u(self.bounds[:, 0]), self.to_u(self.bounds[:, 1])


def _audio(audio):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Finite nonempty [sample, channel] audio required")
    return audio


_BAND_CACHE = {}


def _band_matrix(size, sample_rate):
    key = (size, float(sample_rate))
    if key not in _BAND_CACHE:
        bins = np.fft.rfftfreq(size, 1 / sample_rate)
        top = sample_rate / 2
        edges = np.geomspace(MINIMUM_HZ, top, BANDS + 2)
        matrix = np.zeros((BANDS, bins.size))
        for band in range(BANDS):
            low, centre, high = edges[band:band + 3]
            rising = (bins - low) / (centre - low)
            falling = (high - bins) / (high - centre)
            matrix[band] = np.clip(np.minimum(rising, falling), 0, None)
        # Narrow low bands may fall between bins: give each its nearest bin.
        for band in np.flatnonzero(matrix.sum(1) == 0):
            matrix[band, np.argmin(np.abs(bins - edges[band + 1]))] = 1.0
        _BAND_CACHE[key] = matrix
    return _BAND_CACHE[key]


def _spectra(signal, size):
    hop = size // 4
    padded = np.pad(signal, (size // 2, size // 2 + max(0, size - signal.size)))
    count = 1 + (padded.size - size) // hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop][:count]
    return np.abs(np.fft.rfft(frames * np.hanning(size), axis=1)) ** 2


def _log_frequency_cdf(signal, sample_rate):
    """Smoothed cumulative power over log frequency; mass-conserving, so it moves smoothly with pitch."""
    size = RESOLUTIONS[-1]
    power = _spectra(signal, size).mean(0)
    bins = np.fft.rfftfreq(size, 1 / sample_rate)
    keep = bins >= MINIMUM_HZ
    octaves = np.log2(bins[keep] / MINIMUM_HZ)
    # Each bin owns the octave interval up to the next bin: interpolating the
    # cumulative sum spreads its mass over that interval instead of sampling a peak.
    edges = np.append(octaves, octaves[-1] + (octaves[-1] - octaves[-2]))
    cumulative = np.concatenate(([0.0], np.cumsum(power[keep])))
    grid = np.linspace(0, octaves[-1], GRID_POINTS)
    cdf = np.interp(grid, edges, cumulative)
    total = cumulative[-1]
    if total <= 1e-30:
        return grid / grid[-1], float(octaves[-1])
    width = SMOOTHING_OCTAVES / (grid[1] - grid[0])
    taps = np.arange(-int(4 * width), int(4 * width) + 1)
    kernel = np.exp(-0.5 * (taps / width) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate((np.zeros(taps.size // 2), cdf / total, np.ones(taps.size // 2)))
    return np.convolve(padded, kernel, mode="valid"), float(octaves[-1])


def features(audio, sample_rate):
    """phi(y): blocks scaled so ||phi(a) - phi(b)||^2 is a sum of per-block means."""
    audio = _audio(audio)
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Positive finite sample rate required")
    blocks = []
    for channel in audio.T:
        for size in RESOLUTIONS:
            energy = _spectra(channel, size) @ _band_matrix(size, sample_rate).T
            # Window-normalised: a full-scale sine peaks near 0.25 in its band.
            energy = energy / np.hanning(size).sum() ** 2
            block = np.log(energy + ENERGY_FLOOR).ravel()
            blocks.append(block * math.sqrt(BAND_WEIGHT / (block.size * len(RESOLUTIONS))))
        hops = max(1, channel.size // ENVELOPE_HOP)
        rms = np.sqrt(np.mean(channel[:hops * ENVELOPE_HOP].reshape(hops, -1) ** 2, axis=1))
        envelope = np.log(rms + 1e-5)
        blocks.append(envelope / math.sqrt(envelope.size))
        cdf, octaves = _log_frequency_cdf(channel, sample_rate)
        # Cramer distance in octaves: L2 of CDF difference over log frequency.
        blocks.append(cdf * math.sqrt(CDF_WEIGHT * octaves / cdf.size))
    return np.concatenate(blocks)


def spectral_distance(actual, expected, sample_rate):
    """Phase-free reconstruction loss: ||phi(a) - phi(e)||^2 + W1 on log frequency (octaves)."""
    actual, expected = _audio(actual), _audio(expected)
    if actual.shape != expected.shape:
        raise ValueError("Reconstruction audio shapes differ")
    difference = features(actual, sample_rate) - features(expected, sample_rate)
    w1 = 0.0
    for a, e in zip(actual.T, expected.T):
        cdf_a, octaves = _log_frequency_cdf(a, sample_rate)
        cdf_e, _ = _log_frequency_cdf(e, sample_rate)
        w1 += float(np.mean(np.abs(cdf_a - cdf_e)) * octaves)
    return float(difference @ difference) + W1_WEIGHT * w1


def free_coordinates(document, bounds):
    """Unlocked, present, non-degenerate parameter coordinates of one tensor program."""
    parameters = document["tensors"]["parameters"]
    width = len(parameters["values"])
    locks = document.get("parameter_locks", [False] * width)
    presence = parameters.get("presence", [True] * width)
    bounds = np.asarray(bounds, dtype=np.float64).reshape(-1, 2)
    return np.array([not locks[i] and bool(presence[i]) and bounds[i, 1] > bounds[i, 0] for i in range(width)])


def parameter_metric(render, u_ref, free_mask, coordinates, *, step, rank, sample_rate,
                     descriptors=None, cancelled=None):
    """Offline Gauss-Newton factor of phi around u_ref by central differences in u.

    render(p) returns [sample, channel] audio for parameter vector p. descriptors(audio)
    optionally returns (values[d], valid[d]) and adds Jd rows. At a bound the step becomes
    one-sided so every rendered candidate stays inside the declared range.
    Costs 2 renders per free coordinate, once per example, never per training step.
    """
    if not math.isfinite(step) or step <= 0 or type(rank) is not int or rank <= 0:
        raise ValueError("Positive finite step and positive integer rank required")
    u_ref = np.asarray(u_ref, dtype=np.float64)
    free = np.asarray(free_mask, dtype=bool)
    low, high = coordinates.u_bounds()
    columns, descriptor_columns, valid = [], [], None
    for index in range(u_ref.size):
        if not free[index]:
            columns.append(None)
            continue
        if cancelled and cancelled():
            from .session import TrainingCancelled
            raise TrainingCancelled("Cancelled during perceptual metric measurement")
        plus, minus = u_ref.copy(), u_ref.copy()
        plus[index] = min(high[index], u_ref[index] + step)
        minus[index] = max(low[index], u_ref[index] - step)
        span = plus[index] - minus[index]
        if span <= 0:
            columns.append(None)
            continue
        audio_plus, audio_minus = render(coordinates.to_p(plus)), render(coordinates.to_p(minus))
        columns.append((features(audio_plus, sample_rate) - features(audio_minus, sample_rate)) / span)
        if descriptors is not None:
            (value_plus, valid_plus), (value_minus, valid_minus) = descriptors(audio_plus), descriptors(audio_minus)
            both = np.asarray(valid_plus, dtype=bool) & np.asarray(valid_minus, dtype=bool)
            valid = both if valid is None else valid & both
            descriptor_columns.append((index, (np.asarray(value_plus) - np.asarray(value_minus)) / span))
    used = [c for c in columns if c is not None]
    if not used:
        raise ValueError("Perceptual metric needs at least one free, non-degenerate parameter")
    J = np.zeros((used[0].size, u_ref.size))
    for index, column in enumerate(columns):
        if column is not None:
            J[:, index] = column
    _, singular, vt = np.linalg.svd(J, full_matrices=False)
    keep = min(rank, singular.size)
    result = {"schema": SCHEMA, "feature_id": FEATURE_ID, "u_ref": u_ref, "step": np.float64(step),
              "U": vt[:keep].T.copy(), "S": singular[:keep].copy(), "free_mask": free.copy()}
    if descriptors is not None and descriptor_columns:
        Jd = np.zeros((descriptor_columns[0][1].size, u_ref.size))
        for index, column in descriptor_columns:
            Jd[:, index] = column
        Jd[~valid] = 0.0
        result["Jd"], result["Jd_valid"] = Jd, valid
    return result


def save_metric(path, metric):
    with Path(path).open("wb") as stream:
        np.savez(stream, **{k: np.asarray(v) for k, v in metric.items()})


def load_metric(path, width, rank):
    with np.load(Path(path), allow_pickle=False) as data:
        metric = {k: data[k] for k in data.files}
    if not schema_is(str(metric.get("schema")), SCHEMA) or not schema_is(str(metric.get("feature_id")), FEATURE_ID):
        raise ValueError("Unsupported perceptual metric file")
    U, S = metric["U"].astype(np.float64), metric["S"].astype(np.float64)
    if U.ndim != 2 or U.shape[0] != width or U.shape[1] != S.size or S.size > rank or not np.isfinite(U).all() or not np.isfinite(S).all() or (S < 0).any():
        raise ValueError("Perceptual metric factor does not match the parameter schema")
    # Pad to the declared rank so every example stacks into one [N, K, rank] tensor.
    padded_U, padded_S = np.zeros((width, rank)), np.zeros(rank)
    padded_U[:, :S.size], padded_S[:S.size] = U, S
    metric["U"], metric["S"] = padded_U, padded_S
    return metric
