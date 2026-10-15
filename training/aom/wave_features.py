"""Wave feature knobs (aom.wave-features/1): eight measured audio descriptors.

Each knob asks for a relative move of one descriptor of the rendered sound. Knob 0 is the
loaded sound. The descriptor is the main direction of the move; the sound moves through the
continuous tensor space, so other descriptors may change with it and the network chooses how.

Every descriptor is a fixed algorithm with a unit, measured on output audio, so training needs
no labels: turn knobs, render, measure, compare. Measurements are relative to the knob-0
render of the same input, which makes instrument, FX and hybrid operation identical.

The numpy measurements and the teacher directions are exact. The PyTorch losses take
already-measured percentiles (from any renderer gradient: SPSA, primitive JVPs or a
differentiable proxy); they do not differentiate the measurements themselves.
"""
from dataclasses import dataclass
import json
import math
import numpy as np

SCHEMA = "aom.wave-features/1"
DESCRIPTORS = ("tilt", "attack", "sustain", "tail", "noisiness", "roughness", "motion", "width")
UNITS = ("dB/oct", "-log10 s", "dB", "log10 s", "1 - cos", "modulation depth^2", "dB", "dB")
DEFAULT_GAIN = 0.35          # percentile move at |knob| = 1
_EDGES_HZ = (50.0, 10000.0)
_BANDS = 24


@dataclass(frozen=True)
class Segment:
    """One note or sound event in seconds; offset None when the sound never stops."""
    onset: float
    offset: float | None


def _mono(audio):
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim == 1:
        y = y[None]
    if y.ndim != 2 or y.shape[1] == 0 or not np.isfinite(y).all():
        raise ValueError("Audio must be finite [channel, sample] or [sample]")
    return y, y.mean(axis=0)


def _frames(x, size, hop):
    if x.size < size:
        x = np.pad(x, (0, size - x.size))
    return np.lib.stride_tricks.sliding_window_view(x, size)[::hop]


def _envelope(x, sr):
    """Mean power on ~2.7 ms hops (128 samples at 48 kHz) with ~5.3 ms windows."""
    hop = max(1, round(0.00267 * sr))
    win = 2 * hop
    power = _frames(np.pad(x ** 2, (0, win)), win, hop).mean(axis=1)
    return power, np.arange(power.size) * hop / sr


def detect_segments(audio, sample_rate, *, gate_db=-30.0, min_gap=0.05):
    """Segments from the signal envelope, for FX input or audio without MIDI.

    Onsets and offsets are crossings of a gate `gate_db` below the overall peak, with a
    hysteresis of 6 dB and gaps shorter than `min_gap` seconds merged.
    """
    _, x = _mono(audio)
    power, t = _envelope(x, sample_rate)
    level = 10 * np.log10(power + 1e-20)
    on_level, off_level = level.max() + gate_db, level.max() + gate_db - 6
    segments, start = [], None
    for i, v in enumerate(level):
        if start is None and v > on_level:
            start = t[i]
        elif start is not None and v < off_level:
            if segments and start - (segments[-1].offset or 0) < min_gap:
                start = segments.pop().onset
            segments.append(Segment(start, float(t[i])))
            start = None
    if start is not None:
        segments.append(Segment(float(start), None))
    return segments


def measure(audio, sample_rate, segments=None):
    """Return (values[8], valid[8]) for audio [channel, sample].

    `segments` are note spans, from voice_indexed MIDI in instrument mode or
    `detect_segments` otherwise. A descriptor that cannot be measured on this audio is
    marked invalid (tail without an offset, width of mono, tilt with no spectrum).
    """
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Explicit positive sample rate required")
    y, x = _mono(audio)
    sr = float(sample_rate)
    duration = x.size / sr
    segments = list(segments) if segments is not None else detect_segments(y, sr)
    if not segments:
        segments = [Segment(0.0, None)]
    values, valid = np.zeros(8), np.ones(8, dtype=bool)

    # held regions: from 0.2 s (or 30 % of the note) after onset to 50 ms before offset
    def held(t):
        mask = np.zeros(t.size, dtype=bool)
        for s in segments:
            end = s.offset if s.offset is not None else duration
            start = s.onset + min(0.2, 0.3 * (end - s.onset))
            mask |= (t >= start) & (t < end - 0.05)
        return mask if mask.any() else np.ones(t.size, dtype=bool)

    # spectrogram, ~43 ms windows
    n_fft = 1 << int(round(math.log2(0.0427 * sr)))
    hop = n_fft // 4
    spec = np.abs(np.fft.rfft(_frames(x, n_fft, hop) * np.hanning(n_fft), axis=1)) ** 2
    ft = np.arange(spec.shape[0]) * hop / sr
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(_EDGES_HZ[0], min(_EDGES_HZ[1], 0.45 * sr), _BANDS + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    bands = [(freqs >= lo) & (freqs < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    h = held(ft)
    held_spec = spec[h]

    # D0 tilt: weighted slope of band power density (dB) against log2 frequency
    density = np.array([held_spec[:, b].mean() if b.any() else 0.0 for b in bands]) + 1e-20
    if (density > 1e-18).sum() >= 3:
        slope = np.polyfit(np.log2(centres), 10 * np.log10(density), 1, w=(density / density.max()) ** 0.25)[0]
        values[0] = float(np.clip(slope, -24, 24))
    else:
        valid[0] = False

    # envelope descriptors per segment
    power, et = _envelope(x, sr)
    amp = np.sqrt(power)
    step = et[1] - et[0] if et.size > 1 else 1 / sr
    attacks, sustains, tails = [], [], []
    for i, s in enumerate(segments):
        end = s.offset if s.offset is not None else duration
        note = (et >= s.onset) & (et < end)
        if note.sum() < 3:
            continue
        a, t = amp[note], et[note]
        peak = a.max()
        if peak <= 0:
            continue
        rise = max(t[np.argmax(a >= 0.9 * peak)] - t[np.argmax(a >= 0.1 * peak)], step)
        attacks.append(-math.log10(rise))
        late = t >= s.onset + 0.7 * (end - s.onset)
        sustains.append(10 * math.log10(power[note][late].mean() / peak ** 2 + 1e-9))
        if s.offset is not None:
            nxt = segments[i + 1].onset if i + 1 < len(segments) else duration
            post = power[(et >= s.offset) & (et < nxt)]
            if post.size >= 4 and post.sum() > 0:
                schroeder = np.cumsum(post[::-1])[::-1]
                decay = 10 * np.log10(schroeder / schroeder[0] + 1e-12)
                i30 = int(np.argmax(decay <= -30)) if (decay <= -30).any() else decay.size - 1
                tails.append(math.log10(max(i30, 1) * step))
    for k, items in ((1, attacks), (2, sustains), (3, tails)):
        if items:
            values[k] = float(np.median(items))
        else:
            valid[k] = False

    # D4 noisiness: phase unpredictability, power weighted (steady partials ~0, noise > 0)
    held_x = np.concatenate([x[int(s.onset * sr):int(((s.offset or duration)) * sr)] for s in segments]) \
        if segments else x
    long_fft = 4 * n_fft
    X = np.fft.rfft(_frames(held_x, long_fft, long_fft // 8) * np.hanning(long_fft), axis=1)
    if X.shape[0] >= 3:
        phase = np.angle(X)
        dd = phase[2:] - 2 * phase[1:-1] + phase[:-2]
        weight = np.abs(X[2:]) ** 2
        values[4] = float(((1 - np.cos(dd)) * weight).sum() / (weight.sum() + 1e-20))
    else:
        valid[4] = False

    # D5 roughness: band envelope modulation power in 20-150 Hz over DC, energy weighted
    S = np.fft.fft(held_x)
    fs = np.fft.fftfreq(held_x.size, 1 / sr)
    rough, weights = [], []
    for fc in centres:
        H = np.where(fs > 0, 2.0 * np.exp(-0.5 * (np.log2(np.maximum(fs, 1) / fc) / 0.3) ** 2), 0.0)
        env = np.abs(np.fft.ifft(S * H))
        E = np.abs(np.fft.rfft(env)) ** 2
        fm = np.fft.rfftfreq(env.size, 1 / sr)
        rough.append(E[(fm >= 20) & (fm <= 150)].sum() / (E[0] + 1e-20))
        weights.append((env ** 2).mean())
    weights = np.array(weights)
    values[5] = float(np.dot(rough, weights) / (weights.sum() + 1e-20))

    # D6 motion: std of detrended band level (dB) over held frames, energy weighted
    E = np.stack([held_spec[:, b].sum(axis=1) if b.any() else np.zeros(held_spec.shape[0]) for b in bands], 1) + 1e-12
    if E.shape[0] >= 3:
        L = 10 * np.log10(E)
        tt = ft[h]
        A = np.vstack([tt, np.ones_like(tt)]).T
        resid = L - A @ np.linalg.lstsq(A, L, rcond=None)[0]
        bw = E.mean(axis=0) / E.mean(axis=0).sum()
        values[6] = float((resid.std(axis=0) * bw).sum())
    else:
        valid[6] = False

    # D7 width: side/mid energy ratio
    if y.shape[0] >= 2:
        mid, side = (y[0] + y[1]) / 2, (y[0] - y[1]) / 2
        values[7] = 10 * math.log10((side ** 2).sum() / ((mid ** 2).sum() + 1e-20) + 1e-6)
    else:
        valid[7] = False
    return values, valid


@dataclass(frozen=True)
class Calibration:
    """Empirical CDF of each descriptor over a background set of renders.

    Knob targets live in percentile space, so they saturate at 0 and 1: a sound that is
    already as wide as anything in the background set cannot be asked to get wider.
    """
    quantiles: tuple[tuple[float, ...], ...]   # 8 sorted sample lists

    @staticmethod
    def fit(samples, valid=None, points=101):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 8 or samples.shape[0] < 2:
            raise ValueError("Calibration needs [renders, 8] descriptor samples")
        valid = np.ones(samples.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        q = []
        for k in range(8):
            column = samples[valid[:, k], k]
            if column.size < 2:
                raise ValueError(f"Descriptor {DESCRIPTORS[k]} has fewer than two valid samples")
            q.append(tuple(np.quantile(column, np.linspace(0, 1, points)).tolist()))
        return Calibration(tuple(q))

    def percentile(self, values):
        values = np.asarray(values, dtype=np.float64)
        out = np.empty_like(values)
        for k in range(8):
            q = np.asarray(self.quantiles[k])
            grid = np.linspace(0, 1, q.size)
            # strictly increasing x for interp: ties collapse to their mean percentile
            uq, inverse = np.unique(q, return_inverse=True)
            ug = np.bincount(inverse, grid) / np.bincount(inverse)
            out[..., k] = np.interp(values[..., k], uq, ug) if uq.size > 1 else 0.5
        return out

    def to_json(self):
        return {"schema": SCHEMA, "descriptors": list(DESCRIPTORS), "quantiles": [list(q) for q in self.quantiles]}

    @staticmethod
    def from_json(document):
        if document.get("schema") != SCHEMA or document.get("descriptors") != list(DESCRIPTORS):
            raise ValueError("Unsupported wave feature calibration")
        return Calibration(tuple(tuple(float(v) for v in q) for q in document["quantiles"]))


def knob_target(base_percentile, knobs, gain=DEFAULT_GAIN):
    """Percentile each descriptor is asked to reach: base + gain * knob, clipped to [0, 1]."""
    knobs = np.asarray(knobs, dtype=np.float64)
    if knobs.shape[-1] != 8 or np.abs(knobs).max(initial=0) > 1 or not np.isfinite(knobs).all():
        raise ValueError("Eight knob values in [-1, 1] required")
    return np.clip(np.asarray(base_percentile) + gain * knobs, 0.0, 1.0)


def _solve(lam_diag, U, S, rhs):
    """(diag(lam) + U diag(S^2) U^T)^-1 rhs by Woodbury; rhs is [K] or [K, m]."""
    inv_d = 1.0 / lam_diag
    if U.size == 0:
        return inv_d[:, None] * rhs if rhs.ndim == 2 else inv_d * rhs
    DU = inv_d[:, None] * U
    core = np.diag(1.0 / np.maximum(S, 1e-12) ** 2) + U.T @ DU
    Dr = inv_d[:, None] * rhs if rhs.ndim == 2 else inv_d * rhs
    return Dr - DU @ np.linalg.solve(core, U.T @ Dr)


def teacher_directions(Jd, U, S, lam_diag, free_mask, *, hold_others=0.0, min_row_norm=1e-6):
    """Main-direction knob teacher: the cheapest move (in the perceptual metric) along d_k.

    For metric M = diag(lam) + U diag(S^2) U^T (the training-gradient proposal's G + lambda D,
    in unconstrained coordinates u) and descriptor Jacobian Jd [8, K]:

        dp_k = M_k^-1 j_k / (j_k^T M_k^-1 j_k),   M_k = M + hold_others * J_(-k)^T J_(-k)

    dp_k moves descriptor k by exactly one unit with the least perceptual change. With
    hold_others = 0 (default, Sinin 2026-10-15) the other descriptors move however the
    sound naturally moves along that path; larger values damp them. Locked or absent
    coordinates (free_mask False) are excluded, not zero-weighted. A knob whose row is
    below min_row_norm for this sound is inactive. Returns (directions [K, 8], active [8]).
    """
    Jd, U, S, lam = (np.asarray(a, dtype=np.float64) for a in (Jd, U, S, lam_diag))
    free = np.asarray(free_mask, dtype=bool)
    K = Jd.shape[1]
    if Jd.shape[0] != 8 or lam.shape != (K,) or free.shape != (K,) or U.shape[0] != K or U.shape[1] != S.size:
        raise ValueError("Jd [8,K], U [K,r], S [r], lam [K] and free_mask [K] required")
    if not (lam[free] > 0).all() or not math.isfinite(hold_others) or hold_others < 0:
        raise ValueError("Positive identity weights and nonnegative hold_others required")
    J, Uf, lf = Jd[:, free], U[free], lam[free]
    out, active = np.zeros((K, 8)), np.zeros(8, dtype=bool)
    for k in range(8):
        j = J[k]
        if np.linalg.norm(j) < min_row_norm:
            continue
        if hold_others > 0:
            others = np.delete(J, k, axis=0)
            Wk = np.hstack([Uf * S, math.sqrt(hold_others) * others.T])
            v = _solve(lf, Wk, np.ones(Wk.shape[1]), j)
        else:
            v = _solve(lf, Uf, S, j)
        denom = float(j @ v)
        if not denom > 0:
            continue
        out[free, k] = v / denom
        active[k] = True
    return out, active


def follow_main_direction(u0, k, amount, jacobian_at, *, steps=8, **teacher_options):
    """Integrate du/ds = teacher direction of descriptor k from u0 for `amount` units.

    Large knob moves follow a curve through the tensor space, re-linearised at every step:
    jacobian_at(u) returns (Jd, U, S, lam_diag, free_mask). Used offline to produce teacher
    trajectories beyond the tangent at knob 0. Returns the path [steps + 1, K].
    """
    u = np.asarray(u0, dtype=np.float64).copy()
    path = [u.copy()]
    for _ in range(steps):
        directions, active = teacher_directions(*jacobian_at(u), **teacher_options)
        if not active[k]:
            break
        u = u + (amount / steps) * directions[:, k]
        path.append(u.copy())
    return np.array(path)


def sample_knob_curves(rng, frames, *, move_probability=0.5, theta=0.05, sigma=0.15):
    """Random knob automation [frames, 8] in [-1, 1] for training the runtime compiler.

    Each knob is independently still (0), an Ornstein-Uhlenbeck wander, a step or a ramp.
    """
    if type(frames) is not int or frames <= 0:
        raise ValueError("Positive frame count required")
    curves = np.zeros((frames, 8))
    for k in range(8):
        if rng.random() >= move_probability:
            continue
        kind = rng.integers(3)
        if kind == 0:
            x, values = rng.uniform(-1, 1), []
            for _ in range(frames):
                x += -theta * x + sigma * rng.standard_normal()
                values.append(x)
            curves[:, k] = values
        elif kind == 1:
            at = rng.integers(frames)
            curves[at:, k] = rng.uniform(-1, 1)
        else:
            a, b = rng.uniform(-1, 1, 2)
            curves[:, k] = np.linspace(a, b, frames)
    return np.clip(curves, -1, 1)


def fallback_matrix(directions, active, parameter_per_u, gain=DEFAULT_GAIN, percentile_slope=None):
    """Linear fallback for programs without a compiler model that reads wave features.

    directions [K, 8] are teacher moves in u per descriptor unit; parameter_per_u [K] is
    d(parameter)/du at the sound's u (log, sigmoid and phase reparametrisations); the
    optional percentile_slope [8] is d(percentile)/d(descriptor) at the sound, so knob 1
    asks for `gain` percentile. Returns the program's `wave_features` block, whose matrix
    adds into the parameter tensor after the X layers.
    """
    directions = np.asarray(directions, dtype=np.float64)
    scale = np.asarray(parameter_per_u, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 8 or scale.shape != (directions.shape[0],):
        raise ValueError("directions [K, 8] and parameter_per_u [K] required")
    slope = np.ones(8) if percentile_slope is None else np.asarray(percentile_slope, dtype=np.float64)
    units = np.where(np.asarray(active, dtype=bool) & (slope > 0), gain / np.where(slope > 0, slope, 1), 0.0)
    matrix = scale[:, None] * directions * units[None, :]
    if not np.isfinite(matrix).all():
        raise ValueError("Fallback matrix must be finite")
    return {"schema": SCHEMA, "descriptors": list(DESCRIPTORS),
            "matrix": {"axes": ["destination", "source"], "shape": list(matrix.shape),
                       "values": matrix.flatten().tolist()}}


def knob_losses(base_percentile, rendered_percentile, knobs, valid, *, gain=DEFAULT_GAIN,
                dominance_weight=1.0, monotone_weight=1.0, huber_delta=0.05):
    """PyTorch losses for rendered knob moves, all in percentile units ([batch, 8] tensors).

    target    Huber(rendered_k - target_k) for every turned knob k.
    dominance for each unturned descriptor j: relu(|move_j| - max |move_k| over turned k),
              so the turned descriptors are the main direction while others may follow.
    monotone  relu(-sign(c_k) * move_k): turning a knob never moves its descriptor backwards.
    Unturned descriptors have no target of their own (Sinin 2026-10-15). Invalid
    descriptors (not measurable on that sound) are excluded.
    """
    import torch
    base, out, c = base_percentile, rendered_percentile, knobs
    valid = valid.to(out.dtype)
    turned = (c.abs() > 1e-6).to(out.dtype) * valid
    target = (base + gain * c).clamp(0, 1)
    move = out - base
    huber = torch.nn.functional.huber_loss(out, target, reduction="none", delta=huber_delta)
    target_loss = (huber * turned).sum() / turned.sum().clamp_min(1)
    main = (move.abs() * turned).amax(dim=-1, keepdim=True)
    excess = torch.relu(move.abs() - main) * (1 - turned) * valid
    dominance = excess.sum() / ((1 - turned) * valid).sum().clamp_min(1)
    backwards = torch.relu(-torch.sign(c) * move) * turned
    monotone = backwards.sum() / turned.sum().clamp_min(1)
    return {"target": target_loss, "dominance": dominance, "monotone": monotone,
            "total": target_loss + dominance_weight * dominance + monotone_weight * monotone}


def metric_norm(delta, lam_diag, U, S):
    """delta^T (diag(lam) + U S^2 U^T) delta for PyTorch [batch, K]: the local teacher loss
    (with delta = d p/d c_k - gain * teacher_k) and the knob smoothness term (with delta =
    p_t - p_(t-1)) both use the training-gradient metric."""
    return (lam_diag * delta.square()).sum(-1) + ((delta @ U) * S).square().sum(-1)


def save_calibration(path, calibration):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(calibration.to_json(), handle)
