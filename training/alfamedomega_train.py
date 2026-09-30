#!/usr/bin/env python3
"""Alfamedomega training script.

Implements training/METHOD.md: a frame-rate (250 Hz) control network with a shared trunk and
per-method routed experts, differentiable reference DSP primitives that follow the plugin's
control-frame contract (src/dsp/include/alfa/dsp/ControlFrame.h, layout version 2), procedural
teacher programs, multi-resolution spectral loss, Studio -> smaller-tier distillation, and
multi-graph ONNX export.

Runs on CPU for smoke tests and on a Colab GPU for real training:
    python alfamedomega_train.py report                          # param counts vs tier spec
    python alfamedomega_train.py train --tier lite --smoke       # tiny end-to-end run on CPU
    python alfamedomega_train.py train --tier studio --steps 200000 --out /content/drive/MyDrive/amo
    python alfamedomega_train.py train --tier studio --stage polish --init studio_pretrain.pt
    python alfamedomega_train.py train --tier studio --stage finetune --data nsynth-valid --init studio_polish.pt
    python alfamedomega_train.py train --tier pro --stage distill --teacher studio_finetune.pt
    python alfamedomega_train.py export --ckpt pro_distill.pt --out export/pro
"""
import argparse
import json
import math
import os
import random
import sys
import time
import wave
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- constants (METHOD.md s1, s2, s6)
LAYOUT_VERSION = 2              # must equal kLayoutVersion in ControlFrame.h (src/dsp/layout.json)
SR = 48000
FRAME_HZ = 250
HOP = int(SR / FRAME_HZ + 0.5)  # 192 samples per control frame (hopSamples)
N_CYC = 2048                    # kOscTableSize
N_HARM = 1024                   # kOscHarmonics
N_BANDS = 64                    # kBands (NOISE and SAMPLER EQ)
N_KNOBS = 32
N_PERF = 8
N_GLOBAL = 4
GRAIN_LEN = 8192                # kGrainSourceLength
SAMPLE_ROOT_HZ = 261.63         # factory samples and grain sources are rooted at C4
ON_FRAME = 4                    # note-on frame in training clips (pre-roll trains silence)
MAX_T_S = 4.0
PRIMS = ["osc", "partials", "fm", "noise", "grain", "sampler"]   # order of VoiceFrame.mix

# VoiceFrame fields in layout order (ControlFrame.h offsets): name, floats
VF_FIELDS = [("mix", 6), ("attack_tau", 1), ("reserved", 1), ("osc.re", 1024), ("osc.im", 1024),
             ("partials.ratio", 64), ("partials.amp", 64), ("fm.ratio", 8), ("fm.level", 8),
             ("fm.index", 64), ("noise.band", 64), ("grain.source", 16), ("grain.weight", 16),
             ("grain.position", 8), ("grain.rate", 8), ("grain.gain", 8), ("sampler.rate", 1),
             ("sampler.start", 1), ("sampler.gain", 1), ("sampler.eq", 64),
             # layout v2 (appended): pan, dry gain, OSC unison reads, processing slots, modulation
             ("pan", 1), ("dry_gain", 1), ("osc.detune_ratio", 7), ("osc.detune_level", 7),
             ("slots", 112), ("mod", 48)]
VF_OFFSET = {}
_o = 0
for _n, _c in VF_FIELDS:
    VF_OFFSET[_n] = (_o, _c)
    _o += _c
VF_FLOATS = _o
assert VF_FLOATS == 2627
# Non-zero defaults of a default-constructed VoiceFrame. A producer that leaves them at 0 silences
# the voice (dry_gain) or detunes silent reads to 0 Hz.
VF_DEFAULTS = {"dry_gain": 1.0, "osc.detune_ratio": 1.0}
MAX_DETUNE = 7                  # kMaxOscDetune: extra OSC table reads
SAFETY_KNEE = 2.0               # Synth output safety: linear to +6 dBFS, tanh to +12 dBFS
# InstanceFrame (instance_step's frame for the stereo instance chain), layout order
IF_FIELDS = [("dry_gain", 1), ("master_gain", 1), ("width", 1), ("reserved", 1), ("slots", 112), ("mod", 48)]
IF_FLOATS = sum(c for _, c in IF_FIELDS)
assert IF_FLOATS == 164

METHODS = ["subtractive", "wavetable", "additive", "spectral", "fm", "physical", "granular",
           "sampling", "hybrid"]
M_IDX = {m: i for i, m in enumerate(METHODS)}
METHOD_IN = 3 * len(METHODS)    # [method one-hot, hybrid component A one-hot, component B one-hot]
# Hybrid components are chosen so the only primitives two components can share are OSC and NOISE,
# which merge exactly (OSC) or by power (NOISE) into one VoiceFrame; see merge_hybrid.
HYBRID_PARTS = ["subtractive", "wavetable", "additive", "fm", "physical"]
HYBRID_EXCLUDE = {frozenset(("additive", "physical"))}           # both use PARTIALS


def hybrid_pair():
    while True:
        pair = random.sample(HYBRID_PARTS, 2)
        if frozenset(pair) not in HYBRID_EXCLUDE:
            return pair

# ---------------------------------------------------------------- capacity plan (tier_budget.py)
PREVALENCE = {"sampling": 30, "wavetable": 22, "subtractive": 20, "fm": 8, "hybrid": 7,
              "granular": 4, "physical": 4, "additive": 3, "spectral": 2}
DIFFICULTY = {"sampling": 1.5, "physical": 1.4, "fm": 1.3, "wavetable": 1.2, "spectral": 1.2,
              "subtractive": 1.0, "granular": 0.8, "hybrid": 0.6, "additive": 0.5}
ALPHA, FLOOR, TRUNK_SHARE = 0.6, 0.04, 0.30
TABLE_FRAC = {"wavetable": 0.6, "granular": 0.5, "spectral": 0.4, "additive": 0.3,
              "sampling": 0.05, "physical": 0.05}


def expert_shares(p=PREVALENCE, d=DIFFICULTY, alpha=ALPHA, floor=FLOOR):
    floored = set()
    while True:
        s = sum(p[k] ** alpha * d[k] for k in p if k not in floored)
        c = {k: floor if k in floored else (1 - len(floored) * floor) * p[k] ** alpha * d[k] / s
             for k in p}
        new = floored | {k for k in p if c[k] < floor}
        if new == floored:
            return c
        floored = new


@dataclass
class Tier:
    name: str
    total: float        # total params (capacity plan)
    active: float       # active params per voice per frame (tier spec)
    instance: float     # per-instance per-frame net (tier spec)
    K_h: int            # explicit harmonics
    n_ops: int          # FM operators (TierLimits.fmOperators)
    n_modes: int        # partials (TierLimits.partials)
    n_grains: int       # grain layers (TierLimits.grainLayers)
    unison: int         # OSC unison reads including the main read (TierLimits.oscUnison)
    Hv: int             # voice width
    Hs: int             # routed sub-expert hidden width
    E: int              # patch embedding
    Hi: int             # instance GRU width
    Wp: int             # patch-encoder width
    Wa: int             # audio-encoder width
    lr: float


TIERS = {
    "lite":     Tier("lite",     4e6,   0.12e6, 0.10e6, 32,  2, 16, 2, 2, 96,  96,   128, 112,  256,  128, 3e-4),
    "standard": Tier("standard", 16e6,  0.35e6, 0.50e6, 64,  4, 32, 4, 4, 160, 192,  256, 288,  512,  192, 3e-4),
    "pro":      Tier("pro",      64e6,  0.65e6, 1.50e6, 128, 6, 48, 6, 6, 224, 256,  384, 512,  1024, 256, 2e-4),
    "studio":   Tier("studio",   256e6, 2.8e6,  6.0e6,  256, 8, 64, 8, 8, 416, 768,  512, 1100, 2048, 384, 1.5e-4),
}

# ---------------------------------------------------------------- small helpers


def midi_hz(m):
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def rng(x, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(x)


def log_rng(x, lo, hi):
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(x))


def exp_sig(x, maxv=2.0):
    """DDSP amplitude nonlinearity."""
    return maxv * torch.sigmoid(x) ** math.log(10.0) + 1e-7


def harm_k(dev, n=N_HARM):
    return torch.arange(1, n + 1, device=dev, dtype=torch.float32)


def band_hz(dev):
    """Band centres: 20 * (nyquist / 20)^(b / 63)."""
    return torch.logspace(math.log10(20.0), math.log10(SR / 2), N_BANDS, device=dev)


def band_interp(bands, f_hz):
    """Band curve [..., N_BANDS] at frequencies [..., K]: linear in log-frequency between band
    centres, flat below band 0 and above the top band."""
    u = (torch.log(f_hz.clamp(min=20.0)) - math.log(20.0)) / (math.log(SR / 2) - math.log(20.0))
    u = (u * (N_BANDS - 1)).clamp(0, N_BANDS - 1 - 1e-4)
    i0 = u.floor().long()
    fr = u - i0.float()
    bands = bands.expand(*i0.shape[:-1], N_BANDS)
    v0 = torch.gather(bands, -1, i0)
    v1 = torch.gather(bands, -1, i0 + 1)
    return v0 + (v1 - v0) * fr


class Slicer:
    def __init__(self, spec):
        self.spec, self.n = spec, sum(s for _, s in spec)

    def __call__(self, x):
        out, i = {}, 0
        for name, s in self.spec:
            out[name] = x[..., i:i + s]
            i += s
        return out


def tree_map(fn, x):
    if isinstance(x, dict):
        return {k: tree_map(fn, v) for k, v in x.items()}
    return fn(x) if torch.is_tensor(x) else x


def mix_vec(ref, *prims):
    """VoiceFrame.mix with 1 for the listed primitives, shaped like ref[..., 6]."""
    m = torch.zeros(*ref.shape, len(PRIMS), device=ref.device)
    for p in prims:
        m[..., PRIMS.index(p)] = 1.0
    return m


# ================================================================ DSP primitives (METHOD.md s2)
# Reference implementation of ControlFrame.h. A voice consumes frame j at the start of hop j;
# across hop j every "linear" control ramps from frame j-1 to frame j (a = i/HOP at sample i of
# the hop), and "held" controls use frame j. For hop 0, frame -1 is frame 0. Sample 0 of a clip
# is the note-on sample for the golden vectors; phases start at 0 there.

def _hop_index(S, T, dev):
    n = torch.arange(S, device=dev)
    j = (n // HOP).clamp(max=T - 1)
    return (j - 1).clamp(min=0), j, ((n % HOP).float() / HOP)


def frame_interp(x, S):
    """Linear per-sample ramp [B,T,...] -> [B,S,...]."""
    B, T = x.shape[:2]
    jp, j, a = _hop_index(S, T, x.device)
    a = a.view(1, S, *([1] * (x.dim() - 2)))
    return x[:, jp] * (1 - a) + x[:, j] * a


def frame_hold(x, S):
    B, T = x.shape[:2]
    return x[:, _hop_index(S, T, x.device)[1]]


def phase_cycles(freq, ratio=None, phase0=0.0):
    """Phase accumulator phi[n+1] = phi[n] + f[n]/SR, phi[0] = phase0, with f ramping linearly
    across each hop. Returns frac(phi) [B,S,...] from per-frame frequency in Hz [B,T,...]. With
    `ratio` [B,T,...] the frequency is f * ratio, both ramping linearly (so quadratic within a
    hop), as the OSC unison reads run. Hop starts are accumulated in float64 so long notes stay
    exact."""
    B, T = freq.shape[:2]
    if ratio is None:
        ratio = torch.ones_like(freq)
    sh = lambda x: torch.cat([x[:, :1], x[:, :-1]], 1)
    f_p, r_p = sh(freq), sh(ratio)
    df, dr = freq - f_p, ratio - r_p
    a, b, c = f_p * r_p, (f_p * dr + df * r_p) / HOP, df * dr / HOP ** 2   # f(i) = a + b i + c i^2
    H = float(HOP)
    inc = (a.double() * H + b.double() * H * (H - 1) / 2 + c.double() * (H - 1) * H * (2 * H - 1) / 6) / SR
    start = torch.cumsum(inc, 1) - inc + phase0
    start = (start - start.floor()).float()
    i = torch.arange(HOP, device=freq.device, dtype=torch.float32).view(1, 1, HOP, *([1] * (freq.dim() - 2)))
    within = (a.unsqueeze(2) * i + b.unsqueeze(2) * i * (i - 1) / 2
              + c.unsqueeze(2) * (i - 1) * i * (2 * i - 1) / 6) / SR
    ph = start.unsqueeze(2) + within
    ph = ph - ph.detach().floor()
    return ph.reshape(B, T * HOP, *freq.shape[2:])


def osc_render(re, im, f0, det_ratio=None, det_level=None):
    """OSC: re/im [B,T,N_HARM], f0 [B,T]. Frame j's wave is sum_k re cos(2pi(k+1)phi) -
    im sin(2pi(k+1)phi), stored as a 2048-point table read with linear interpolation; hop j
    crossfades table j-1 -> table j. Table j drops harmonics at or above Nyquist for the f0 at the
    start of hop j (frame j-1's; frame j's on the first hop) times the largest ratio of any unison
    read sounding in frame j-1 or j (at least 1).
    Unison (layout v2): det_ratio/det_level [B,T,R], R <= 7. Read r runs its own phase at
    f0 * ratio_r (both linear across the hop) from phase (r+1)/8 at sample 0, reads the same
    crossfaded tables, and adds level_r (linear) times the result."""
    B, T, H = re.shape
    S = T * HOP
    k = harm_k(re.device, H)
    sh = lambda x: torch.cat([x[:, :1], x[:, :-1]], 1)
    scale = torch.ones_like(f0)
    if det_level is not None:
        on = ((sh(det_level) != 0) | (det_level != 0)).float()
        top = (torch.maximum(sh(det_ratio), det_ratio) * on).amax(-1)
        scale = torch.maximum(scale, top.detach())
    mask = (k * (sh(f0) * scale).unsqueeze(-1) < SR / 2).float()
    Z = F.pad(torch.complex(re * mask, im * mask), (1, N_CYC // 2 + 1 - H - 1))
    cyc = (torch.fft.irfft(Z, n=N_CYC) * (N_CYC / 2)).reshape(B, T * N_CYC)
    pos = phase_cycles(f0) * N_CYC                                # [B,S]
    i0 = pos.floor().long() % N_CYC
    i1 = (i0 + 1) % N_CYC
    fr = pos - pos.floor()
    jp, j, a = _hop_index(S, T, re.device)

    def read(t, i0, i1, fr):
        base = (t * N_CYC).view(1, S)
        g0 = torch.gather(cyc, 1, base + i0)
        g1 = torch.gather(cyc, 1, base + i1)
        return g0 + (g1 - g0) * fr

    y = read(jp, i0, i1, fr) * (1 - a) + read(j, i0, i1, fr) * a
    if det_level is not None:
        lv = frame_interp(det_level, S)
        for r in range(det_level.shape[-1]):
            pos = phase_cycles(f0, det_ratio[..., r], (r + 1) / (MAX_DETUNE + 1)) * N_CYC
            q0 = pos.floor().long() % N_CYC
            q1 = (q0 + 1) % N_CYC
            qf = pos - pos.floor()
            y = y + lv[..., r] * (read(jp, q0, q1, qf) * (1 - a) + read(j, q0, q1, qf) * a)
    return y


def partials_render(ratio, amp, f0):
    """PARTIALS: ratio/amp [B,T,M], both linear per sample; silent at or above Nyquist."""
    S = f0.shape[1] * HOP
    fr = f0.unsqueeze(-1) * ratio
    a = frame_interp(amp, S) * (frame_interp(fr, S) < SR / 2).float()
    return (a * torch.sin(2 * math.pi * phase_cycles(fr))).sum(-1)


def fm_render(ratio, level, index, f0):
    """FM: ratio/level [B,T,N], index [B,T,N,N] (radians, j>i used). y_i = sin(theta_i +
    sum_{j>i} index[i][j] y_j), last operator first; output sum_i level_i y_i."""
    B, T, N = ratio.shape
    S = T * HOP
    ph = phase_cycles(f0.unsqueeze(-1) * ratio)
    lv = frame_interp(level, S)
    ix = frame_interp(torch.triu(index, 1).reshape(B, T, N * N), S).view(B, S, N, N)
    ys = [None] * N
    for i in reversed(range(N)):
        m = 0.0
        for j in range(i + 1, N):
            m = m + ix[..., i, j] * ys[j]
        ys[i] = torch.sin(2 * math.pi * ph[..., i] + m)
    return sum(lv[..., i] * ys[i] for i in range(N))


def _hann(n, dev):
    return torch.hann_window(n, periodic=True, device=dev)


def _ola(frames, S, first=0):
    """Overlap-add [B,T',2*HOP] frames; frame i starts at (i + first) * HOP."""
    B, Tp, n = frames.shape
    y = F.fold(frames.transpose(1, 2), output_size=(1, (Tp + 1) * HOP), kernel_size=(1, n),
               stride=(1, HOP)).view(B, -1)
    return y[:, -first * HOP:][:, :S] if first < 0 else y[:, :S]


def _band_gain(frames, bands, dc_zero):
    """Multiply the spectrum of [B,T',2*HOP] frames by the band curve. NOISE zeroes the DC bin;
    the SAMPLER EQ keeps it so unity bands stay transparent (the DC bin spans 0-62 Hz here)."""
    n = frames.shape[-1]
    fb = torch.fft.rfftfreq(n, 1.0 / SR).to(frames.device)
    curve = band_interp(bands, fb.expand(*bands.shape[:-1], fb.shape[0]))
    if dc_zero:
        curve = curve * (fb > 0).float()
    return torch.fft.irfft(torch.fft.rfft(frames) * curve, n=n)


def noise_render(bands, gen=None):
    """NOISE: each hop j starts a 2-hop grain of uniform noise in [-1,1), shaped by frame j's band
    curve, then Hann-windowed and overlap-added."""
    B, T, _ = bands.shape
    n = 2 * HOP
    noise = torch.rand(B, T, n, device=bands.device, generator=gen) * 2 - 1
    return _ola(_band_gain(noise, bands, True) * _hann(n, bands.device), T * HOP)


def eq_apply(x, bands):
    """Short-time EQ: 2-hop Hann frames at 50% overlap, the frame starting at hop j uses frame j's
    band curve (the frame starting one hop before note-on uses frame 0), overlap-added."""
    B, S = x.shape
    T = bands.shape[1]
    n = 2 * HOP
    fr = F.pad(x, (HOP, HOP)).unfold(-1, n, HOP)[:, :T + 1]     # frame i starts at (i-1)*HOP
    bi = torch.cat([bands[:, :1], bands], 1)[:, :T + 1]
    return _ola(_band_gain(fr * _hann(n, x.device), bi, False), S, first=-1)


def grain_render(bank, source, weight, position, rate, gain):
    """GRAIN: bank [N,L]; source (long or float) / weight [B,T,G,2]; position/rate/gain [B,T,G],
    held per grain. Hop j starts one 2-hop Hann grain per layer reading
    w0*bank[s0] + w1*bank[s1] from position*L, advancing rate per sample, wrapping."""
    B, T, G = position.shape
    n = 2 * HOP
    L = bank.shape[1]
    i = torch.arange(n, device=position.device, dtype=torch.float32)
    rp = position.unsqueeze(-1) * L + i * rate.unsqueeze(-1)       # [B,T,G,n]
    rp = rp - (rp.detach() / L).floor() * L
    i0 = rp.floor().long().clamp(0, L - 1)
    fr = rp - rp.detach().floor()
    i1 = (i0 + 1) % L
    fr = fr.detach()          # read-position gradients through a buffer are huge and noisy; see METHOD.md s5
    src = torch.round(source.float()).long() if source.is_floating_point() else source
    valid = ((src >= 0) & (src < bank.shape[0])).float()
    src = src.clamp(0, bank.shape[0] - 1)
    flat = bank.reshape(-1)
    acc = 0.0
    for r in range(src.shape[-1]):
        base = src[..., r].unsqueeze(-1) * L
        v = flat[base + i0] * (1 - fr) + flat[base + i1] * fr
        acc = acc + (weight[..., r] * valid[..., r]).unsqueeze(-1) * v
    grains = (acc * gain.unsqueeze(-1)).sum(2) * _hann(n, position.device)
    return _ola(grains, T * HOP)


def sampler_render(sample, root_hz, f0, rate, start, gain, eq, on_frame):
    """SAMPLER: reads sample [B,Ls] at (f0/root) * rate (f0 linear, rate held per hop) from
    start * Ls (start from the note-on frame), then EQ, then gain (linear)."""
    B, T = f0.shape
    S = T * HOP
    Ls = sample.shape[1]
    r = frame_interp(f0, S) / root_hz.view(B, 1) * frame_hold(rate, S)
    on = on_frame * HOP
    active = (torch.arange(S, device=f0.device) >= on).float().view(1, S)
    p = torch.cumsum(r * active, 1) - r * active + start[:, on_frame].view(B, 1) * Ls
    i0 = p.floor().long().clamp(0, Ls - 2)
    fr = (p - p.floor()).detach()   # as in grain_render
    x = torch.gather(sample, 1, i0) * (1 - fr) + torch.gather(sample, 1, i0 + 1) * fr
    x = x * active * (p < Ls - 1).float()
    return eq_apply(x, eq) * frame_interp(gain, S)


def attack_env(tau, on_frame, S):
    """1 - exp(-t/tau), t from the note-on sample, tau from the note-on frame (>= 0.1 ms)."""
    B = tau.shape[0]
    t = (torch.arange(S, device=tau.device, dtype=torch.float32) - on_frame * HOP) / SR
    tt = tau[:, on_frame].view(B, 1).clamp(min=1e-4)
    return torch.where(t >= 0, 1 - torch.exp(-t.clamp(min=0) / tt), torch.zeros(B, S, device=tau.device))


def safety(x):
    """Synth output safety stage: exactly linear to +6 dBFS (2.0), then a tanh knee that never
    passes +12 dBFS (4.0). Part of the instrument, so training renders through it."""
    a = x.abs()
    return torch.where(a <= SAFETY_KNEE, x,
                       torch.sign(x) * (SAFETY_KNEE + SAFETY_KNEE * torch.tanh((a - SAFETY_KNEE) / SAFETY_KNEE)))


def render(ctrl, f0, on_frame, ctx):
    """One voice through an otherwise empty Synth: safety(dry_gain * attack(t) * sum_p mix[p] *
    primitive_p). Mono: pan is centre (unity on both sides) and no slot is running (the slot
    families and modulation are not rendered here yet; METHOD.md s2). ctx: grain bank, sample, gen."""
    S = f0.shape[1] * HOP
    gen = ctx.get("gen")
    mix = frame_interp(ctrl["mix"], S)
    y = torch.zeros(f0.shape[0], S, device=f0.device)
    if "osc.re" in ctrl:
        y = y + mix[..., 0] * osc_render(ctrl["osc.re"], ctrl["osc.im"], f0,
                                         ctrl.get("osc.detune_ratio"), ctrl.get("osc.detune_level"))
    if "partials.ratio" in ctrl:
        y = y + mix[..., 1] * partials_render(ctrl["partials.ratio"], ctrl["partials.amp"], f0)
    if "fm.ratio" in ctrl:
        y = y + mix[..., 2] * fm_render(ctrl["fm.ratio"], ctrl["fm.level"], ctrl["fm.index"], f0)
    if "noise.band" in ctrl:
        y = y + mix[..., 3] * noise_render(ctrl["noise.band"], gen)
    if "grain.source" in ctrl:
        y = y + mix[..., 4] * grain_render(ctx["grain_bank"], ctrl["grain.source"], ctrl["grain.weight"],
                                           ctrl["grain.position"], ctrl["grain.rate"], ctrl["grain.gain"])
    if "sampler.rate" in ctrl:
        y = y + mix[..., 5] * sampler_render(ctx["sample"], ctx["sample_root"], f0, ctrl["sampler.rate"],
                                             ctrl["sampler.start"], ctrl["sampler.gain"], ctrl["sampler.eq"], on_frame)
    y = y * attack_env(ctrl["attack_tau"], on_frame, S)
    if "dry_gain" in ctrl:
        y = y * frame_interp(ctrl["dry_gain"], S)
    return safety(y)


def merge_hybrid(ca, cb, xf):
    """Merge two component frames into one VoiceFrame with crossfade xf (weight of A). OSC spectra
    add (same f0), NOISE bands add by power, any other primitive belongs to one component and
    takes that component's weighted mix. The inference host applies the same rule."""
    wa, wb = xf.unsqueeze(-1), (1 - xf).unsqueeze(-1)
    out = {"attack_tau": torch.minimum(ca["attack_tau"], cb["attack_tau"])}
    mix = wa * ca["mix"] + wb * cb["mix"]
    if "osc.re" in ca or "osc.re" in cb:
        re = im = 0.0
        for c, w in ((ca, wa), (cb, wb)):
            if "osc.re" in c:
                g = w * c["mix"][..., 0:1]
                re, im = re + g * c["osc.re"], im + g * c["osc.im"]
        out["osc.re"], out["osc.im"] = re, im
        mix[..., 0] = 1.0
        # Unison reads act on the merged table, so one set serves both components: the OSC-weighted
        # mean of their ratios and levels (a component without unison counts as ratio 1, level 0).
        if "osc.detune_level" in ca or "osc.detune_level" in cb:
            R = max(c["osc.detune_level"].shape[-1] for c in (ca, cb) if "osc.detune_level" in c)
            num_r = num_l = den = 0.0
            for c, w in ((ca, wa), (cb, wb)):
                if "osc.re" in c:
                    g = w * c["mix"][..., 0:1]
                    lv = c.get("osc.detune_level", torch.zeros_like(g).expand(*g.shape[:-1], R))
                    rt = c.get("osc.detune_ratio", torch.ones_like(lv))
                    num_r, num_l, den = num_r + g * lv * rt, num_l + g * lv, den + g
            out["osc.detune_level"] = num_l / (den + 1e-6)
            out["osc.detune_ratio"] = torch.where(num_l > 1e-9, num_r / (num_l + 1e-12), torch.ones_like(num_l))
    if "noise.band" in ca or "noise.band" in cb:
        pw = 0.0
        for c, w in ((ca, wa), (cb, wb)):
            if "noise.band" in c:
                pw = pw + (w * c["mix"][..., 3:4] * c["noise.band"]) ** 2
        out["noise.band"] = torch.sqrt(pw + 1e-12)
        mix[..., 3] = 1.0
    for c in (ca, cb):
        for k, v in c.items():
            if k not in out and k != "mix":
                if k in (ca.keys() & cb.keys()):
                    raise ValueError(f"hybrid components share {k}")
                out[k] = v
    out["mix"] = mix
    return out


# ================================================================ builders (METHOD.md s3)
# Fixed differentiable math shared by the teachers and the network (exported inside the expert
# graphs). Per frame: inputs [..., D] with any leading dims, f0 [...]. Output keys are
# VoiceFrame field names; arrays are tier-sized prefixes of the VoiceFrame arrays.

def _filter_mag(f, cutoff, q, mode):
    """4-pole state-variable magnitude: a resonant 2-pole stage (Q = q) times a Butterworth 2-pole
    stage (Q = 0.707) for LP and HP, and a resonant BP, blended by mode [...,3]. The resonant peak
    is about q, not q^2, like a ladder filter."""
    x = f / cutoff.unsqueeze(-1)
    qq = q.unsqueeze(-1)
    den = torch.sqrt((1 - x * x) ** 2 + (x / qq) ** 2 + 1e-12)
    den2 = torch.sqrt((1 - x * x) ** 2 + 2 * x * x + 1e-12)
    lp, bp, hp = 1 / den, (x / qq) / den, x * x / den
    return mode[..., 0:1] * lp / den2 + mode[..., 1:2] * bp + mode[..., 2:3] * hp * x * x / den2


def _pad_h(x):
    return F.pad(x, (0, N_HARM - x.shape[-1]))


def _tail(amps, slope_db, oddeven):
    """Extend explicit harmonic amps [...,K] to N_HARM with a dB/octave slope and even-harmonic
    attenuation (oddeven=1 removes even harmonics above K)."""
    K = amps.shape[-1]
    k = harm_k(amps.device)[K:]
    ref = amps[..., K // 2:].mean(-1, keepdim=True)
    tail = ref * (k / K) ** (slope_db.unsqueeze(-1) / 6.0206)
    tail = tail * (1 - oddeven.unsqueeze(-1) * (k % 2 == 0).float())
    return torch.cat([amps, tail], -1)


def _unison(p, out):
    """OSC unison reads (layout v2) for the methods that play OSC tables."""
    if "det_level" in p:
        out["osc.detune_ratio"], out["osc.detune_level"] = p["det_ratio"], p["det_level"]
    return out


def build_subtractive(p, f0):
    k = harm_k(f0.device)
    sgn = torch.where(k % 2 == 1, 1.0, -1.0)
    saw = (2 / math.pi) * sgn / k
    tri = (8 / math.pi ** 2) * (k % 2 == 1).float() * torch.where(((k - 1) / 2) % 2 == 0, 1.0, -1.0) / k ** 2
    pulse = (2 / (math.pi * k)) * torch.sin(math.pi * k * p["pw"])
    mix = p["mix"]
    im = -(mix[..., 0:1] * saw + mix[..., 2:3] * tri)
    re = mix[..., 1:2] * pulse
    cut, q, mode = p["cutoff"][..., 0], p["q"][..., 0], p["mode"]
    H = _filter_mag(f0.unsqueeze(-1) * k, cut, q, mode) * p["amp"]
    Hn = _filter_mag(band_hz(f0.device).expand(*f0.shape, N_BANDS), cut, q, mode)
    return _unison(p, {"mix": mix_vec(f0, "osc", "noise"), "attack_tau": p["attack"][..., 0],
                       "osc.re": re * H, "osc.im": im * H, "noise.band": Hn * p["noise"] * p["amp"]})


def build_wavetable(p, f0):
    K = p["wt_re"].shape[-1]
    g = harm_k(f0.device, K) ** (p["tilt"] / 6.0206) * p["amp"]
    return _unison(p, {"mix": mix_vec(f0, "osc"), "attack_tau": p["attack"][..., 0],
                       "osc.re": _pad_h(p["wt_re"] * g), "osc.im": _pad_h(p["wt_im"] * g)})


def build_additive(p, f0):
    """Additive is its own family on PARTIALS: M partials at k*sqrt(1 + B k^2) * (1 + detune_k)."""
    M = p["harm"].shape[-1]
    k = harm_k(f0.device, M)
    ratio = k * torch.sqrt(1 + p["stretch"] * k * k) * (1 + p["detune"])
    return {"mix": mix_vec(f0, "partials"), "attack_tau": p["attack"][..., 0],
            "partials.ratio": ratio, "partials.amp": p["harm"] * p["amp"]}


def build_spectral(p, f0):
    a = band_interp(p["env"], f0.unsqueeze(-1) * harm_k(f0.device)) * p["balance"] * p["amp"]
    return {"mix": mix_vec(f0, "osc", "noise"), "attack_tau": p["attack"][..., 0],
            "osc.re": torch.zeros_like(a), "osc.im": -a,
            "noise.band": p["env"] * (1 - p["balance"]) * p["amp"] * 0.1}


def build_fm(p, f0):
    N = p["ratio"].shape[-1]
    triu = torch.triu(torch.ones(N, N, device=f0.device), 1)
    return {"mix": mix_vec(f0, "fm"), "attack_tau": p["attack"][..., 0], "fm.ratio": p["ratio"],
            "fm.level": p["level"] * p["amp"],
            "fm.index": p["index"].reshape(*p["index"].shape[:-1], N, N) * triu}


def build_physical(p, f0, state=None):
    """Modal bank. The per-mode amplitude integrator a_t = a_{t-1} exp(-1/(tau*250)) + exc_t runs
    here, inside the graph, so PARTIALS only ever sees amplitudes. Training passes whole clips
    [B,T,M] (state=None); the exported step graph passes and returns the state [V,M]."""
    d = torch.exp(-1.0 / (p["tau"].clamp(min=1e-3) * FRAME_HZ))
    if state is None:
        a = torch.zeros_like(p["exc"][:, 0])
        amps = []
        for t in range(p["exc"].shape[1]):
            a = a * d[:, t] + p["exc"][:, t]
            amps.append(a)
        amp, new_state = torch.stack(amps, 1), None
    else:
        amp = new_state = state * d + p["exc"]
    out = {"mix": mix_vec(f0, "partials", "noise"), "attack_tau": p["attack"][..., 0],
           "partials.ratio": p["ratio"], "partials.amp": amp, "noise.band": p["breath"]}
    return out if state is None else (out, new_state)


def build_granular(p, f0):
    pitched = p["pitched"]
    rate = pitched * (f0.unsqueeze(-1) / SAMPLE_ROOT_HZ) * p["ratio"] + (1 - pitched) * p["ratio"]
    return {"mix": mix_vec(f0, "grain"), "attack_tau": p["attack"][..., 0], "grain.source": p["idx"],
            "grain.weight": p["w"], "grain.position": p["pos"], "grain.rate": rate, "grain.gain": p["gain"]}


def build_sampling(p, f0):
    r = _pad_h(p["resid"])
    return {"mix": mix_vec(f0, "osc", "sampler"), "attack_tau": p["attack"][..., 0],
            "sampler.rate": p["rate"][..., 0], "sampler.start": p["start"][..., 0],
            "sampler.gain": p["gain"][..., 0], "sampler.eq": p["eq"],
            "osc.re": torch.zeros_like(r), "osc.im": -r}


BUILD = {"subtractive": build_subtractive, "wavetable": build_wavetable, "additive": build_additive,
         "spectral": build_spectral, "fm": build_fm, "physical": build_physical,
         "granular": build_granular, "sampling": build_sampling}


def to_voiceframe(ctrl):
    """Pack a control dict [B,T,...] into flat VoiceFrames [B,T,VF_FLOATS] (ControlFrame.h)."""
    B, T = ctrl["mix"].shape[:2]
    out = torch.zeros(B, T, VF_FLOATS)
    for k, v in VF_DEFAULTS.items():
        o, c = VF_OFFSET[k]
        out[..., o:o + c] = v
    for k, v in ctrl.items():
        v = v.detach().float().cpu()
        if k == "fm.index":
            N = v.shape[-1]
            v = F.pad(v, (0, 8 - N, 0, 8 - N))
        v = v.reshape(B, T, -1)
        o, c = VF_OFFSET[k]
        assert v.shape[-1] <= c, k
        out[..., o:o + v.shape[-1]] = v
    return out


def check_layout(path):
    """Compare VF_FIELDS / IF_FIELDS / tier limits with the DSP's exported src/dsp/layout.json.
    Returns a list of mismatches (empty when in sync)."""
    with open(path) as f:
        L = json.load(f)
    bad = []
    if L.get("layout_version") != LAYOUT_VERSION:
        bad.append(f"layout_version {L.get('layout_version')} != {LAYOUT_VERSION}")
    for key, ours, n in (("fields", VF_FIELDS, VF_FLOATS), ("instance_fields", IF_FIELDS, IF_FLOATS)):
        off, o = {}, 0
        for name, c in ours:
            off[name] = (o, c)
            o += c
        for fld in L.get(key, []):
            if off.get(fld["name"]) != (fld["offset"], fld["count"]):
                bad.append(f"{key}.{fld['name']}: DSP {(fld['offset'], fld['count'])}, training {off.get(fld['name'])}")
    if L.get("frame_floats") != VF_FLOATS or L.get("instance_frame_floats") != IF_FLOATS:
        bad.append(f"frame sizes DSP {L.get('frame_floats')}/{L.get('instance_frame_floats')}, "
                   f"training {VF_FLOATS}/{IF_FLOATS}")
    for name, lim in L.get("tiers", {}).items():
        t = TIERS.get(name)
        ours = t and {"partials": t.n_modes, "fm_operators": t.n_ops, "grain_layers": t.n_grains, "osc_unison": t.unison}
        for k, v in (ours or {}).items():
            if lim.get(k) != v:
                bad.append(f"tiers.{name}.{k}: DSP {lim.get(k)}, training {v}")
    return bad


# ================================================================ control distance (METHOD.md s5)

def _l1log(a, b, eps=1e-4):
    return (torch.log(a.abs() + eps) - torch.log(b.abs() + eps)).abs().mean()


def ctrl_distance(p, t):
    d = (p["mix"] - t["mix"]).abs().mean()
    if "osc.re" in p and "osc.re" in t:
        mp = torch.sqrt(p["osc.re"] ** 2 + p["osc.im"] ** 2 + 1e-12) * p["mix"][..., 0:1]
        mt = torch.sqrt(t["osc.re"] ** 2 + t["osc.im"] ** 2 + 1e-12) * t["mix"][..., 0:1]
        w = 1.0 / harm_k(mp.device)
        d = d + ((torch.log(mp + 1e-4) - torch.log(mt + 1e-4)).abs() * w).sum(-1).mean() / w.sum()
    for key in ("noise.band", "sampler.eq", "fm.level", "partials.amp", "grain.gain", "sampler.gain"):
        if key in p and key in t:
            d = d + _l1log(p[key], t[key])
    for key in ("fm.ratio", "partials.ratio", "grain.rate", "sampler.rate", "attack_tau"):
        if key in p and key in t:
            d = d + (torch.log(p[key].clamp(min=1e-5)) - torch.log(t[key].clamp(min=1e-5))).abs().mean()
    if "fm.index" in p and "fm.index" in t:
        d = d + (p["fm.index"] - t["fm.index"]).abs().mean() / 4
    if "osc.detune_level" in p and "osc.detune_level" in t:
        d = d + _l1log(p["osc.detune_level"], t["osc.detune_level"])
        cents = 1200 * (torch.log2(p["osc.detune_ratio"]) - torch.log2(t["osc.detune_ratio"])).abs()
        d = d + (cents * t["osc.detune_level"]).mean() / 10          # only where the teacher read sounds
    return d


# ================================================================ losses

def mrstft(x, y, ffts=(2048, 1024, 512, 256, 128, 64), lowpass_hz=None):
    loss = 0.0
    for n in ffts:
        win = torch.hann_window(n, device=x.device)
        X = torch.stft(x, n, n // 4, window=win, return_complex=True).abs()
        Y = torch.stft(y, n, n // 4, window=win, return_complex=True).abs()
        if lowpass_hz is not None:
            keep = int(n * lowpass_hz / SR) + 1
            X, Y = X[:, :keep], Y[:, :keep]
        sc = (X - Y).abs().sum() / (Y.abs().sum() + 1e-7)
        lg = (torch.log(X + 1e-5) - torch.log(Y + 1e-5)).abs().mean()
        loss = loss + sc + lg
    return loss / len(ffts)


_MEL = {}


def logmel(x, n_mels=64):
    """Log-mel [B, n_mels, frames] of 48 kHz audio: n_fft 2048, hop 512, Hann, 20 Hz .. 16 kHz,
    triangular filters on the HTK mel scale. The plugin computes the same for audio_encoder."""
    dev = x.device
    if dev not in _MEL:
        hz = lambda m: 700 * (10 ** (m / 2595) - 1)
        mel = lambda f: 2595 * math.log10(1 + f / 700)
        pts = torch.tensor([hz(m) for m in np.linspace(mel(20), mel(16000), n_mels + 2)])
        fb = torch.fft.rfftfreq(2048, 1 / SR)
        W = torch.zeros(n_mels, fb.shape[0])
        for i in range(n_mels):
            lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
            W[i] = torch.clamp(torch.minimum((fb - lo) / (c - lo), (hi - fb) / (hi - c)), min=0)
        _MEL[dev] = W.to(dev)
    X = torch.stft(x, 2048, 512, window=torch.hann_window(2048, device=dev), return_complex=True).abs()
    return torch.log(_MEL[dev] @ X + 1e-5)


# ================================================================ performance + teachers (s4)

def make_perf(B, T, dev, g, pitch=None, vel=None, length=None, expressive=True):
    """Random single-note performance. Returns f0, note features and envelope timing."""
    r = lambda *s: torch.rand(*s, device=dev, generator=g)
    pitch = pitch if pitch is not None else (28 + (96 - 28) * r(B)).round()
    vel = vel if vel is not None else 0.2 + 0.8 * r(B)
    nlen = length if length is not None else ((0.30 + 0.55 * r(B)) * T).round()
    off = ON_FRAME + nlen
    t = torch.arange(T, device=dev, dtype=torch.float32).view(1, T)
    started = (t >= ON_FRAME).float().expand(B, T)
    released = (t >= off.view(B, 1)).float()
    gate = started * (1 - released)
    t_on = ((t - ON_FRAME) / FRAME_HZ).clamp(min=0) * started
    t_off = ((t - off.view(B, 1)) / FRAME_HZ).clamp(min=0) * released
    if expressive:
        m0, m1 = r(B, 1) * (r(B, 1) < 0.5).float(), r(B, 1)
        mod = m0 + (m1 * (r(B, 1) < 0.3).float() - m0) * t / T
        at = gate * r(B, 1) * (r(B, 1) < 0.3).float()
        bend = (r(B, 1) < 0.15).float() * torch.sin(2 * math.pi * t / T * (1 + 3 * r(B, 1))) * r(B, 1) * 0.5
    else:
        mod = at = bend = torch.zeros(B, T, device=dev)
    mod, at, bend = mod.expand(B, T), at.expand(B, T), bend.expand(B, T)
    f0 = midi_hz(pitch.view(B, 1) + 2 * bend)
    lg = math.log1p(MAX_T_S)
    feats = torch.stack([pitch.view(B, 1).expand(B, T) / 127, vel.view(B, 1).expand(B, T), gate,
                         torch.log1p(t_on) / lg, torch.log1p(t_off) / lg, mod, at, bend], -1)
    glob = torch.stack([mod, at, bend, torch.zeros_like(mod)], -1)
    return {"f0": f0, "feats": feats, "glob": glob, "gate": gate, "started": started,
            "released": released, "t_on": t_on, "t_off": t_off, "len_s": nlen / FRAME_HZ,
            "vel": vel, "pitch": pitch, "mod": mod, "at": at}


def adsr(perf, A, D, S, R):
    """Frame-rate ADSR; A,D,S,R [B] (seconds, sustain level)."""
    A, D, S, R = (v.view(-1, 1) for v in (A, D, S, R))

    def pre(t):
        return torch.where(t < A, t / A, S + (1 - S) * torch.exp(-(t - A) / D * 4))

    at_off = pre(perf["len_s"].view(-1, 1))
    rel = at_off * torch.exp(-perf["t_off"] / R * 4)
    env = torch.where(perf["released"] > 0, rel, pre(perf["t_on"]))
    return env * perf["started"]


def _adsr_knobs(perf, k, i):
    A = 0.001 * 2 ** (k[:, i] * 11)
    D = 0.01 * 2 ** (k[:, i + 1] * 9)
    R = 0.01 * 2 ** (k[:, i + 3] * 9)
    return adsr(perf, A, D, k[:, i + 2], R), A


def _attack(A, T):
    return (A / 4).clamp(3e-4, 0.08).view(-1, 1, 1).expand(-1, T, 1)


def interp_frames(tab, pos):
    """tab [...,F,K,2] frames; pos [...] in [0,1] -> [...,K,2] linear between frames."""
    Fr = tab.shape[-3]
    u = pos.clamp(0, 1) * (Fr - 1)
    i0 = u.floor().long().clamp(max=Fr - 2)
    fr = (u - i0.float()).unsqueeze(-1).unsqueeze(-1)
    idx = i0.view(*i0.shape, 1, 1, 1).expand(*i0.shape, 1, *tab.shape[-2:])
    f0_ = torch.gather(tab, -3, idx).squeeze(-3)
    f1_ = torch.gather(tab, -3, idx + 1).squeeze(-3)
    return f0_ + (f1_ - f0_) * fr


def _teacher_unison(k, B, T, R, dev):
    """Supersaw-style unison from knobs 22 (amount; below 0.5 none) and 23 (spread 3..50 cents):
    n reads of the R available at level 0.8, spread symmetrically around the main read. Returns
    ratio, level [B,T,R] and the gain that keeps the summed level near one read's."""
    n_on = torch.round((k[:, 22] * 2 - 1).clamp(0, 1) * R).view(B, 1)
    r = torch.arange(R, device=dev).float().view(1, R)
    pairs = torch.ceil(n_on / 2).clamp(min=1)
    cents = (3 + 47 * k[:, 23:24] ** 2) * (torch.div(r, 2, rounding_mode="floor") + 1) / pairs
    cents = torch.where(r % 2 == 0, cents, -cents)
    on = (r < n_on).float()
    ratio = torch.where(on > 0, 2 ** (cents / 1200), torch.ones_like(cents))
    level = 0.8 * on
    gain = 1 / torch.sqrt(1 + (level ** 2).sum(-1))
    return ratio.view(B, 1, R).expand(B, T, R), level.view(B, 1, R).expand(B, T, R), gain


class Teachers:
    """Procedural classic synth per method: 32 knobs + performance -> method params. Factory
    content (wavetables, grain sources, mode sets, samples) is generated from a fixed seed."""

    BEAM = [1.0, 2.756, 5.404, 8.933, 13.34, 18.64, 24.81, 31.87]
    MEMBRANE = [1.0, 1.594, 2.136, 2.296, 2.653, 2.918, 3.156, 3.501, 3.600, 3.652, 4.060, 4.154,
                4.230, 4.601, 4.832, 4.903]

    def __init__(self, tier, dev, seed=1234):
        self.t, self.dev = tier, dev
        g = torch.Generator(device="cpu").manual_seed(seed)
        H = TIERS["studio"].K_h                                   # factory tables at full width
        k = torch.arange(1, H + 1).float()
        tabs = []
        for _ in range(64):                                       # 64 factory wavetables x 32 frames
            a = k ** -(0.5 + 1.5 * torch.rand(1, generator=g)) * (0.3 + torch.rand(2, H, generator=g))
            fc = 2 ** (torch.rand(2, 1, generator=g) * 6) * 2
            a = a * (1 + 3 * torch.exp(-((torch.log2(k) - torch.log2(fc)) ** 2) * 2))
            ph = torch.rand(H, generator=g) * 2 * math.pi * (torch.rand(1, generator=g) < 0.5).float()
            w = torch.linspace(0, 1, 32).view(32, 1)
            mag = a[0] * (1 - w) + a[1] * w
            mag = mag / mag.sum(-1, keepdim=True) * 2                 # peak level near 1 before amp
            tabs.append(torch.stack([mag * torch.cos(ph), mag * torch.sin(ph)], -1))
        self.wt = torch.stack(tabs).to(dev)                       # [64,32,H,2]
        src = []
        for i in range(16):                                       # grain sources, rooted at C4
            n = torch.arange(GRAIN_LEN).float() / SR
            kind = i % 4
            if kind == 0:
                x = torch.randn(GRAIN_LEN, generator=g)
                roll = torch.exp(-torch.linspace(0, 6 + 4 * torch.rand(1, generator=g).item(), GRAIN_LEN // 2 + 1))
                x = torch.fft.irfft(torch.fft.rfft(x) * roll, n=GRAIN_LEN)
            elif kind == 1:
                x = sum(torch.sin(2 * math.pi * SAMPLE_ROOT_HZ * h * n) / h for h in range(1, 12))
            elif kind == 2:
                x = torch.sin(2 * math.pi * (100 + 4000 * n / n[-1]) * n * (1 + torch.rand(1, generator=g)))
            else:
                x = torch.sin(2 * math.pi * SAMPLE_ROOT_HZ * n + 3 * torch.sin(2 * math.pi * SAMPLE_ROOT_HZ * 1.41 * n))
            src.append(x / (x.abs().max() + 1e-6) * 0.8)
        self.grain_bank = torch.stack(src).to(dev)
        plate = 1 + torch.cumsum(torch.rand(8, 63, generator=g) * 0.9 + 0.1, 1)
        self.plate = torch.cat([torch.ones(8, 1), plate], 1).to(dev)
        self.samples = None                                       # built lazily (needs render)
        self.gen = torch.Generator(device=dev).manual_seed(seed + 1)

    def _factory_samples(self):
        """8 factory samples at C4, 1.5 s, rendered from other teachers with fixed knobs."""
        g = torch.Generator(device=self.dev).manual_seed(99)
        T = int(1.5 * FRAME_HZ)
        outs = []
        for m in ["physical", "fm", "subtractive", "additive"] * 2:
            k = torch.rand(1, N_KNOBS, device=self.dev, generator=g)
            perf = make_perf(1, T, self.dev, g, pitch=torch.tensor([60.0], device=self.dev),
                             vel=torch.tensor([0.8], device=self.dev),
                             length=torch.tensor([T * 0.7], device=self.dev), expressive=False)
            ctrl = BUILD[m](getattr(self, m)(k, perf), perf["f0"])
            y = render(ctrl, perf["f0"], ON_FRAME, {"gen": g})[:, ON_FRAME * HOP:]
            outs.append(y / (y.abs().max() + 1e-6) * 0.8)
        L = min(o.shape[1] for o in outs)
        self.samples = torch.cat([o[:, :L] for o in outs]).detach()

    # -- teacher programs: knobs k [B,32] in [0,1], perf -> method params [B,T,...]
    def subtractive(self, k, perf):
        B, T = perf["f0"].shape
        e = lambda v: v.view(B, 1, 1).expand(B, T, 1)
        mix = k[:, 0:3] ** 2
        mix = mix / (mix.sum(-1, keepdim=True) + 1e-6)
        fenv, _ = _adsr_knobs(perf, k, 10)
        aenv, A = _adsr_knobs(perf, k, 18)
        pitch = perf["pitch"].view(B, 1)
        cut = 30 * 2 ** (k[:, 6:7] * 9.5) * 2 ** ((k[:, 7:8] * 2 - 1) * 6 * fenv + k[:, 8:9] * (pitch - 60) / 12
                                                  + k[:, 9:10] * perf["vel"].view(B, 1) * 2 + perf["mod"] * k[:, 14:15] * 3)
        m = k[:, 16]
        lp, hp = (1 - 2 * m).clamp(0, 1), (2 * m - 1).clamp(0, 1)
        mode = torch.stack([lp, 1 - lp - hp, hp], -1).view(B, 1, 3).expand(B, T, 3)
        q = 0.5 + 19.5 * k[:, 15] ** 2
        dr, dl, ug = _teacher_unison(k, B, T, self.t.unison - 1, self.dev)
        return {"mix": mix.view(B, 1, 3).expand(B, T, 3), "pw": e(0.05 + 0.9 * k[:, 3]),
                "cutoff": cut.clamp(20, 20000).unsqueeze(-1), "q": e(q), "mode": mode,
                "noise": e(0.5 * k[:, 17] ** 2),
                "amp": (aenv * (0.3 + 0.7 * perf["vel"].view(B, 1)) * 0.5 * ug.view(B, 1) / q.sqrt().view(B, 1)).unsqueeze(-1),
                "attack": _attack(A, T), "det_ratio": dr, "det_level": dl}

    def wavetable(self, k, perf):
        B, T = perf["f0"].shape
        e = lambda v: v.view(B, 1, 1).expand(B, T, 1)
        wenv, _ = _adsr_knobs(perf, k, 4)
        aenv, A = _adsr_knobs(perf, k, 18)
        tab = self.wt[(k[:, 0] * 63.999).long()][:, :, :self.t.K_h]   # [B,32,K,2]
        pos = (k[:, 1:2] + (k[:, 2:3] * 2 - 1) * wenv + k[:, 3:4] * perf["mod"]).clamp(0, 1)
        fr = interp_frames(tab.unsqueeze(1).expand(B, T, *tab.shape[1:]), pos)
        tilt = (k[:, 8] * 2 - 1) * 4
        mag = tab.norm(dim=-1).mean(1)                                      # [B,K]
        kk = torch.arange(1, mag.shape[-1] + 1, device=self.dev).float()
        comp = mag.sum(-1) / (mag * kk ** (tilt.view(B, 1) / 6.0206)).sum(-1)  # keep level under tilt
        dr, dl, ug = _teacher_unison(k, B, T, self.t.unison - 1, self.dev)
        return {"wt_re": fr[..., 0], "wt_im": fr[..., 1], "tilt": e(tilt),
                "amp": (aenv * (0.3 + 0.7 * perf["vel"].view(B, 1)) * 0.5 * (comp * ug).view(B, 1)).unsqueeze(-1),
                "attack": _attack(A, T), "det_ratio": dr, "det_level": dl}

    def additive(self, k, perf):
        B, T = perf["f0"].shape
        e = lambda v: v.view(B, 1, 1).expand(B, T, 1)
        K = self.t.n_modes
        kk = torch.arange(1, K + 1, device=self.dev).float().view(1, 1, K)
        roll = (0.5 + 2.5 * k[:, 0]).view(B, 1, 1)
        even = (kk % 2 == 0).float()
        fc = (200 * 2 ** (k[:, 2] * 6)).view(B, 1, 1)
        fh = perf["f0"].unsqueeze(-1) * kk
        bump = torch.exp(-((torch.log2(fh) - torch.log2(fc)) ** 2) / (0.1 + k[:, 3].view(B, 1, 1)))
        a = kk ** -roll * (1 - k[:, 1].view(B, 1, 1) * even) * (1 + 4 * k[:, 4].view(B, 1, 1) * bump)
        a = a * torch.exp(-perf["t_on"].unsqueeze(-1) * k[:, 5].view(B, 1, 1) * 8 * kk / K)
        aenv, A = _adsr_knobs(perf, k, 18)
        stretch = (k[:, 6] > 0.6).float() * 1e-4 * 10 ** (k[:, 7] * 2)      # piano-like stretch on some patches
        detune = (torch.rand(B, 1, K, device=self.dev, generator=self.gen) * 2 - 1) * k[:, 8].view(B, 1, 1) * 0.003
        a = a / (a.sum(-1, keepdim=True) + 1e-6) * 0.6                 # peak level near 0.6
        return {"harm": a, "stretch": e(stretch), "detune": detune.expand(B, T, K),
                "amp": (aenv * (0.3 + 0.7 * perf["vel"].view(B, 1))).unsqueeze(-1),
                "attack": _attack(A, T)}

    def spectral(self, k, perf):
        B, T = perf["f0"].shape
        e = lambda v: v.view(B, 1, 1).expand(B, T, 1)
        lf = torch.log2(band_hz(self.dev)).view(1, 1, N_BANDS)
        lfo = 0.5 + 0.5 * torch.sin(2 * math.pi * perf["t_on"] * (0.2 + 4 * k[:, 7:8]))
        mv = (lfo * k[:, 8:9] + perf["mod"]).clamp(0, 1).unsqueeze(-1)
        env = 0.02 * torch.ones(B, T, N_BANDS, device=self.dev)
        for i in range(3):
            c0 = 7 + 7 * k[:, i].view(B, 1, 1)
            c1 = 7 + 7 * k[:, 3 + i].view(B, 1, 1)
            c = c0 * (1 - mv) + c1 * mv
            env = env + torch.exp(-((lf - c) ** 2) / 0.08) / (i + 1)
        aenv, A = _adsr_knobs(perf, k, 18)
        fh = perf["f0"].unsqueeze(-1) * harm_k(self.dev)
        n_eff = (band_interp(env, fh) * (fh < SR / 2).float()).sum(-1, keepdim=True).clamp(min=1.0)
        return {"env": env, "balance": e(0.3 + 0.7 * k[:, 6]),
                "amp": (aenv * (0.3 + 0.7 * perf["vel"].view(B, 1))).unsqueeze(-1) * 0.4 / n_eff,
                "attack": _attack(A, T)}

    def fm(self, k, perf):
        B, T = perf["f0"].shape
        N = self.t.n_ops
        algo = (k[:, 0] * 3.999).long()                               # stack, pairs, one-to-many, additive
        ratio = torch.round(1 + k[:, 1:1 + N] * 7) * torch.where(k[:, 9:9 + N] < 0.15, 0.5, 1.0)
        ratio = ratio * (1 + (k[:, 17:18] - 0.5) * 0.004)
        ratio[:, 0] = 1.0
        idx = torch.zeros(B, N, N, device=self.dev)
        for i in range(N - 1):
            stack = (algo == 0).float()
            pairs = ((algo == 1) & (i % 2 == 0)).float()
            idx[:, i, i + 1] = stack + pairs
            idx[:, i, N - 1] = torch.maximum(idx[:, i, N - 1], (algo == 2).float())
        carrier = torch.where(idx.sum(1) > 0, 0.0, 1.0)             # modulators are silent
        carrier = carrier / carrier.sum(-1, keepdim=True).clamp(min=1)
        ienv, _ = _adsr_knobs(perf, k, 10)
        aenv, A = _adsr_knobs(perf, k, 18)
        depth = (k[:, 14] * 8).view(B, 1) * ienv * (0.5 + 0.5 * perf["vel"].view(B, 1)) + perf["mod"] * 2 * k[:, 15:16]
        index = idx.view(B, 1, N, N) * depth.view(B, T, 1, 1)
        return {"ratio": ratio.view(B, 1, N).expand(B, T, N), "level": carrier.view(B, 1, N).expand(B, T, N),
                "index": index.reshape(B, T, N * N),
                "amp": (aenv * (0.3 + 0.7 * perf["vel"].view(B, 1)) * 0.5).unsqueeze(-1),
                "attack": _attack(A, T)}

    def physical(self, k, perf):
        B, T = perf["f0"].shape
        M = self.t.n_modes
        dev = self.dev
        m = torch.arange(1, M + 1, device=dev).float().view(1, M)
        kind = (k[:, 0] * 4.999).long().view(B, 1)
        Bst = 1e-4 * 10 ** (k[:, 1:2] * 3)
        string = m * torch.sqrt(1 + Bst * m * m)
        beam = torch.tensor(self.BEAM, device=dev)
        beam = torch.cat([beam, ((2 * torch.arange(len(beam) + 1, M + 1, device=dev).float() + 1) / 3) ** 2])[:M].view(1, M)
        mem = torch.tensor(self.MEMBRANE, device=dev)
        mem = torch.cat([mem, mem[-1] + 0.3 * torch.arange(1, max(1, M - len(mem)) + 1, device=dev).float()])[:M].view(1, M)
        tube = 2 * m - 1
        plate = self.plate[(k[:, 1] * 7.999).long()][:, :M]
        ratio = torch.where(kind == 0, string, torch.where(kind == 1, beam.expand(B, M), torch.where(
            kind == 2, mem.expand(B, M), torch.where(kind == 3, tube, plate))))
        base = 0.05 * 2 ** (k[:, 2:3] * 7)
        tau = base / (1 + k[:, 3:4] * 20 * (ratio - 1) / ratio.amax(-1, keepdim=True))
        damp = torch.where(perf["released"] > 0, 0.02 + 0.1 * k[:, 4:5], torch.ones_like(perf["released"]))
        tau_t = tau.view(B, 1, M) * damp.unsqueeze(-1)
        x = 0.05 + 0.45 * k[:, 6:7]
        w = ratio ** (-2 * (1 - k[:, 5:6])) * torch.sin(math.pi * m * x).abs()
        w = w / (w.sum(-1, keepdim=True) + 1e-6) * 0.6
        cont = (k[:, 7:8] > 0.5).float()
        onset = torch.zeros(B, T, 1, device=dev)
        onset[:, ON_FRAME] = 1.0
        press = (perf["gate"] * (0.3 + 0.7 * perf["vel"].view(B, 1) + perf["mod"] * 0.5)).unsqueeze(-1)
        exc = w.view(B, 1, M) * perf["vel"].view(B, 1, 1) * (onset * (1 - cont.view(B, 1, 1))
                                                           + press * cont.view(B, 1, 1) / (tau_t * FRAME_HZ).clamp(min=1))
        f0 = perf["f0"]
        breath = (cont.view(B, 1, 1) * k[:, 8].view(B, 1, 1) * 0.05 * press
                  * torch.exp(-((torch.log2(band_hz(dev)).view(1, 1, -1) - torch.log2(f0 * 2).unsqueeze(-1)) ** 2) / 2))
        return {"ratio": ratio.view(B, 1, M).expand(B, T, M), "tau": tau_t, "exc": exc, "breath": breath,
                "attack": torch.full((B, T, 1), 3e-4, device=dev)}

    def granular(self, k, perf):
        B, T = perf["f0"].shape
        G = self.t.n_grains
        g = torch.arange(G, device=self.dev).float().view(1, 1, G)
        aenv, A = _adsr_knobs(perf, k, 18)
        src = ((k[:, 0].view(B, 1, 1) * 15.999).long() + g.long()) % 16
        idx = torch.stack([src, src], -1).expand(B, T, G, 2)
        w = torch.tensor([1.0, 0.0], device=self.dev).expand(B, T, G, 2)
        jitter = (torch.rand(B, T, G, device=self.dev, generator=self.gen) * 2 - 1) * k[:, 6].view(B, 1, 1) * 0.05
        pos = (k[:, 1].view(B, 1, 1) + g * k[:, 7].view(B, 1, 1) * 0.1
               + perf["t_on"].unsqueeze(-1) * k[:, 2].view(B, 1, 1) * 0.5 + jitter) % 1.0
        ratio = 2 ** ((k[:, 3].view(B, 1, 1) * 2 - 1) + g * k[:, 4].view(B, 1, 1) * 0.02)
        gain = (aenv * (0.3 + 0.7 * perf["vel"].view(B, 1))).unsqueeze(-1).expand(B, T, G) * 0.6 / G
        return {"idx": idx, "w": w, "pos": pos, "ratio": ratio.expand(B, T, G),
                "pitched": (k[:, 5] > 0.5).float().view(B, 1, 1).expand(B, T, 1),
                "gain": gain, "attack": _attack(A, T)}

    def sampling(self, k, perf):
        B, T = perf["f0"].shape
        e = lambda v: v.view(B, 1, 1).expand(B, T, 1)
        aenv, A = _adsr_knobs(perf, k, 18)
        lf = torch.log2(band_hz(self.dev)).view(1, 1, N_BANDS)
        tilt = ((k[:, 3] * 2 - 1) * 0.2).view(B, 1, 1) * (lf - 10) + (perf["vel"].view(B, 1, 1) - 0.7) * 0.15 * (lf - 10)
        bump = (k[:, 4] * 2 - 1).view(B, 1, 1) * 1.5 * torch.exp(-((lf - 7 - 7 * k[:, 5].view(B, 1, 1)) ** 2) / 0.2)
        eq = torch.exp(tilt + bump).expand(B, T, N_BANDS)
        return {"rate": e(1 + (k[:, 1] * 2 - 1) * 0.01), "start": e(k[:, 2] * 0.3), "gain": (aenv * 0.9).unsqueeze(-1),
                "eq": eq, "resid": torch.zeros(B, T, 16, device=self.dev), "attack": _attack(A, T)}

    # -- batches
    def batch(self, method, B, T, pair=None, k=None, perf=None):
        g = self.gen
        k = torch.rand(B, N_KNOBS, device=self.dev, generator=g) if k is None else k
        perf = make_perf(B, T, self.dev, g) if perf is None else perf
        mvec = torch.zeros(B, METHOD_IN, device=self.dev)
        mvec[:, M_IDX[method]] = 1
        ctx = {"grain_bank": self.grain_bank, "gen": g}
        if method == "sampling":
            if self.samples is None:
                self._factory_samples()
            si = (k[:, 0] * (self.samples.shape[0] - 0.001)).long()
            ctx["sample"] = self.samples[si]
            ctx["sample_root"] = torch.full((B,), SAMPLE_ROOT_HZ, device=self.dev)
        if method == "hybrid":
            a, b = pair
            mvec[:, len(METHODS) + M_IDX[a]] = 1
            mvec[:, 2 * len(METHODS) + M_IDX[b]] = 1
            ca = BUILD[a](getattr(self, a)(k, perf), perf["f0"])
            cb = BUILD[b](getattr(self, b)(torch.roll(k, 11, 1), perf), perf["f0"])
            xf = (k[:, 31:32] + perf["mod"] * 0.3).clamp(0, 1)
            ctrl = merge_hybrid(ca, cb, xf)
        else:
            ctrl = BUILD[method](getattr(self, method)(k, perf), perf["f0"])
        with torch.no_grad():
            audio = render(ctrl, perf["f0"], ON_FRAME, ctx)
        return {"method": method, "pair": pair, "knobs": k, "mvec": mvec, "perf": perf, "ctx": ctx,
                "ctrl": ctrl, "audio": audio, "lowpass": None}


# ================================================================ preview rendering

_PREVIEW = {}


def render_patch(patch, pitch=60, velocity=0.8, seconds=1.0, ckpt=None, tier="standard"):
    """Render one note of a patch to 48 kHz mono float32 audio (np.ndarray), starting at note-on.

    patch: {"method": one of METHODS, "knobs": 32 values in [0, 1], "pair": [a, b] for hybrid}.
    The note is held for 70% of `seconds`, then released. Levels are not normalised; in a sweep of
    360 random patches peaks ran 0.1 to 0.95 (median 0.1 to 0.5 per method). Without a checkpoint the patch plays
    through the procedural teacher programs (what stage 1 trains toward); with `ckpt` (or the
    ALFA_CKPT environment variable) it plays through the trained network. Deterministic per call.
    """
    method = patch["method"]
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    knobs = torch.tensor([float(x) for x in patch["knobs"]], dtype=torch.float32).clamp(0, 1)
    if knobs.numel() != N_KNOBS:
        raise ValueError(f"patch needs {N_KNOBS} knobs, got {knobs.numel()}")
    pair = list(patch.get("pair") or ["wavetable", "fm"]) if method == "hybrid" else None
    if pair is not None and (len(pair) != 2 or frozenset(pair) in HYBRID_EXCLUDE
                             or any(p not in HYBRID_PARTS for p in pair)):
        raise ValueError(f"hybrid pair must be two of {HYBRID_PARTS}, not additive with physical")
    ckpt = ckpt or os.environ.get("ALFA_CKPT")
    key = ckpt or tier
    if key not in _PREVIEW:
        model = None
        if ckpt:
            model, _ = load_ckpt(ckpt, torch.device("cpu"))
            model.eval()
        t = model.tier if model is not None else TIERS[tier]
        _PREVIEW[key] = (model, Teachers(t, torch.device("cpu")))
    model, teachers = _PREVIEW[key]
    teachers.gen.manual_seed(1234)
    T = ON_FRAME + max(1, int(round(seconds * FRAME_HZ)))
    perf = make_perf(1, T, torch.device("cpu"), teachers.gen, pitch=torch.tensor([float(pitch)]),
                     vel=torch.tensor([float(velocity)]), length=torch.tensor([round(0.7 * (T - ON_FRAME))]),
                     expressive=False)
    with torch.no_grad():
        b = teachers.batch(method, 1, T, pair, k=knobs.view(1, -1), perf=perf)
        if model is None:
            y = b["audio"]
        else:
            ctrl, _, _ = model(b)
            y = render(_floatify(ctrl), perf["f0"], ON_FRAME, model.render_ctx(b))
    return y[0, ON_FRAME * HOP:].numpy().astype(np.float32)


# ================================================================ model (METHOD.md s1, s3)

def gru_step(x, h, g):
    """One step of an nn.GRU (single layer) using its weights; used by the exported step graphs."""
    gi = x @ g.weight_ih_l0.t() + g.bias_ih_l0
    gh = h @ g.weight_hh_l0.t() + g.bias_hh_l0
    ir, iz, inn = gi.chunk(3, -1)
    hr, hz, hn = gh.chunk(3, -1)
    r = torch.sigmoid(ir + hr)
    z = torch.sigmoid(iz + hz)
    n = torch.tanh(inn + r * hn)
    return (1 - z) * n + z * h


class RoutedMLP(nn.Module):
    """Residual switch layer: top-1 routing over K sub-experts (straight-through gate)."""

    def __init__(self, d, hs, K):
        super().__init__()
        self.K = K
        self.ln = nn.LayerNorm(d)
        self.router = nn.Linear(d, K) if K > 1 else None
        self.w1 = nn.Parameter(torch.randn(K, d, hs) * d ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(K, hs))
        self.w2 = nn.Parameter(torch.randn(K, hs, d) * hs ** -0.5 * 0.5)
        self.b2 = nn.Parameter(torch.zeros(K, d))

    def forward(self, x):
        shp = x.shape
        xn = self.ln(x.reshape(-1, shp[-1]))
        if self.K == 1:
            y = F.gelu(xn @ self.w1[0] + self.b1[0]) @ self.w2[0] + self.b2[0]
            z = torch.zeros((), device=x.device)
            return x + y.view(shp), z, z
        logits = self.router(xn).float()
        p = logits.softmax(-1)
        idx = p.argmax(-1)
        gate = p.gather(-1, idx[:, None])
        y = torch.zeros_like(xn)
        for k in range(self.K):
            m = idx == k
            if m.any():
                y[m] = (F.gelu(xn[m] @ self.w1[k] + self.b1[k]) @ self.w2[k] + self.b2[k]).to(y.dtype)
        y = y * gate.to(y.dtype)
        f = F.one_hot(idx, self.K).float().mean(0)
        bal = self.K * (f * p.mean(0)).sum()
        zl = torch.logsumexp(logits, -1).pow(2).mean()
        return x + y.view(shp), bal, zl

    def step(self, x):
        """Gather form for export: x [V,d]; per-voice compute is one sub-expert."""
        xn = self.ln(x)
        if self.K == 1:
            return x + F.gelu(xn @ self.w1[0] + self.b1[0]) @ self.w2[0] + self.b2[0]
        p = self.router(xn).softmax(-1)
        idx = p.argmax(-1)
        gate = p.gather(-1, idx[:, None])
        h = F.gelu(torch.bmm(xn.unsqueeze(1), self.w1[idx]).squeeze(1) + self.b1[idx])
        y = torch.bmm(h.unsqueeze(1), self.w2[idx]).squeeze(1) + self.b2[idx]
        return x + y * gate


class PKSelect(nn.Module):
    """Product-key top-2 selection over N = N1*N2 table entries, per group. Cost d*(N1+N2)."""

    def __init__(self, d, N, groups=1):
        super().__init__()
        N = max(N, 4)
        self.N1 = max(2, math.ceil(math.sqrt(N)))
        self.N2 = max(2, math.ceil(N / self.N1))
        self.N, self.G = self.N1 * self.N2, groups
        self.qa = nn.Linear(d, groups * self.N1)
        self.qb = nn.Linear(d, groups * self.N2)

    def forward(self, h):
        a = self.qa(h).view(*h.shape[:-1], self.G, self.N1).float()
        b = self.qb(h).view(*h.shape[:-1], self.G, self.N2).float()
        av, ai = a.topk(2, -1)
        bv, bi = b.topk(2, -1)
        cand = (av.unsqueeze(-1) + bv.unsqueeze(-2)).flatten(-2)
        ci = (ai.unsqueeze(-1) * self.N2 + bi.unsqueeze(-2)).flatten(-2)
        v, j = cand.topk(2, -1)
        return ci.gather(-1, j), v.softmax(-1)                   # [...,G,2]


def _head_spec(m, t):
    K, N, M, G, R = t.K_h, t.n_ops, t.n_modes, t.n_grains, t.unison - 1
    return {
        "subtractive": [("mix", 3), ("pw", 1), ("cutoff", 1), ("q", 1), ("mode", 3), ("noise", 1),
                        ("amp", 1), ("attack", 1), ("det_ratio", R), ("det_level", R)],
        "wavetable": [("pos", 1), ("tilt", 1), ("amp", 1), ("attack", 1), ("det_ratio", R), ("det_level", R)],
        "additive": [("harm", M), ("detune", M), ("stretch", 1), ("amp", 1), ("attack", 1)],
        "spectral": [("env", N_BANDS), ("balance", 1), ("amp", 1), ("attack", 1)],
        "fm": [("ratio", N), ("snap", 1), ("level", N), ("index", N * N), ("amp", 1), ("attack", 1)],
        "physical": [("ratio", M), ("tau", M), ("exc", M), ("breath", N_BANDS), ("attack", 1)],
        "granular": [("pos", G), ("ratio", G), ("gain", G), ("pitched", 1), ("attack", 1)],
        "sampling": [("rate", 1), ("start", 1), ("gain", 1), ("eq", N_BANDS), ("resid", 16), ("attack", 1)],
    }[m]


def _table_entry(m, t):
    return {"wavetable": 32 * t.K_h * 2, "additive": t.n_modes, "spectral": N_BANDS, "physical": t.n_modes,
            "granular": GRAIN_LEN, "sampling": N_BANDS}.get(m, 0)


class Expert(nn.Module):
    def __init__(self, m, t, budget):
        super().__init__()
        self.m, self.t = m, t
        Hv, Hs = t.Hv, t.Hs
        self.sl = Slicer(_head_spec(m, t))
        self.head = nn.Linear(Hv, self.sl.n)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        entry = _table_entry(m, t)
        groups = t.n_grains if m == "granular" else 1
        per_k = 2 * (2 * Hv * Hs + Hs + Hv + Hv)                 # two routed layers, one sub-expert
        if entry:
            tab_budget = budget * TABLE_FRAC[m]
            rest = budget - tab_budget - self.head.weight.numel() - self.head.bias.numel()
            if rest < 2 * (per_k + 2 * Hv):                       # too small for two sub-experts:
                tab_budget += rest - per_k                        # keep one, the rest goes to the table
            self.sel = PKSelect(Hv, max(4, int(tab_budget / entry)), groups)
            n_tab = self.sel.N
            if m == "wavetable":
                kk = torch.arange(1, t.K_h + 1).float().view(1, 1, -1, 1)
                tab = torch.randn(n_tab, 32, t.K_h, 2) * 0.3 / kk
            elif m == "granular":
                tab = torch.randn(n_tab, GRAIN_LEN) * 0.1
            else:
                init = {"additive": -3.0, "spectral": -3.0, "physical": 0.0, "sampling": 0.0}[m]
                tab = torch.randn(n_tab, entry) * 0.3 + init
            self.table = nn.Parameter(tab)
        fixed = sum(p.numel() for p in self.parameters())
        K = int((budget - fixed) // (per_k + 2 * Hv))
        if K < 2:                                                 # "small control map" fallback
            K = 1
            Hs = max(16, min(Hs, int((budget - fixed) / 2 / (2 * Hv + 1))))
        self.K, self.Hs = K, Hs
        self.layers = nn.ModuleList([RoutedMLP(Hv, Hs, K) for _ in range(2)])

    def trunk(self, h, step=False):
        bal = zl = torch.zeros((), device=h.device)
        for layer in self.layers:
            if step:
                h = layer.step(h)
            else:
                h, b, z = layer(h)
                bal, zl = bal + b, zl + z
        return h, bal, zl

    def _lookup(self, h):
        idx, w = self.sel(h)
        return (self.table[idx[..., 0, :]] * w[..., 0, :, None]).sum(-2)

    def params(self, h, f0):
        """Activate the head (and look up tables) into method params. Per frame, any leading dims."""
        r = self.sl(self.head(h).float())
        m = self.m
        att = log_rng(r["attack"] - 2, 3e-4, 0.08)
        uni = {}
        if "det_level" in r:          # +-1 semitone; exact zeros below 1e-4 so silent reads cost nothing
            uni = {"det_ratio": 2 ** (torch.tanh(r["det_ratio"]) / 12),
                   "det_level": F.relu(exp_sig(r["det_level"] - 3, 1.0) - 1e-4)}
        if m == "subtractive":
            return {"mix": F.softplus(r["mix"]), "pw": rng(r["pw"], 0.05, 0.95),
                    "cutoff": log_rng(r["cutoff"], 20, 20000), "q": log_rng(r["q"], 0.5, 20),
                    "mode": r["mode"].softmax(-1), "noise": exp_sig(r["noise"] - 3, 0.5),
                    "amp": exp_sig(r["amp"] - 1, 1.0), "attack": att, **uni}
        if m == "wavetable":
            idx, w = self.sel(h)                                  # [...,1,2]
            tab = (self.table[idx[..., 0, :]] * w[..., 0, :, None, None, None]).sum(-4)
            fr = interp_frames(tab, torch.sigmoid(r["pos"][..., 0]))
            return {"wt_re": fr[..., 0], "wt_im": fr[..., 1], "tilt": torch.tanh(r["tilt"]) * 4,
                    "amp": exp_sig(r["amp"], 1.0), "attack": att, **uni}
        if m == "additive":
            return {"harm": exp_sig(self._lookup(h) + r["harm"], 1.0), "detune": torch.tanh(r["detune"]) * 0.01,
                    "stretch": log_rng(r["stretch"] - 3, 1e-6, 1e-2), "amp": exp_sig(r["amp"], 1.0), "attack": att}
        if m == "spectral":
            return {"env": exp_sig(self._lookup(h) + r["env"], 2.0), "balance": torch.sigmoid(r["balance"]),
                    "amp": exp_sig(r["amp"], 1.0), "attack": att}
        if m == "fm":
            raw = log_rng(r["ratio"], 0.25, 16)
            snap = torch.sigmoid(r["snap"] + 2)
            rnd = raw + (torch.round(raw) - raw).detach()
            ratio = snap * rnd.clamp(min=0.5) + (1 - snap) * raw
            return {"ratio": ratio, "level": exp_sig(r["level"], 1.0), "index": F.softplus(r["index"]) * 2,
                    "amp": exp_sig(r["amp"] - 1, 1.0), "attack": att}
        if m == "physical":
            steps = F.softplus(self._lookup(h) + 0.1 * r["ratio"])
            ratio = 1 + torch.cumsum(steps, -1) - steps[..., :1]
            return {"ratio": ratio, "tau": log_rng(r["tau"] - 2, 0.005, 20), "exc": exp_sig(r["exc"] - 4, 1.0),
                    "breath": exp_sig(r["breath"] - 3, 0.5), "attack": att}
        if m == "granular":
            idx, w = self.sel(h)                                  # [...,G,2]
            return {"idx": idx, "w": w, "pos": torch.sigmoid(r["pos"]), "ratio": log_rng(r["ratio"], 0.25, 4),
                    "pitched": torch.sigmoid(r["pitched"]), "gain": exp_sig(r["gain"], 1.0), "attack": att}
        if m == "sampling":
            return {"rate": 1 + torch.tanh(r["rate"]) * 0.06, "start": torch.sigmoid(r["start"] - 3),
                    "gain": exp_sig(r["gain"], 2.0), "eq": torch.exp(self._lookup(h) + r["eq"]),
                    "resid": exp_sig(r["resid"] - 3, 0.2), "attack": att}
        raise KeyError(m)


class HybridExpert(nn.Module):
    """Routed MLP -> additive shifts of h for the two component experts, and their crossfade."""

    def __init__(self, t, budget):
        super().__init__()
        Hv = t.Hv
        self.head = nn.Linear(Hv, 2 * Hv + 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        fixed = sum(p.numel() for p in self.parameters())
        per_k = 2 * (2 * Hv * t.Hs + t.Hs + 2 * Hv)
        K = int((budget - fixed) // (per_k + 2 * Hv))
        Hs = t.Hs
        if K < 2:
            K, Hs = 1, max(16, min(Hs, int((budget - fixed) / 2 / (2 * Hv + 1))))
        self.K, self.Hs = K, Hs
        self.layers = nn.ModuleList([RoutedMLP(Hv, Hs, K) for _ in range(2)])

    def forward(self, h, step=False):
        bal = zl = torch.zeros((), device=h.device)
        x = h
        for layer in self.layers:
            if step:
                x = layer.step(x)
            else:
                x, b, z = layer(x)
                bal, zl = bal + b, zl + z
        o = self.head(x).float()
        Hv = h.shape[-1]
        return h + o[..., :Hv].to(h.dtype), h + o[..., Hv:2 * Hv].to(h.dtype), torch.sigmoid(o[..., -1]), bal, zl


class PatchEncoder(nn.Module):
    def __init__(self, d_in, W, E, n_blocks):
        super().__init__()
        self.inp = nn.Linear(d_in, W)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(W), nn.Linear(W, 2 * W), nn.GELU(),
                                                   nn.Linear(2 * W, W)) for _ in range(n_blocks)])
        self.out = nn.Linear(W, E)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


class AudioEncoder(nn.Module):
    """Log-mel of a reference note or a loaded sample -> patch-embedding offset."""

    def __init__(self, W, E):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(64, W, 5, padding=2), nn.GELU(), nn.Conv1d(W, W, 5, padding=2),
                                 nn.GELU(), nn.Conv1d(W, W, 5, padding=2), nn.GELU())
        self.out = nn.Linear(W, E)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, mel):
        return self.out(self.net(mel).mean(-1))


class Alfamedomega(nn.Module):
    def __init__(self, tier_name):
        super().__init__()
        t = TIERS[tier_name]
        self.tier = t
        C = t.Hv
        self.inst = nn.GRU(t.E + N_GLOBAL, t.Hi, batch_first=True)
        self.inst_out = nn.Linear(t.Hi, C)
        self.v_in = nn.Linear(N_PERF + C, t.Hv)
        self.voice = nn.GRU(t.Hv, t.Hv, batch_first=True)
        self.v_ln = nn.LayerNorm(t.Hv)
        self.audio = AudioEncoder(t.Wa, t.E)
        small = sum(p.numel() for p in self.parameters())
        n_blocks = max(1, round((t.total * TRUNK_SHARE - small - t.E * t.Wp) / (4 * t.Wp * t.Wp + 5 * t.Wp)))
        self.patch = PatchEncoder(N_KNOBS + METHOD_IN, t.Wp, t.E, n_blocks)
        shares = expert_shares()
        pool = t.total * (1 - TRUNK_SHARE)
        self.experts = nn.ModuleDict({m: Expert(m, t, pool * shares[m]) for m in METHODS if m != "hybrid"})
        self.hybrid = HybridExpert(t, pool * shares["hybrid"])

    # -- training-time (whole clip) forward
    def embed(self, knobs, mvec, ref_audio=None):
        e = self.patch(torch.cat([knobs, mvec], -1))
        if ref_audio is not None:
            e = e + self.audio(logmel(ref_audio))
        return e

    def trunk(self, e, perf):
        B, T = perf["feats"].shape[:2]
        c = self.inst_out(self.inst(torch.cat([e.unsqueeze(1).expand(B, T, -1), perf["glob"]], -1))[0])
        return self.v_ln(self.voice(self.v_in(torch.cat([perf["feats"], c], -1)))[0])

    def controls(self, method, h, f0, pair=None):
        if method == "hybrid":
            ha, hb, xf, bal, zl = self.hybrid(h)
            ca, ba, za = self.controls(pair[0], ha, f0)
            cb, bb, zb = self.controls(pair[1], hb, f0)
            return merge_hybrid(ca, cb, xf), bal + ba + bb, zl + za + zb
        ex = self.experts[method]
        x, bal, zl = ex.trunk(h)
        ctrl = BUILD[method](ex.params(x, f0.float()), f0.float())
        return ctrl, bal, zl

    def forward(self, batch):
        ref = batch["ctx"].get("sample") if batch["method"] == "sampling" else batch.get("ref")
        e = self.embed(batch["knobs"], batch["mvec"], ref)
        h = self.trunk(e, batch["perf"])
        return self.controls(batch["method"], h, batch["perf"]["f0"], batch.get("pair"))

    def render_ctx(self, batch):
        ctx = dict(batch["ctx"])
        ctx["grain_bank"] = self.experts["granular"].table
        return ctx


# ================================================================ param report (METHOD.md s7.4)

def count(m):
    return sum(p.numel() for p in m.parameters())


def active_per_voice(model):
    """Per-voice, per-frame params: voice core + one sub-expert per routed layer + router + head
    + selection, for the largest expert."""
    core = count(model.v_in) + count(model.voice) + count(model.v_ln)
    best = 0
    for ex in model.experts.values():
        a = count(ex.head) + (count(ex.sel) if hasattr(ex, "sel") else 0)
        for layer in ex.layers:
            a += count(layer.ln) + (count(layer.router) if layer.router is not None else 0)
            a += layer.w1[0].numel() + layer.b1[0].numel() + layer.w2[0].numel() + layer.b2[0].numel()
        best = max(best, a)
        ex.active = core + a
    return core + best


def report(tiers=None):
    rows = []
    for name in tiers or TIERS:
        with torch.device("meta"):
            m = Alfamedomega(name)
        t = m.tier
        tot = count(m)
        trunk = (count(m.patch) + count(m.inst) + count(m.inst_out) + count(m.v_in) + count(m.voice)
                 + count(m.v_ln) + count(m.audio))
        inst = count(m.inst) + count(m.inst_out)
        act = active_per_voice(m)
        flag = lambda a, b: "" if abs(a / b - 1) <= 0.10 else "  <-- off by %+.0f%%" % (100 * (a / b - 1))
        print(f"\n== {name}: total {tot/1e6:.2f}M (target {t.total/1e6:.0f}M){flag(tot, t.total)}; "
              f"trunk {trunk/1e6:.2f}M ({trunk/tot*100:.0f}%); active/voice {act/1e6:.3f}M "
              f"(target {t.active/1e6:.2f}M){flag(act, t.active)}; instance/frame {inst/1e6:.2f}M "
              f"(target {t.instance/1e6:.2f}M){flag(inst, t.instance)}; patch blocks {len(m.patch.blocks)}")
        for k, ex in list(m.experts.items()) + [("hybrid", m.hybrid)]:
            tab = ex.table.numel() if hasattr(ex, "table") else 0
            print(f"   {k:<12} {count(ex)/1e6:7.2f}M  K={ex.K:<3} Hs={ex.Hs:<5} table {tab/1e6:6.2f}M"
                  + (f"  active/voice {ex.active/1e6:.3f}M" if hasattr(ex, "active") else ""))
        rows.append((name, tot, act, inst))
    return rows


# ================================================================ data: real notes (stage 3)

class NotesDataset:
    """Labelled single notes. Supports NSynth layout: <root>/examples.json + <root>/audio/<id>.wav.
    Returns a target note and a reference note (another note of the same instrument)."""

    def __init__(self, root, clip_frames, dev):
        with open(os.path.join(root, "examples.json")) as f:
            meta = json.load(f)
        self.root, self.T, self.dev = root, clip_frames, dev
        self.items = list(meta.items())
        self.by_inst = {}
        for i, (key, v) in enumerate(self.items):
            self.by_inst.setdefault(v.get("instrument_str", v.get("instrument", "x")), []).append(i)

    def _load(self, key):
        with wave.open(os.path.join(self.root, "audio", key + ".wav")) as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            x = np.frombuffer(w.readframes(n), dtype={2: np.int16, 4: np.int32}[sw]).astype(np.float32)
            x = x.reshape(-1, ch).mean(1) / float(2 ** (8 * sw - 1))
        x = torch.from_numpy(x.copy())
        if sr != SR:                                              # band-limited FFT resample
            n_out = int(round(x.shape[0] * SR / sr))
            X = torch.fft.rfft(x)
            X = F.pad(X, (0, max(0, n_out // 2 + 1 - X.shape[0])))[:n_out // 2 + 1]
            x = torch.fft.irfft(X, n=n_out) * (n_out / len(x))
        return x, sr

    def batch(self, B):
        S = self.T * HOP
        tgt, ref, pitch, vel, rpitch, lp = [], [], [], [], [], SR / 2
        for _ in range(B):
            key, v = self.items[random.randrange(len(self.items))]
            rkey, rv = self.items[random.choice(self.by_inst[v.get("instrument_str", v.get("instrument", "x"))])]
            x, sr = self._load(key)
            r, _ = self._load(rkey)
            lp = min(lp, sr / 2)
            x = F.pad(x, (ON_FRAME * HOP, 0))[:S]
            tgt.append(F.pad(x, (0, S - x.shape[0])))
            Lr = int(1.5 * SR)
            ref.append(F.pad(r[:Lr], (0, max(0, Lr - r.shape[0]))))
            pitch.append(float(v["pitch"]))
            vel.append(float(v.get("velocity", 100)) / 127)
            rpitch.append(float(rv["pitch"]))
        dev = self.dev
        perf = make_perf(B, self.T, dev, None, pitch=torch.tensor(pitch, device=dev),
                         vel=torch.tensor(vel, device=dev),
                         length=torch.full((B,), float(self.T + 100), device=dev), expressive=False)
        return {"knobs": torch.full((B, N_KNOBS), 0.5, device=dev), "perf": perf,
                "audio": torch.stack(tgt).to(dev), "ref": torch.stack(ref).to(dev),
                "ref_pitch": torch.tensor(rpitch, device=dev), "lowpass": lp}


# ================================================================ training (METHOD.md s5)

def method_probs(stage):
    sh = expert_shares()
    q = {m: sh[m] ** 0.5 for m in METHODS if m != "hybrid"}
    q["hybrid"] = sum(q.values()) * (0.3 / 0.7 if stage == "polish" else 0.05)
    s = sum(q.values())
    return {m: v / s for m, v in q.items()}


def make_optimizer(model, lr):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.dim() < 2 or n.endswith("table") or ".ln." in n else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.99))


def lr_at(step, total, base, warmup=2000):
    warmup = min(warmup, max(1, total // 10))
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def write_wav(path, x):
    x = (x.detach().float().cpu().clamp(-1, 1).numpy() * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.tobytes())


def load_ckpt(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = Alfamedomega(ck["tier"]).to(dev)
    model.load_state_dict(ck["model"])
    return model, ck


def _floatify(ctrl):
    return tree_map(lambda x: x.float() if x.is_floating_point() else x, ctrl)


def train(args):
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.smoke:
        args.steps, args.batch, args.clip, args.accum = min(args.steps, 6), 2, 0.12, 1
        args.log_every, args.ckpt_every, args.audio_every = 1, 3, 3
    T = max(ON_FRAME + 8, int(round(args.clip * FRAME_HZ)))
    os.makedirs(args.out, exist_ok=True)
    if args.init:
        model, _ = load_ckpt(args.init, dev)
        assert model.tier.name == args.tier, "--init checkpoint tier differs from --tier"
    else:
        model = Alfamedomega(args.tier).to(dev)
    model.train()
    teacher_model = None
    if args.stage == "distill":
        teacher_model, _ = load_ckpt(args.teacher, dev)
        teacher_model.eval().requires_grad_(False)
    teachers = Teachers(model.tier, dev, args.seed)
    notes = NotesDataset(args.data, T, dev) if args.stage in ("finetune", "distill") and args.data else None
    lr = args.lr or model.tier.lr * (0.3 if args.stage in ("polish", "finetune") else 1.0)
    opt = make_optimizer(model, lr)
    probs = method_probs(args.stage)
    amp = dev.type == "cuda" and torch.cuda.is_bf16_supported() and not args.no_amp
    ft_methods = args.ft_methods.split(",")
    print(f"tier {args.tier}, stage {args.stage}, device {dev}, bf16 {amp}, {count(model)/1e6:.2f}M params, "
          f"clip {T} frames, batch {args.batch}x{args.accum}")
    t0, log = time.time(), {}
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, lr)
        lam = 1.0 - 0.9 * min(1.0, step / max(1, 0.5 * args.steps)) if args.stage == "pretrain" else 0.1
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            use_real = notes is not None and (args.stage == "finetune" or random.random() < 0.5)
            if use_real:
                method = random.choice(ft_methods)
                batch = notes.batch(args.batch)
                mvec = torch.zeros(args.batch, METHOD_IN, device=dev)
                mvec[:, M_IDX[method]] = 1
                batch.update(method=method, pair=None, mvec=mvec, ctrl=None,
                             ctx={"sample": batch["ref"], "sample_root": midi_hz(batch["ref_pitch"])})
            else:
                method = random.choices(list(probs), weights=list(probs.values()))[0]
                pair = hybrid_pair() if method == "hybrid" else None
                batch = teachers.batch(method, args.batch, T, pair)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                ctrl, bal, zl = model(batch)
            ctrl = _floatify(ctrl)
            y = render(ctrl, batch["perf"]["f0"], ON_FRAME, model.render_ctx(batch))
            loss_a = mrstft(y, batch["audio"], lowpass_hz=batch.get("lowpass"))
            loss = loss_a + 0.01 * bal + 0.001 * zl
            loss_c = torch.zeros((), device=dev)
            if batch["ctrl"] is not None and args.stage != "distill":
                loss_c = ctrl_distance(ctrl, batch["ctrl"])
                loss = loss + lam * loss_c
            if teacher_model is not None:
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    tctrl, _, _ = teacher_model(batch)
                with torch.no_grad():
                    tctrl = _floatify(tctrl)
                    ty = render(tctrl, batch["perf"]["f0"], ON_FRAME, teacher_model.render_ctx(batch))
                loss_c = ctrl_distance(ctrl, tctrl)
                loss = 0.25 * loss + loss_c + mrstft(y, ty, lowpass_hz=batch.get("lowpass"))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step} ({method})")
            (loss / args.accum).backward()
            for k_, v_ in (("loss", loss), ("mrstft", loss_a), ("ctrl", loss_c), ("bal", bal)):
                log.setdefault(k_, []).append(float(v_.detach()))
            log.setdefault("methods", []).append(method)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            msg = " ".join(f"{k} {np.mean(v):.4f}" for k, v in log.items() if k != "methods")
            print(f"step {step+1}/{args.steps} {msg} gnorm {float(gn):.2f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{(time.time()-t0)/args.log_every:.2f}s/step [{','.join(sorted(set(log['methods'])))}]", flush=True)
            log, t0 = {}, time.time()
        if (step + 1) % args.audio_every == 0:
            write_wav(os.path.join(args.out, f"s{step+1}_{method}_pred.wav"), y[0])
            write_wav(os.path.join(args.out, f"s{step+1}_{method}_target.wav"), batch["audio"][0])
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            path = os.path.join(args.out, f"{args.tier}_{args.stage}.pt")
            torch.save({"tier": args.tier, "model": model.state_dict(), "step": step + 1,
                        "layout_version": LAYOUT_VERSION, "args": vars(args)}, path)
            print(f"saved {path}")
    return model


# ================================================================ export (METHOD.md s7)

class _PatchG(nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, knobs, method, hybrid_parts):
        return s.m.patch(torch.cat([knobs, method, hybrid_parts], -1))


class _AudioG(nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, mel):
        return s.m.audio(mel)


class _InstG(nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, e, glob, state):
        h = gru_step(torch.cat([e, glob], -1), state, s.m.inst)
        return s.m.inst_out(h), h


class _VoiceG(nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, feats, c, state):
        h = gru_step(s.m.v_in(torch.cat([feats, c.expand(feats.shape[0], -1)], -1)), state, s.m.voice)
        return s.m.v_ln(h), h


class _ExpertG(nn.Module):
    """hidden [V,Hv], f0 [V] (+ modal state for physical) -> VoiceFrame fields, flattened per
    voice (fm.index as N x N row-major, grain.source as float), per src/inference/MODEL_CONTRACT.md."""

    def __init__(s, m, name):
        super().__init__()
        s.ex, s.name = m.experts[name], name

    def forward(s, h, f0, *state):
        x, _, _ = s.ex.trunk(h, step=True)
        p = s.ex.params(x, f0)
        if s.name == "physical":
            ctrl, new_state = build_physical(p, f0, state[0])
        else:
            ctrl, new_state = BUILD[s.name](p, f0), None
        outs = []
        for k in sorted(ctrl):
            v = ctrl[k]
            if k == "grain.source":
                v = v.float()
            outs.append(v.reshape(v.shape[0], -1))
        return tuple(outs) + ((new_state,) if new_state is not None else ())


class _HybridG(nn.Module):
    def __init__(s, m):
        super().__init__()
        s.hy = m.hybrid

    def forward(s, h):
        ha, hb, xf, _, _ = s.hy(h, step=True)
        return ha, hb, xf


def _stream(model, batch):
    """Control frames from the step graphs' torch code, frame by frame (streaming parity)."""
    e = model.embed(batch["knobs"], batch["mvec"])
    perf = batch["perf"]
    B, T = perf["f0"].shape
    hi = torch.zeros(B, model.tier.Hi)
    hv = torch.zeros(B, model.tier.Hv)
    ms = torch.zeros(B, model.tier.n_modes)
    ig, vg, eg = _InstG(model), _VoiceG(model), _ExpertG(model, batch["method"])
    out = []
    for t in range(T):
        c, hi = ig(e, perf["glob"][:, t], hi)
        rows = [vg(perf["feats"][b:b + 1, t], c[b:b + 1], hv[b:b + 1]) for b in range(B)]  # one instance per clip
        h = torch.cat([r[0] for r in rows])
        hv = torch.cat([r[1] for r in rows])
        o = eg(h, perf["f0"][:, t], ms) if batch["method"] == "physical" else eg(h, perf["f0"][:, t])
        if batch["method"] == "physical":
            o, ms = o[:-1], o[-1]
        out.append(o)
    return [torch.stack([o[i] for o in out], 1) for i in range(len(out[0]))]


def export(args):
    import onnxruntime as ort
    dev = torch.device("cpu")
    model = load_ckpt(args.ckpt, dev)[0] if args.ckpt else Alfamedomega(args.tier).to(dev)
    model.eval()
    t = model.tier
    root = args.out
    folder = f"{t.name}-{args.version}"
    args = argparse.Namespace(**{**vars(args), "out": os.path.join(root, folder)})
    os.makedirs(args.out, exist_ok=True)
    teachers = Teachers(t, dev, 7)
    V = 3
    checks, graph_inputs, layout = [], {}, {}

    # 1. streaming parity: step graphs vs whole-clip forward
    for m in ["subtractive", "fm", "physical", "granular"]:
        batch = teachers.batch(m, 2, 24)
        with torch.no_grad():
            full, _, _ = model.controls(m, model.trunk(model.embed(batch["knobs"], batch["mvec"]), batch["perf"]),
                                        batch["perf"]["f0"])
            steps = _stream(model, batch)
        err = 0.0
        for k, s in zip(sorted(full), steps):
            f = full[k].float()
            err = max(err, float((f.reshape(*s.shape) - s).abs().max()))
        checks.append(f"streaming parity {m}: max abs err {err:.2e}")
        assert err < 1e-3, f"streaming parity failed for {m}: {err}"

    # 2. export each graph and check ORT parity with dynamic V
    def do(name, mod, inputs, in_names, out_names, dyn):
        path = os.path.join(args.out, name + ".onnx")
        with torch.no_grad():
            ref = mod(*inputs)
        ref = ref if isinstance(ref, tuple) else (ref,)
        torch.onnx.export(mod, inputs, path, input_names=in_names, output_names=out_names,
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
        import onnx
        g = onnx.load(path)                                     # pin non-voice dims to fixed sizes so the
        for o, r in zip(g.graph.output, ref):                  # host can read widths from the graph
            for i, d in enumerate(o.type.tensor_type.shape.dim):
                if i > 0 or not dyn or o.name not in dyn:
                    d.dim_value = int(r.shape[i])
        onnx.save(g, path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        present = {i.name for i in sess.get_inputs()}          # unused inputs (e.g. f0) get pruned
        got = sess.run(None, {n: x.numpy() for n, x in zip(in_names, inputs) if n in present})
        err = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref) if r.is_floating_point())
        graph_inputs[name] = [n for n in in_names if n in present]
        checks.append(f"{name}.onnx: {os.path.getsize(path)/1e6:.1f} MB, ORT max abs err {err:.2e}")
        assert err < 1e-3, f"ORT parity failed for {name}: {err}"

    E, Hv, Hi = t.E, t.Hv, t.Hi
    do("patch_encoder", _PatchG(model), (torch.rand(1, N_KNOBS), torch.eye(len(METHODS))[:1],
                                         torch.zeros(1, 2 * len(METHODS))),
       ["knobs", "method", "hybrid_parts"], ["embedding"], None)
    do("audio_encoder", _AudioG(model), (torch.randn(1, 64, 141),), ["logmel"], ["e_audio"], {"logmel": {2: "frames"}})
    do("instance_step", _InstG(model), (torch.randn(1, E), torch.rand(1, N_GLOBAL), torch.zeros(1, Hi)),
       ["embedding", "global_mod", "state"], ["context", "state_out"], None)
    do("voice_core", _VoiceG(model), (torch.rand(V, N_PERF), torch.randn(1, Hv), torch.zeros(V, Hv)),
       ["features", "context", "state"], ["hidden", "state_out"],
       {"features": {0: "V"}, "state": {0: "V"}, "hidden": {0: "V"}, "state_out": {0: "V"}})
    hv = torch.randn(V, Hv)
    f0 = torch.tensor([110.0, 261.6, 880.0])
    for m in model.experts:
        g = _ExpertG(model, m)
        ins, in_names = (hv, f0), ["hidden", "f0"]
        if m == "physical":
            ins, in_names = (hv, f0, torch.rand(V, t.n_modes)), ["hidden", "f0", "modal_state"]
        with torch.no_grad():
            outs = g(*ins)
            names = sorted(BUILD[m](model.experts[m].params(hv, f0), f0) if m != "physical"
                           else build_physical(model.experts[m].params(hv, f0), f0, ins[2])[0])
        names = names + (["modal_state_out"] if m == "physical" else [])
        layout[m] = {n: int(o.shape[1]) for n, o in zip(names, outs)}
        do(f"expert_{m}", g, ins, in_names, names, {**{n: {0: "V"} for n in in_names}, **{n: {0: "V"} for n in names}})
    do("expert_hybrid", _HybridG(model), (hv,), ["hidden"], ["hidden_a", "hidden_b", "xf"],
       {"hidden": {0: "V"}, "hidden_a": {0: "V"}, "hidden_b": {0: "V"}, "xf": {0: "V"}})
    np.save(os.path.join(args.out, "grain_bank.npy"), model.experts["granular"].table.detach().numpy())

    # 3. golden DSP vectors per method, as flat VoiceFrames starting at the note-on frame
    gold = {}
    for name in METHODS + ["unison"]:
        m = "wavetable" if name == "unison" else name        # "unison": wavetable with every OSC read on
        pair = ["wavetable", "physical"] if m == "hybrid" else None
        k = None
        if name == "unison":
            k = torch.rand(1, N_KNOBS, generator=torch.Generator().manual_seed(11))
            k[:, 22], k[:, 23] = 1.0, 0.7
        b = teachers.batch(m, 1, 60, pair, k=k)
        cut = tree_map(lambda x: x[:, ON_FRAME:] if x.dim() >= 2 and x.shape[:2] == b["perf"]["f0"].shape else x, b["ctrl"])
        f0c = b["perf"]["f0"][:, ON_FRAME:]
        ctx = dict(b["ctx"], gen=torch.Generator().manual_seed(3))
        with torch.no_grad():
            y = render(cut, f0c, 0, ctx)
        gold[f"{name}/frames"] = to_voiceframe(cut)[0].numpy()
        gold[f"{name}/f0"] = f0c[0].numpy()
        gold[f"{name}/audio"] = y[0].numpy()
        if m == "sampling":
            gold[f"{m}/sample"] = b["ctx"]["sample"][0].numpy()
            gold[f"{m}/sample_root_hz"] = np.float32(SAMPLE_ROOT_HZ)
    gold["grain_bank"] = teachers.grain_bank.numpy()
    np.savez_compressed(os.path.join(args.out, "dsp_golden.npz"), **gold)
    checks.append("dsp_golden.npz: <method>/frames [T, VF_FLOATS] from the note-on frame, f0, audio at 48 kHz. "
                  "NOISE uses random noise, so compare noise-bearing methods spectrally.")

    # 4. param report
    act = active_per_voice(model)
    checks.append(f"params: total {count(model)/1e6:.2f}M (target {t.total/1e6:.0f}M), "
                  f"active/voice {act/1e6:.3f}M (target {t.active/1e6:.2f}M)")
    manifest = {"tier": t.name, "layout_version": LAYOUT_VERSION, "sr_trained": SR, "frame_hz": FRAME_HZ,
                "voiceframe_floats": VF_FLOATS, "voiceframe_offsets": VF_OFFSET,
                "voiceframe_defaults": VF_DEFAULTS,
                "tier_config": asdict(t), "graph_inputs": graph_inputs, "expert_outputs": layout,
                "note_features": ["pitch/127", "velocity", "gate", "log1p(t_on)/log1p(4)",
                                  "log1p(t_off)/log1p(4)", "mod", "aftertouch", "bend(+-2 st)"],
                "global_mod": ["mod", "aftertouch", "bend", "reserved"],
                "method_vector": "patch_encoder takes method one-hot [1,9] and hybrid_parts [1,18] = component A "
                                  "one-hot + component B one-hot (zeros unless hybrid), order " + ",".join(METHODS),
                "hybrid_merge": "OSC re/im = xf*mixA*A + (1-xf)*mixB*B, mix.osc = 1; NOISE band = sqrt((xf*mixA*A)^2 + "
                                "((1-xf)*mixB*B)^2), mix.noise = 1; other primitives from their one component with mix "
                                "scaled by its weight; attack_tau = min(A, B)",
                "checks": checks}
    with open(os.path.join(args.out, "export_info.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if args.int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        for fn in sorted(os.listdir(args.out)):
            if fn.endswith(".onnx") and not fn.endswith(".int8.onnx"):
                quantize_dynamic(os.path.join(args.out, fn), os.path.join(args.out, fn[:-5] + ".int8.onnx"),
                                 weight_type=QuantType.QInt8)
        checks.append("int8 copies written (*.int8.onnx); check them by ear before shipping")
    write_host_manifest(root, folder, t.name, args.version)
    checks.append(f"wrote {os.path.join(root, 'manifest.json')} entry for {folder}")
    print("\n".join(checks))


def write_host_manifest(root, folder, tier, version):
    """Models/manifest.json in the inference host's schema (MODEL_CONTRACT.md), with the SHA-256 of
    every file in the tier folder. Other tiers' entries are kept; the same tier+version is replaced."""
    import hashlib
    path = os.path.join(root, "manifest.json")
    man = {"schema": 1, "models": []}
    if os.path.exists(path):
        with open(path) as f:
            man = json.load(f)
    files = {}
    for fn in sorted(os.listdir(os.path.join(root, folder))):
        with open(os.path.join(root, folder, fn), "rb") as f:
            files[fn] = hashlib.sha256(f.read()).hexdigest()
    man["models"] = [m for m in man["models"] if not (m["tier"] == tier and m["version"] == version)]
    man["models"].append({"tier": tier, "version": version, "folder": folder, "opset": 17,
                          "frame_rate_hz": FRAME_HZ, "files": files})
    with open(path, "w") as f:
        json.dump(man, f, indent=1)


# ================================================================ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="param counts per tier vs the tier spec")
    r.add_argument("--tier", default=None)
    lc = sub.add_parser("layout", help="check the frame layout against the DSP's layout.json")
    lc.add_argument("path", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "..", "src", "dsp", "layout.json"))
    tr = sub.add_parser("train")
    tr.add_argument("--tier", default="lite", choices=list(TIERS))
    tr.add_argument("--stage", default="pretrain", choices=["pretrain", "polish", "finetune", "distill"])
    tr.add_argument("--steps", type=int, default=200000)
    tr.add_argument("--batch", type=int, default=16)
    tr.add_argument("--accum", type=int, default=4)
    tr.add_argument("--clip", type=float, default=1.0, help="seconds")
    tr.add_argument("--lr", type=float, default=None)
    tr.add_argument("--init", default=None, help="checkpoint to continue from")
    tr.add_argument("--teacher", default=None, help="Studio checkpoint for --stage distill")
    tr.add_argument("--data", default=None, help="NSynth-layout folder for finetune/distill")
    tr.add_argument("--ft_methods", default="sampling,physical,additive,subtractive,wavetable")
    tr.add_argument("--out", default="runs")
    tr.add_argument("--device", default=None)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--log_every", type=int, default=100)
    tr.add_argument("--ckpt_every", type=int, default=2000)
    tr.add_argument("--audio_every", type=int, default=5000)
    tr.add_argument("--no_amp", action="store_true")
    tr.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run")
    ex = sub.add_parser("export")
    ex.add_argument("--ckpt", default=None, help="omit to export an untrained model (pipeline test)")
    ex.add_argument("--tier", default="lite", choices=list(TIERS))
    ex.add_argument("--out", default="Models", help="Models folder; the tier goes in <out>/<tier>-<version>")
    ex.add_argument("--version", default="0.1.0")
    ex.add_argument("--int8", action="store_true", help="also write int8 dynamic-quantised copies (Lite)")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        report([args.tier] if args.tier else None)
    elif args.cmd == "layout":
        bad = check_layout(args.path)
        print("\n".join(bad) if bad else f"layout v{LAYOUT_VERSION} matches {args.path}")
        sys.exit(1 if bad else 0)
    elif args.cmd == "train":
        if args.stage == "distill" and not args.teacher:
            ap.error("--stage distill needs --teacher")
        if args.stage == "finetune" and not args.data:
            ap.error("--stage finetune needs --data")
        train(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
