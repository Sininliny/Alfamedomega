"""Compute budget for Alfamedomega model tiers.
Every input below is a planning assumption documented in model-tiers.md. Edit and re-run.

Execution model assumed: a neural control network runs at a fixed control rate
(frames per second, independent of sample rate) on worker threads, one frame ahead
of the audio thread (latency = 1 hop, reported to the host). The audio thread only
runs the DSP render (oscillator banks, filters, grains, waveguides, noise) driven by
those frames, so its per-block cost scales with buffer size and polyphony.
"""
import math

GFLOPS_CORE = 10.0        # ONNX Runtime CPU via anira (ADR 0001), one Zen3 / M1-class core, voices batched;
                          # hand-tuned SIMD kernels would reach ~20
MIN_SPEC_SLOWDOWN = 2.0   # ~2016 4-core laptop
DSP_SHARE_MAX = 0.25      # audio-thread render may use at most 25% of each block's wall time
NN_DEADLINE_SHARE = 0.5   # a frame must finish in half a hop (headroom for OS jitter)
BLOCK_OVERHEAD_US = 10.0

# Capacity plan (Claude Doc "Alfamedomega synthesis capacity plan"): expert share
# c_i = max(FLOOR, (1 - k*FLOOR) * p_i^ALPHA * d_i / sum over non-floored p^ALPHA * d).
# Edit prevalence (p) or difficulty (d) here; e.g. a bundled sample library raises d["sampling"].
PREVALENCE = {"sampling": 30, "wavetable": 22, "subtractive": 20, "fm": 8, "hybrid": 7,
              "granular": 4, "physical": 4, "additive": 3, "spectral": 2}
DIFFICULTY = {"sampling": 1.5, "physical": 1.4, "fm": 1.3, "wavetable": 1.2, "spectral": 1.2,
              "subtractive": 1.0, "granular": 0.8, "hybrid": 0.6, "additive": 0.5}
ALPHA, FLOOR, TRUNK_SHARE = 0.6, 0.04, 0.30

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

# Total params follow the capacity plan. Active params per voice are the compute budget:
# the per-voice slice of the trunk plus the routed part of one expert (the rest of the trunk
# runs once per instance, and large tables such as wavetable/codebook banks are gathered,
# not multiplied).
# name, total params, trunk share, expert floor (unused, kept for column order), active/voice, per-instance, control Hz,
#   polyphony, DSP FLOPs per voice-sample, cores allowed for NN, weight bytes (1=int8, 2=fp16),
#   min buffer, lookahead frames (NN may run this many hops ahead; adds that much latency)
TIERS = [
    ("Lite",     4e6,   .30, 40e3,  0.12e6, 0.10e6, 250, 16,  400, 1, 1,  32, 1),
    ("Standard", 16e6,  .30, 250e3, 0.35e6, 0.50e6, 250, 16,  900, 1, 2,  64, 2),
    ("Pro",      64e6,  .30, 1e6,   0.90e6, 1.50e6, 250, 16, 1800, 2, 2, 128, 2),
    ("Studio",   256e6, .30, 4e6,   4.0e6,  6.0e6,  250,  8, 4000, 4, 2, 512, 2),
]
SRS = [44100, 48000, 96000]
BUFS = [64, 128, 256, 512]

def split(total, trunk, floor=None):
    pool = total * (1 - trunk)
    return {k: pool * v for k, v in expert_shares().items()}, total * trunk

def report(t):
    name, tot, trunk, floor, act, inst, hz, poly, dsp, cores, wb, minbuf, la = t
    nn_gf = 2 * (act * poly + inst) * hz / 1e9
    frame_ms = 2 * (act * poly + inst) / (GFLOPS_CORE * 1e9 * cores) * 1e3
    hop_ms = 1e3 / hz
    dl = hop_ms * la * NN_DEADLINE_SHARE
    bw = (act + inst) * wb * hz / 1e9  # GB/s if every frame streams active weights from DRAM
    print(f"\n== {name}: {tot/1e6:.0f}M total, {act/1e6:.2f}M active/voice, {inst/1e6:.2f}M per-instance, "
          f"{hz} Hz control, {poly} voices, weights {tot*wb/1e6:.0f} MB")
    print(f"   NN sustained {nn_gf:.1f} GFLOP/s = {nn_gf/GFLOPS_CORE:.2f} ref cores "
          f"({nn_gf/GFLOPS_CORE*MIN_SPEC_SLOWDOWN:.2f} min-spec); frame {frame_ms:.2f} ms on {cores} core(s) "
          f"vs deadline {dl:.2f} ms (lookahead {la} = {hop_ms*la:.1f} ms latency) -> {'OK' if frame_ms*la <= dl*la and frame_ms <= dl else 'MISS'} ref, "
          f"{'OK' if frame_ms*MIN_SPEC_SLOWDOWN <= dl else 'MISS'} min-spec; "
          f"weight traffic {bw:.1f} GB/s, active set {(act+inst)*wb/1e6:.1f} MB")
    print("   audio-thread DSP per block (us, % of block):")
    print("     sr     " + "".join(f"{b:>15}" for b in BUFS))
    for sr in SRS:
        cells = []
        for b in BUFS:
            us = b * poly * dsp / (GFLOPS_CORE * 1e3) + BLOCK_OVERHEAD_US
            pct = us / (b / sr * 1e6) * 100
            flag = "!" if pct > DSP_SHARE_MAX * 100 else (" " if b >= minbuf else "-")
            cells.append(f"{us:6.0f} {pct:4.0f}%{flag}")
        print(f"     {sr:<6} " + " ".join(f"{c:>14}" for c in cells))
    vpc = GFLOPS_CORE * 1e9 / (2 * act * hz)
    ex, tr = split(tot, trunk, floor)
    dense = tr + max(ex.values())
    print(f"   voices per ref core: {vpc:.0f} as budgeted; {GFLOPS_CORE*1e9/(2*dense*hz):.2f} if the whole "
          f"trunk + largest expert ({dense/1e6:.1f}M) ran per voice")
    print(f"   capacity: trunk {tr/1e6:.2f}M; " + ", ".join(f"{k} {v/1e6:.2f}M" for k, v in sorted(ex.items(), key=lambda x: -x[1])))

if __name__ == "__main__":
    for t in TIERS:
        report(t)
