"""Compare Alfamedomega tiers under two render approaches and two inference engines.
Frame-rate (DDSP-style): NN emits control/spectral frames, DSP renders audio.
Sample-rate: NN emits audio samples directly (same active params per voice).
Engines: hand-tuned SIMD (20 GFLOP/s/core, ~5 us per call) vs ONNX Runtime CPU via anira
(assumed 10 GFLOP/s/core at these batch sizes, ~50 us fixed per Run)."""
from tier_budget import TIERS

ENGINES = {"hand SIMD": (20.0, 5.0), "ORT+anira": (10.0, 50.0)}
SR = 48000

def frame_cost(act, inst, poly, hz, gflops, call_us):
    flops = 2 * (act * poly + inst)
    per_frame_ms = flops / (gflops * 1e9) * 1e3 + call_us / 1e3
    cores = per_frame_ms * hz / 1e3
    return cores, per_frame_ms

print(f"{'tier':9} {'engine':10} {'rate':>7} {'cores':>7} {'frame ms':>9} {'max lat ms':>10}")
for t in TIERS:
    name, tot, trunk, floor, act, inst, hz, poly, dsp, ncores, wb, minbuf, la = t
    for eng, (g, call) in ENGINES.items():
        for rate in (hz, 250, 100):
            c, f = frame_cost(act, inst, poly, rate, g * ncores, call)
            c *= ncores  # express as single-core equivalents
            print(f"{name:9} {eng:10} {rate:>5}Hz {c:7.2f} {f:9.2f} {f*2 + 1e3/rate:10.1f}")
    # sample-rate render: one net evaluation per sample per voice (batched over voices)
    for eng, (g, call) in ENGINES.items():
        cores = 2 * (act * poly + inst) * SR / (g * 1e9)
        print(f"{name:9} {eng:10} {'48kHz':>7} {cores:7.0f}  (sample-rate render; call overhead ignored)")
    fit = 0.25 * 20e9 / (2 * SR * poly)
    print(f"{'':9} sample-rate net that fits 0.25 core at {poly} voices: {fit/1e3:.1f}k params/voice\n")
