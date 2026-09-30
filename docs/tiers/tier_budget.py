"""Compute budget and capacity allocation for Alfamedomega model tiers.

Inputs live in allocation.json (the one place to change shares, tiers or DSP costs).
Running this prints the budget check and writes allocation_by_tier.json, which DSP code and
the inference parameter layout read.

Execution model (ADR 0001): a neural control network runs at a fixed frame rate on anira's
worker threads (ONNX Runtime CPU), a few frames ahead of the audio thread (latency reported to
the host). The audio thread only renders DSP modules driven by those frames.
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "allocation.json")))

GFLOPS_CORE = 10.0        # ONNX Runtime CPU via anira, one Zen3 / M1-class core; hand SIMD ~20
DSP_GFLOPS_CORE = 10.0    # conservative for DSP too; hand-written SIMD DSP should reach ~20
MIN_SPEC_SLOWDOWN = 2.0   # ~2016 4-core laptop
DSP_SHARE_MAX = 0.25      # audio-thread render may use at most 25% of each block's wall time
NN_DEADLINE_SHARE = 0.6   # budgeted frame time may use at most 60% of the declared max inference time
THREAD_EFFICIENCY = {1: 1.0, 2: 0.85, 4: 0.7}  # small-batch ORT graphs do not scale linearly
BLOCK_OVERHEAD_US = 10.0
TIER_NAMES = ["Lite", "Standard", "Pro", "Studio"]
SRS = [44100, 48000, 96000]
BUFS = [64, 128, 256, 512]


def weights():
    return {k: v for k, v in CFG["base_weights_pct"].items() if not k.startswith("_")}


def families():
    return list(weights()) + [co["family"] for co in CFG.get("carve_outs", [])]


def tier(name):
    return CFG["tiers"][name]


def expert_params(name):
    """Family capacity for one tier: share ~ w^gamma over the expert pool, with an absolute
    floor so no family vanishes; then carve-outs take a target share for a family out of the
    families named as its funders. Returns {family: params}."""
    t, w = tier(name), weights()
    pool = t["total_params"] * (1 - t["trunk_share"])
    floored = set()
    while True:
        s = sum(w[k] ** t["gamma"] for k in w if k not in floored)
        rest = pool - t["floor_params"] * len(floored)
        c = {k: t["floor_params"] if k in floored else rest * w[k] ** t["gamma"] / s for k in w}
        new = floored | {k for k in w if c[k] < t["floor_params"]}
        if new == floored:
            break
        floored = new
    for co in CFG.get("carve_outs", []):
        take = pool * co["target_pct_by_tier"][name] / 100
        for src, frac in co["funded_from"].items():
            c[src] -= take * frac
            assert c[src] >= t["floor_params"], f"{name}: carve-out drops {src} below floor"
        c[co["family"]] = take
    return c

def conv_cost(ir_s, channels, block=128, sr=48000):
    """Non-uniform partitioned convolution: two partitions per size, doubling from the block
    size until the IR is covered. FLOPs/sample = complex MACs + FFT pairs, per channel path."""
    need, size, parts, fft = ir_s * sr, block, 0, 0.0
    covered = 0
    while covered < need:
        for _ in range(2):
            if covered >= need:
                break
            covered += size
            parts += 1
        fft += 2 * 2.5 * 2 * size * math.log2(2 * size) / size  # real FFT fwd + inv of 2B
        size *= 2
    return channels * (8 * parts + fft)


def module_cost(mod, i):
    m = CFG["module_cost"][mod]
    if mod == "convolution":
        return conv_cost(m["ir_seconds"][i], m["channels"][i])
    return m["cost"][i]


def scenario_cost(mods, i):
    total, missing = 0.0, []
    for mod in mods:
        c = module_cost(mod, i)
        if c is None:
            missing.append(mod)
        else:
            total += c
    return total, missing


def worst_case(scope, i):
    return sum(module_cost(m, i) or 0 for m, v in CFG["module_cost"].items()
               if not m.startswith("_") and v["scope"] == scope)


def report(name, i):
    t = tier(name)
    act, inst, hz, poly = t["active_per_voice"], t["per_instance"], t["frame_hz"], t["polyphony"]
    flops_frame = 2 * (act * poly + inst)
    nn_gf = flops_frame * hz / 1e9
    threads = t["nn_threads"] * THREAD_EFFICIENCY[t["nn_threads"]]
    frame_ms = flops_frame / (GFLOPS_CORE * 1e9 * threads) * 1e3 + 0.05  # + ORT Run() overhead
    assert t["max_inference_ms"] < 1e3 / hz, "recurrent frames cannot overlap"
    dl = t["max_inference_ms"] * NN_DEADLINE_SHARE
    print(f"\n== {name}: {t['total_params']/1e6:.0f}M total, {act/1e6:.2f}M active/voice, "
          f"{inst/1e6:.2f}M per-instance, {hz} Hz frames, {poly} voices")
    print(f"   NN: {nn_gf:.1f} GFLOP/s = {nn_gf/GFLOPS_CORE:.2f} ref cores "
          f"({nn_gf/GFLOPS_CORE*MIN_SPEC_SLOWDOWN:.2f} min-spec); frame {frame_ms:.2f} ms on "
          f"{t['nn_threads']} thread(s) vs {dl:.2f} ms (60% of declared {t['max_inference_ms']} ms) -> "
          f"{'OK' if frame_ms <= dl else 'MISS'} ref, {'OK' if frame_ms*MIN_SPEC_SLOWDOWN <= t['max_inference_ms'] else 'MISS'} min-spec (full declared time); "
          f"voices/core {GFLOPS_CORE*1e9/(2*act*hz):.0f}; measured latency {t['measured_latency_ms_48k']} ms")

    vcap, icap = t["voice_dsp_cap"], t["instance_dsp_cap"]
    print(f"   DSP caps: voice {vcap} FLOP/sample, instance {icap} FLOP/sample")
    for sc, mods in CFG["scenarios"].items():
        if sc.startswith("_"):
            continue
        cost, missing = scenario_cost(mods, i)
        cap = icap if sc.startswith("fx") else vcap
        flag = "OK" if cost <= cap else "OVER"
        extra = f" (unavailable: {', '.join(missing)})" if missing else ""
        print(f"     {sc:17} {cost:7.0f}  {flag}{extra}")
    wv, wi = worst_case("voice", i), worst_case("instance", i)
    print(f"     {'every voice module':17} {wv:7.0f}  {'OK' if wv <= vcap else 'OVER -> router must enforce cap'}")
    conv = module_cost("convolution", i)
    print(f"     convolution alone {conv:7.0f}  (IR {CFG['module_cost']['convolution']['ir_seconds'][i]} s)")

    # Caps are defined at 48 kHz and enforced per second: at higher sample rates the router's
    # per-sample budget shrinks by 48k/sr (fewer modules or lower quality), so load is flat.
    per_sample = vcap * poly + icap
    print("   audio-thread DSP at caps (budget per second, so flat across sample rates), per block (us, % of block):")
    print("     sr     " + "".join(f"{b:>15}" for b in BUFS))
    for sr in SRS:
        cells = []
        for b in BUFS:
            us = b * per_sample * (48000 / sr) / (DSP_GFLOPS_CORE * 1e3) + BLOCK_OVERHEAD_US
            pct = us / (b / sr * 1e6) * 100
            flag = "!" if pct > DSP_SHARE_MAX * 100 else (" " if b >= t["min_buffer"] else "-")
            cells.append(f"{us:6.0f} {pct:4.0f}%{flag}")
        print(f"     {sr:<6} " + " ".join(f"{c:>14}" for c in cells))
    ceil = CFG["dsp_limit_ceilings"]
    for k, v in t["dsp_limits"].items():
        assert k not in ceil or v <= ceil[k], f"{name}: dsp_limits.{k}={v} exceeds ceiling {ceil[k]}"
    assert t["dsp_limits"]["polyphony"] == poly, f"{name}: dsp_limits.polyphony != polyphony"
    return {
        "dsp_limits": t["dsp_limits"],
        "total_params": t["total_params"],
        "trunk_params": t["total_params"] * t["trunk_share"],
        "family_params": {k: round(v) for k, v in expert_params(name).items()},
        "family_share_pct": {k: round(v / (t["total_params"] * (1 - t["trunk_share"])) * 100, 2)
                             for k, v in expert_params(name).items()},
        "active_per_voice": act, "per_instance": inst, "frame_hz": hz, "polyphony": poly,
        "audio_budget_guard": CFG["audio_budget_guard"],
        "voice_dsp_cap_flops_per_sample": vcap, "instance_dsp_cap_flops_per_sample": icap,
        "module_quality": {m: {"cost": module_cost(m, i), "quality": v["quality"]}
                           for m, v in CFG["module_cost"].items() if not m.startswith("_")},
    }


if __name__ == "__main__":
    out = {"_generated_by": "tier_budget.py from allocation.json; do not edit by hand"}
    for i, name in enumerate(TIER_NAMES):
        out[name] = report(name, i)
    print("\nFamily share of expert capacity (%):")
    print(f"  {'family':20}" + "".join(f"{n:>10}" for n in TIER_NAMES))
    for k in families():
        print(f"  {k:20}" + "".join(f"{out[n]['family_share_pct'][k]:>10.2f}" for n in TIER_NAMES)
              + "   " + " ".join(f"{out[n]['family_params'][k]/1e6:.2f}M" for n in TIER_NAMES))
    # Rule (Sinin, 2026-09-29): a family's share may shrink in larger tiers, but its absolute
    # parameter count must never decrease from one tier to the next. Checked for every family;
    # a violation fails the run before allocation_by_tier.json is written.
    shrinking = [f"{k}: {a} {out[a]['family_params'][k]:,} > {b} {out[b]['family_params'][k]:,}"
                 for k in families() for a, b in zip(TIER_NAMES, TIER_NAMES[1:])
                 if out[b]["family_params"][k] < out[a]["family_params"][k]]
    if shrinking:
        raise SystemExit("family capacity decreases from one tier to the next:\n  " + "\n  ".join(shrinking))
    print("\nMonotonic capacity check: every family's params grow from tier to tier. OK")
    json.dump(out, open(os.path.join(HERE, "allocation_by_tier.json"), "w"), indent=2)
