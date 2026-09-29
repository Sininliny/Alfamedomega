# Alfamedomega model size tiers (draft v3, 2026-09-29)

Four tiers. Every number is a planning estimate from `tier_budget.py` (same folder); change an
assumption there and re-run. Nothing here is measured yet. v2 rebases the budgets on the stack in
ADR 0001 (JUCE 8, ONNX Runtime CPU through anira) and a 250 Hz frame rate. v3 adopts the capacity
plan's tier totals and expert shares (Claude Doc "Alfamedomega synthesis capacity plan").

## Tier table

| | **Lite** | **Standard** (default) | **Pro** | **Studio** |
|---|---|---|---|---|
| Total params (capacity plan) | 4 M | 16 M | 64 M | 256 M |
| Active params per voice (compute budget) | 0.12 M (3%) | 0.35 M (2.2%) | 0.9 M (1.4%) | 4.0 M (1.6%) |
| Voices per reference core | 167 | 57 | 22 | 5 |
| Per-instance (shared) net | 0.10 M | 0.5 M | 1.5 M | 6.0 M |
| Weights on disk / RAM | 4 MB (int8) | 32 MB (fp16) | 128 MB (fp16) | 512 MB (fp16) |
| Frame rate (NN output) | 250 Hz | 250 Hz | 250 Hz | 250 Hz |
| Polyphony at budget | 16 | 16 | 16 | 8 |
| anira inference threads | 1 | 1 | 2 | 4 |
| Frame compute, reference CPU | 0.4 ms | 1.2 ms | 1.6 ms | 1.9 ms |
| Declared max inference time (anira) | 2 ms | 4 ms | 4 ms | 8 ms |
| Reported latency (≈ one frame + max inference) | ~6 ms | ~8 ms | ~8 ms | ~12 ms |
| Cores used, reference CPU | 0.10 | 0.30 | 0.8 | 1.9 |
| Cores used, minimum-spec CPU | 0.20 | 0.61 | 1.6 (tight) | 3.8 (not supported) |
| Audio-thread DSP, 128 @ 48 kHz | 92 µs, 3% | 194 µs, 7% | 379 µs, 14% | 420 µs, 16% |
| Audio-thread DSP, 128 @ 96 kHz | 7% | 15% | 28% (over 25% cap) | 31% (over cap) |
| Smallest buffer supported | 32 | 64 | 128 | 512 |
| Weight traffic from RAM | 0.1 GB/s | 0.4 GB/s | 1.2 GB/s | 5 GB/s |
| Target hardware | Any laptop, many instances | Any 4-core machine, several instances | 6+ core desktop / Apple M-series | 8+ core desktop; else freeze/offline render |

Reference CPU is one Zen 3 or Apple M1 class core; ONNX Runtime is assumed to sustain
10 GFLOP/s per core on these small batched graphs, at about 50 µs fixed cost per `Run()`.
Minimum spec is a ~2016 4-core laptop at half that. DSP cost is budgeted at the same
conservative rate; hand-written SIMD DSP should halve it, which brings Pro and Studio at 96 kHz
back under the 25% audio-thread cap (the alternative is rendering at 48 kHz and oversampling).

## Render approach and engine check

The network either emits control/spectral frames that DSP renders (frame-rate, DDSP-style), or
emits audio samples itself (sample-rate). Same active params per voice, 16 voices, 48 kHz, from
`render_compare.py`:

| Tier | Frame-rate 250 Hz, ORT | Frame-rate 250 Hz, hand SIMD | Sample-rate, ORT | Sample-rate, hand SIMD |
|---|---|---|---|---|
| Lite | 0.11 cores | 0.05 | 19 cores | 10 |
| Standard | 0.32 | 0.15 | 59 | 29 |
| Pro | 0.82 | 0.40 | 153 | 76 |
| Studio (8 voices) | 1.95 | 0.95 | 365 | 182 |

Sample-rate rendering is 190x more expensive at 48 kHz and no tier survives it. A sample-rate net
that fits a quarter of a core at 16 voices has about **3k parameters per voice**, which is a
waveshaper, not a synthesiser. **The tiers therefore require the frame-rate approach**; a tiny
sample-rate network is only viable as an optional per-voice residual/waveshaper stage.

Frame rate: 250 Hz sits at the top of the ADR's 50 to 250 Hz range. It costs 1.5 to 2x less than the 375
to 500 Hz rates in v1, and ONNX Runtime at 250 Hz costs about what hand-tuned kernels cost at
500 Hz. To keep attacks sharp at 4 ms frames, note-on triggers, envelopes and oscillator phase
resets live in the DSP layer (the network sets their parameters), so transient timing never
waits on a frame. Going down to 100 Hz saves another 2.5x if tests show no audible loss.

## What the tiers trade (fidelity)

- **Lite:** Sounds right for bread-and-butter patches (basses, pads, plucks, leads). Expect ~32
  harmonics per voice, one filter model, 2-operator FM, a coarse wavetable morph space, and
  approximate transients. Hybrid, granular, spectral and additive have only 0.11 to
  0.13 M params each and fall back to trunk DSP with a small control map, so they give a
  recognizable flavour rather than faithful behaviour.
- **Standard:** The default. Every method is usable for production. Expect ~64 harmonics, 4-op
  FM, sampled-instrument attacks that hold up solo, and physical models that respond to
  velocity and damping plausibly. Fast modulation (audio-rate FM sweeps, envelope segments under
  4 ms) is smoothed by the frame rate unless the DSP layer handles them.
- **Pro:** Near reference quality for the common methods, and the first tier where granular,
  spectral and physical-model patches should be hard to tell from their target DSP in a mix.
  Expect ~128 harmonics and 6-op FM.
- **Studio:** The reference model the others are distilled from. Full-band detail, 8-op FM, rich
  inharmonic and noisy behaviour, best transient and release modelling. It needs a 512-sample
  buffer and about 12 ms of latency, so it is for mixing and bouncing, not live playing on a laptop.

Suggested acceptance test per tier: multi-scale spectral distance to the Studio render (and to
ground-truth DSP renders) on a fixed patch set per method, with a ceiling per tier set once the
first models exist.

## How the budgets are met

1. Frame-rate rendering (above): the network predicts parameters for DSP modules (oscillator
   banks, wavetables, FM operators, grain engines, waveguides, filtered noise).
2. anira runs ONNX Runtime on its pre-allocated thread pool and turns the declared max inference
   time into a fixed latency reported to the host. The audio thread only renders DSP.
3. Voices are batched into one `Run()` per frame, so weights stream once per frame and the fixed
   call cost is paid once, not per voice.
4. Only the experts a patch uses are active: a shared trunk plus one method expert (two for
   hybrid patches). Total params set quality and disk size; active params set CPU cost.

## Tie to the capacity plan

Tier totals, the 30% shared trunk and the expert shares come from the capacity plan, whose
formula is reproduced in `tier_budget.py` (`PREVALENCE`, `DIFFICULTY`, `ALPHA`, `FLOOR`). Change a
prevalence or difficulty value there and every tier re-derives. If sampling ships a bundled
library, raise `DIFFICULTY["sampling"]` (and possibly its prevalence) and re-run.

| Expert (share) | Lite 4 M | Standard 16 M | Pro 64 M | Studio 256 M |
|---|---|---|---|---|
| Shared trunk (30% of total) | 1.20 M | 4.80 M | 19.2 M | 76.8 M |
| Sampling (29%) | 0.80 M | 3.22 M | 12.9 M | 51.5 M |
| Wavetable (19%) | 0.53 M | 2.14 M | 8.6 M | 34.2 M |
| Subtractive (15%) | 0.42 M | 1.68 M | 6.7 M | 26.9 M |
| FM (11%) | 0.32 M | 1.26 M | 5.1 M | 20.2 M |
| Physical modelling (8%) | 0.22 M | 0.90 M | 3.6 M | 14.3 M |
| Hybrid (5%) | 0.13 M | 0.54 M | 2.2 M | 8.6 M |
| Granular (5%) | 0.13 M | 0.51 M | 2.1 M | 8.2 M |
| Spectral (4%) | 0.13 M | 0.51 M | 2.0 M | 8.1 M |
| Additive (4%, floor) | 0.11 M | 0.45 M | 1.8 M | 7.2 M |

Per the capacity plan, Lite experts under about 0.15 M (hybrid, granular, spectral, additive)
fall back to trunk DSP with a small learned control map.

**Constraint this puts on the architecture.** The budget allows only 1.4 to 3% of each tier's
parameters to run per voice per frame. If the whole trunk plus the largest expert ran for every
voice, a reference core would manage 10 voices in Lite, 2.5 in Standard, 0.6 in Pro and 0.16 in
Studio, so the totals are only affordable if:

1. most of the trunk runs once per instance (timbre/patch encoding), with a thin per-voice slice;
2. each expert is internally sparse (routed sub-experts, top-1 or top-2 per frame); and
3. bulky parameters are lookup tables (wavetable frames, learned codebooks) that are gathered
   rather than multiplied.

The totals set quality and download size; the active budget sets CPU cost. If the model design
cannot get that sparse, the totals have to shrink toward the active budget, not the other way
round.

Smaller tiers are trained by distillation from Studio, method by method, so every tier plays the
same patches and a project can switch tier (for example Lite while writing, Studio for the bounce)
without the sound changing character.

## Open questions

- Sampling scope (user audio only, or a bundled library) is open in the capacity plan and would
  change the split.
- Benchmark each tier with anira's harness; the 10 GFLOP/s and 50 µs ORT figures are the
  assumptions most likely to move.
- Per-voice expert routing inside one ONNX graph: if each patch must be a separate graph or
  session, the "one Run() per frame" assumption needs checking (ADR 0001 open question 3).
- Studio on GPU/NPU is out of scope for v1 per ADR 0001; it stays CPU with a 512 buffer or
  offline render.
