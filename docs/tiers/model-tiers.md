# Alfamedomega model size tiers (draft v5.3, 2026-09-29)

Four tiers. Every number is a planning estimate from `tier_budget.py` (same folder); change an
assumption there and re-run. Nothing here is measured yet. v2 rebases the budgets on the stack in
ADR 0001 (JUCE 8, ONNX Runtime CPU through anira) and a 250 Hz frame rate. v3 adopts the capacity
plan's tier totals (Claude Doc "Alfamedomega synthesis capacity plan"). v4 moves allocation to
14 DSP families with per-tier scaling, centralised in `allocation.json`, and adds per-tier DSP
cost caps for the new filter, modulation, nonlinear, delay, physical, spectral, convolution and
pitch/time modules. v5 applies the inference host's measured limits: each recurrent frame must
finish inside a declared max inference time under the 4 ms frame period (2 / 3 / 3 / 3.5 ms), so
Pro and Studio per-frame compute was cut to fit.

**Files:** `allocation.json` is the only place to change shares, tiers or DSP costs.
`tier_budget.py` reads it, prints the checks and writes `allocation_by_tier.json`, which DSP code
and the inference parameter layout should read instead of hard-coding anything.

## Tier table

| | **Lite** | **Standard** (default) | **Pro** | **Studio** |
|---|---|---|---|---|
| Total params (capacity plan) | 4 M | 16 M | 64 M | 256 M |
| Active params per voice (compute budget) | 0.12 M (3%) | 0.35 M (2.2%) | 0.65 M (1.0%) | 2.8 M (1.1%) |
| Voices per reference core | 167 | 57 | 31 | 7 |
| Per-instance (shared) net | 0.10 M | 0.5 M | 1.2 M | 5.0 M |
| Weights on disk / RAM | 4 MB (int8) | 32 MB (fp16) | 128 MB (fp16) | 512 MB (fp16) |
| Frame rate (NN output) | 250 Hz | 250 Hz | 250 Hz | 250 Hz |
| Polyphony at budget | 16 | 16 | 16 | 8 |
| anira inference threads | 1 | 1 | 2 | 4 |
| Frame compute, reference CPU (thread scaling 85% at 2, 70% at 4) | 0.45 ms | 1.27 ms | 1.41 ms | 2.01 ms |
| Frame compute, minimum spec | 0.9 ms | 2.5 ms | 2.8 ms (tight) | 4.0 ms (over) |
| Declared max inference time (measured setting) | 2 ms | 3 ms | 3 ms | 3.5 ms |
| Reported latency at 48 kHz (measured) | 8 ms | 12 ms | 12 ms | 20 ms (512 buffer) |
| Cores used, reference CPU | 0.10 | 0.30 | 0.58 | 1.37 |
| Cores used, minimum-spec CPU | 0.20 | 0.61 | 1.16 | 2.74 (not supported) |
| DSP cap per voice (FLOP/sample @ 48 kHz) | 400 | 900 | 1,800 | 4,000 |
| DSP cap per instance, stereo FX | 1,500 | 3,500 | 7,000 | 14,000 |
| Audio-thread DSP with every voice at cap, 128 @ 48 kHz | 111 µs, 4% | 239 µs, 9% | 468 µs, 18% | 599 µs, 22% |
| Same at 96 kHz (caps are per second) | 5% | 9% | 18% | 23% |
| Smallest buffer supported | 32 | 64 | 128 | 512 |
| Weight traffic from RAM | 0.1 GB/s | 0.4 GB/s | 0.9 GB/s | 3.9 GB/s |
| Target hardware | Any laptop, many instances | Any 4-core machine, several instances | 6+ core desktop / Apple M-series | 8+ core desktop; else freeze/offline render |

Reference CPU is one Zen 3 or Apple M1 class core; ONNX Runtime is assumed to sustain
10 GFLOP/s per core on these small batched graphs, at about 50 µs fixed cost per `Run()`.
Minimum spec is a ~2016 4-core laptop at half that. DSP cost is budgeted at the same
conservative 10 GFLOP/s; hand-written SIMD DSP should roughly halve every audio-thread figure.
DSP caps are defined at 48 kHz and enforced per second, so at 96 kHz the router gets half the
per-sample budget (fewer modules or lower quality settings) and the load stays under the 25%
audio-thread cap in every tier.

**Frame deadline rule (v5).** Recurrent frames cannot overlap, so a frame's inference must finish
inside the declared max inference time, which must stay under the 4 ms frame period. The spec
budgets each tier's frame at no more than 60% of that time on the reference CPU (headroom for OS
jitter), and requires it to fit the full declared time on minimum spec. Studio went from 4.0 M to
2.8 M active params per voice and Pro from 0.9 M to 0.65 M to meet this; total params and the
family split are unchanged. Studio stays unsupported on minimum spec. If Studio needs its old
per-voice size back, the alternatives are 4 voices instead of 8 (4.0 M per voice then fits), or a
200 Hz frame rate (5 ms period, about 1.3x more time per frame).

## Render approach and engine check

The network either emits control/spectral frames that DSP renders (frame-rate, DDSP-style), or
emits audio samples itself (sample-rate). Same active params per voice, 16 voices, 48 kHz, from
`render_compare.py`:

| Tier | Frame-rate 250 Hz, ORT | Frame-rate 250 Hz, hand SIMD | Sample-rate, ORT | Sample-rate, hand SIMD |
|---|---|---|---|---|
| Lite | 0.11 cores | 0.05 | 19 cores | 10 |
| Standard | 0.32 | 0.15 | 59 | 29 |
| Pro | 0.60 | 0.29 | 111 | 56 |
| Studio (8 voices) | 1.42 | 0.69 | 263 | 132 |

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

- **Lite:** Sounds right for bread-and-butter patches (basses, pads, plucks, leads). Expect 16
  partials per voice, one filter model, 2-operator FM, a coarse wavetable morph space, and
  approximate transients, 2x-oversampled saturation and a short (0.1 s) convolution IR.
  Physical/modal, noise, convolution/spatial and pitch/time get 20 to 60 k params each, so
  they give a recognizable flavour rather than faithful behaviour; spectral is per-instance FX
  only.
- **Standard:** The default. Every method is usable for production. Expect 32 partials, 4-op
  FM, sampled-instrument attacks that hold up solo, and physical models that respond to
  velocity and damping plausibly. Fast modulation (audio-rate FM sweeps, envelope segments under
  4 ms) is smoothed by the frame rate unless the DSP layer handles them.
- **Pro:** Near reference quality for the common methods, and the first tier where granular,
  spectral and physical-model patches should be hard to tell from their target DSP in a mix.
  Expect 48 partials and 6-op FM.
- **Studio:** The reference model the others are distilled from. 64 partials, 8-op FM, rich
  inharmonic and noisy behaviour, best transient and release modelling. It needs a 512-sample
  buffer and about 20 ms of latency, so it is for mixing and bouncing, not live playing on a laptop.

Suggested acceptance test per tier: multi-scale spectral distance to the Studio render (and to
ground-truth DSP renders) on a fixed patch set per method, with a ceiling per tier set once the
first models exist.

## How the budgets are met

1. Frame-rate rendering (above): the network predicts parameters for DSP modules; it never
   produces audio samples.
2. anira runs ONNX Runtime on its pre-allocated thread pool and turns the declared max inference
   time into a fixed latency reported to the host. The audio thread only renders DSP.
3. Voices are batched into one `Run()` per frame, so weights stream once per frame and the fixed
   call cost is paid once, not per voice.
4. Only the families a patch routes through are active. Total params set quality and download
   size; active params set CPU cost.
5. **Cost-aware routing.** Every DSP module has a cost per tier in `allocation.json`. The router
   (part of the shared trunk) may select, weight, reorder or bypass modules freely, but the sum
   of active module costs per voice may not exceed the tier's voice cap, and per-instance FX may
   not exceed the instance cap. Turning every voice module on at once would exceed the cap in
   Lite and Pro (441 / 853 / 2,006 / 3,546 FLOP/sample against caps of 400 / 900 / 1,800 / 4,000), so the cap is what bounds the worst case,
   and the audio thread's cost is known at `prepare()` time. All module instances up to the cap
   are pre-allocated there.

## DSP families and capacity allocation

Allocation now covers 15 families. Additive (PARTIALS) is its own family with its own capacity,
never merged with oscillator/subtractive or filter-based synthesis (Sinin, 2026-09-29). It is a
carve-out (`carve_outs` in `allocation.json`), not a base weight: after per-tier scaling it takes
3% in Lite and Standard, 5% in Pro and 6.5% in Studio, 60% of that from spectral and 40% from
wavetable, and no other family changes (capacity plan, 2026-09-29). At Standard that leaves
spectral at 3.2% and wavetable at 8.8%. In the larger tiers spectral falls below its Standard share
(2.6% in Pro, 2.3% in Studio); whether it should hold its Standard share is still open with Sinin. The base weights are Sinin's initial design target, which the
capacity-plan thread may revise; they sit in `allocation.json` under `base_weights_pct` and
nothing else needs to change when they do. Standard reproduces them exactly. Other tiers reshape
them with `share ∝ weight^gamma`:

- **Lite (gamma 1.3)** concentrates capacity on sampling, oscillators and filters.
- **Pro (0.8)** and **Studio (0.6)** progressively move capacity to rare and costly families, so
  Studio has 2 to 3x the relative share for noise, convolution/spatial and pitch/time that
  Standard has, rather than just bigger copies of the common modules.
- **Floors** (20 k / 60 k / 200 k / 800 k params) keep every family alive in every tier. In Lite
  the floored families (noise, convolution/spatial, pitch/time) are a small learned control map
  over their DSP module rather than a full expert. The floor is why pitch/time's Lite share
  (0.71%) sits slightly above its Standard share (0.54%).

The shared trunk (patch encoder, router, thin per-voice slice) keeps 30% of each tier; the
shares below split the other 70%.

| Family (base weight) | Lite 4 M | Standard 16 M | Pro 64 M | Studio 256 M |
|---|---|---|---|---|
| Sampling (20) | 24.3% · 0.68 M | 20.0% · 2.2 M | 17.1% · 7.7 M | 14.3% · 25.7 M |
| Oscillator + subtractive (15) | 16.7% · 0.47 M | 15.0% · 1.7 M | 13.6% · 6.1 M | 12.1% · 21.6 M |
| Additive / PARTIALS (carve-out 3/3/5/6.5%) | 3.0% · 0.08 M | 3.0% · 0.34 M | 5.0% · 2.2 M | 6.5% · 11.6 M |
| Filter / resonator (13) | 13.9% · 0.39 M | 13.0% · 1.5 M | 12.1% · 5.4 M | 11.1% · 19.9 M |
| Wavetable (10, funds additive) | 8.7% · 0.24 M | 8.8% · 0.99 M | 7.8% · 3.5 M | 6.9% · 12.3 M |
| FM (8) | 7.4% · 0.21 M | 8.0% · 0.90 M | 8.2% · 3.7 M | 8.3% · 14.8 M |
| Modulation (8) | 7.4% · 0.21 M | 8.0% · 0.90 M | 8.2% · 3.7 M | 8.3% · 14.8 M |
| Nonlinear (6) | 5.1% · 0.14 M | 6.0% · 0.67 M | 6.5% · 2.9 M | 7.0% · 12.5 M |
| Granular (5) | 4.0% · 0.11 M | 5.0% · 0.56 M | 5.6% · 2.5 M | 6.2% · 11.2 M |
| Spectral (5, funds additive) | 2.2% · 0.06 M | 3.2% · 0.36 M | 2.6% · 1.2 M | 2.3% · 4.2 M |
| Delay / diffusion (4) | 3.0% · 0.08 M | 4.0% · 0.45 M | 4.7% · 2.1 M | 5.5% · 9.8 M |
| Physical / modal (3) | 2.1% · 0.06 M | 3.0% · 0.34 M | 3.8% · 1.7 M | 4.6% · 8.2 M |
| Noise (1.5) | 0.8% · 0.02 M | 1.5% · 0.17 M | 2.1% · 0.97 M | 3.0% · 5.4 M |
| Convolution / spatial (1) | 0.7% · 0.02 M | 1.0% · 0.11 M | 1.6% · 0.70 M | 2.4% · 4.3 M |
| Pitch / time (0.5) | 0.7% · 0.02 M | 0.5% · 0.06 M | 0.9% · 0.40 M | 1.6% · 2.8 M |

Mapping to the existing engines: PARTIALS is the additive family, the modal bank is shared by filter/resonator and physical/modal, and the existing FFT layer feeds
both spectral modules. Hybrid patches are no longer a family: they are routes the router builds
across families.

## DSP limits per tier

Structured limits the DSP engines generate their `TierLimits` from (`dsp_limits` in each tier of
`allocation_by_tier.json`). `tier_budget.py` fails if any value exceeds its ceiling. Partials,
FM operators and grain layers are fixed by the v1 frame layout (raising them means retraining);
slots and mod routes are v2 limits; mod routes can go to 16 and resonator modes to 64.

| Limit | Lite | Standard | Pro | Studio | Ceiling |
|---|---|---|---|---|---|
| partials | 16 | 32 | 48 | 64 | 64 |
| fm_operators | 2 | 4 | 6 | 8 | 8 |
| grain_layers | 2 | 4 | 6 | 8 | 8 |
| polyphony | 16 | 16 | 16 | 8 |  |
| voice_slots | 2 | 3 | 4 | 4 | 4 |
| instance_slots | 2 | 3 | 4 | 4 | 4 |
| mod_routes | 4 | 6 | 8 | 8 | 8 |
| resonator_modes | 8 | 16 | 24 | 32 | 64 |
| osc_unison | 2 | 4 | 6 | 8 | 8 |

## DSP cost check

Costs per module and tier are in `allocation.json` (`module_cost`), with the quality setting each
number assumes (unison count, partials, oversampling factor, filter stages, grain count, modes,
IR length). Representative patches, as a percentage of the tier's cap:

| Patch (modules) | Lite | Standard | Pro | Studio |
|---|---|---|---|---|
| VA lead (osc, filter, mod, nonlinear, chorus, pan) | 34% | 28% | 27% | 22% |
| Additive pad (partials, filter, mod, chorus, pan) | 30% | 26% | 22% | 15% |
| Pluck with body (waveguide, modal bank, filter, mod, pan) | 29% | 23% | 21% | 15% |
| Hybrid heavy (wavetable, FM, filter, formant, nonlinear, mod, chorus, pan) | 42% | 34% | 33% | 26% |
| Convolution FX (IR 0.1 / 1 / 3 / 6 s, delay, width) | 79% | 61% | 41% | 47% |
| Spectral FX chain (STFT, delay, diffusion, width) | 44% | 36% | 19% | 19% |
| Every voice module at once | 110% | 95% | 111% | 89% |

Everything a real patch uses fits with room to spare; "everything at once" breaks the cap in Lite
and Pro, which the router prevents. Slot limits (below) usually bind first: Lite's 2 voice slots
cannot hold the hybrid-heavy chain, so it keeps its two highest-weighted processors. Notes for the DSP engines:

- **Convolution** is the most expensive single module per instance. Non-uniform partitioning
  (two partitions per size, doubling from the host block) keeps it at 1,100 to 6,400
  FLOP/sample. The large partitions must run on a pre-allocated background thread with a
  deadline, and IR FFTs are computed when the IR loads, never in the callback. At 32 to 64 sample
  buffers the first partition shrinks and the cost rises; the numbers assume 128.
- **Per-voice spectral processing** is Pro and Studio only; Lite and Standard get spectral as a
  per-instance FX, so the family never disappears.
- **Pitch/time**: in Lite it is resampling- and grain-based stretch; phase-vocoder pitch and time
  are per-instance from Standard up, and formant-preserving only in Studio. Offline-quality
  variants stay outside the real-time path.
- **Nonlinear** oversampling rises with tier (2x, 2x, 4x, 8x) and is the main source of Studio's
  extra per-voice cost.

## Tier-to-tier consistency

Smaller tiers are trained by distillation from Studio, family by family, so every tier plays the
same patches and a project can switch tier (for example Lite while writing, Studio for the bounce)
without the sound changing character. A patch that routes through more modules than a smaller
tier's cap allows keeps its highest-weighted modules and drops or downgrades the rest.

**Constraint on the architecture (unchanged from v3).** The budget allows only 1.0 to 3% of each
tier's parameters to run per voice per frame, so the trunk must mostly run once per instance,
family experts must be internally sparse, and bulky parameters should be lookup tables.

## Open questions

- Sampling scope (user audio only, or a bundled library) is open in the capacity plan and would
  change the split. The capacity-plan thread is revising the 14 base weights; edit
  `base_weights_pct` when it lands.
- DSP module costs are estimates; replace them with benchmark numbers from the DSP engines as each
  primitive lands, and re-run.
- Benchmark each tier with anira's harness; the 10 GFLOP/s and 50 µs ORT figures are the
  assumptions most likely to move.
- Per-voice expert routing inside one ONNX graph: if each patch must be a separate graph or
  session, the "one Run() per frame" assumption needs checking (ADR 0001 open question 3).
- Studio on GPU/NPU is out of scope for v1 per ADR 0001; it stays CPU with a 512 buffer or
  offline render.
