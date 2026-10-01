# Priority product completion — 2026-10-01

This implementation addresses the five priorities from the design compliance audit: DSP,
FX mixing, free composition, state migration and tier conversion. Training remains paused.
All dimensions/capacities below describe authored contracts, not legacy slot quotas.

## Advanced DSP

The registry contains 73 operations. Five new versioned operations execute uniformly in
float32/float64 and own no callback allocations:

| Operation | Authored layout and behavior | Cost / workspace |
| --- | --- | --- |
| `time.partitioned_convolution.v1` | Mono IR asset, power-of-two partition B=8..4096. Preparation converts the real IR into P complex 2B-point spectra (`partition,bin,complex`). State is `4+3B+4BP`: two-limb phase/cursor, input, overlap, queued output and frequency delay line. Forward FFT unnormalized; inverse normalized. Output onset B local updates. | O(1) ordinary update; O(PB+B log B) boundary update; 8B scalar scratch, at most 6B+5 sparse writes. IR size/layout edits reprepare. |
| `pitchtime.asset_resample_antialias.v1` | Mono sample, zero outside asset; speed in source frames/second, amplitude and integer base half-width 8..128. Cutoff `1/max(1,abs(speed/local_rate))`, support expands with ratio (maximum 32). Hann-windowed normalized sinc. State is fractional cursor. | O(width × max(1,ratio)); no persistent filter buffer. Not a streaming rate converter. |
| `physical.modal_rotation.v1` | Two rotation coordinates per mode; parameter triples frequency/positive T60/excitation gain. Arbitrary mode count. Explicit local-rate exponential decay. | O(modes), dense state of 2×modes; double long-decay fixture. |
| `physical.tuned_waveguide.v1` | State C+7: ring cursor, damping/fractional/dispersion memories, C samples. Parameters frequency, feedback, damping pole, dispersion allpass. Fractional tuning compensates filter phase delay at requested frequency; roundtrip must fit capacity. | O(1), eight sparse writes. Frequency must be below local Nyquist; abs(feedback/dispersion)<1. |
| `pitchtime.phase_vocoder_locked.v1` | Full Hermitian complex frame 2N plus valid flag, N power-of-two 8..4096. State 4N+2 includes analysis/synthesis phases, queued spectrum and flags. Analysis/synthesis hops and pitch ratio explicit. Positive-bin nearest-peak identity phase locking, bin redistribution, conjugate output. Output frame flag arrives one local update later. | O(N) per valid frame; 6N scratch; at most 4N+2 writes, one write on empty frame. Host must author STFT/transport/resynthesis timing. |

These are bounded production contracts with independent direct-convolution, alias-suppression,
decay, tuning and spectral reference tests. They do not establish complete parity with every
legacy physical model or the legacy nonuniform/PFFFT convolution scheduler. Partition boundaries
still require CPU qualification for a particular composition/hardware.

## FX mixing and state publication

`host_latency_samples` declares compensation for the wet program. The stereo dry history uses
`host_alignment_bytes >= latency*8`, allocated before publication. Parameter bypass and the
DAW bypass callback use the same aligned dry contribution. This is the global host mix; wet
branch alignment inside a recurrence remains explicitly authored. Input/output gain semantics
are unchanged. Zero latency requires no history buffer.

Instance `instance_id` is optional for compatibility; authored IDs must be unique. Composition
assigns fragment-qualified IDs automatically. An off-thread fingerprint covers initial state,
assets, callbacks, clocks, bindings, masks and allocation. Compatible value edits swap live
buffers in O(1) with respect to audio history length and retain the sample clock. Compiler
identity/shape checks precede its recurrent-buffer swap. The small instance/model list checks
remain proportional to configuration count. Structural or precision changes start with the new
prepared DSP state; held MIDI survives if the performance policy is compatible. Explicit Load
clears performance. Panic prepares a reset outside the callback and publishes once; its public
request is a non-realtime command. The standalone core `reset()` API is still proportional to
storage and should be used outside realtime callbacks for large programs.

## Free composition

Fragments may carry `runtimeModel` or previously scoped `runtimeModels`. Lowering preserves
each model's own clock, weights, recurrent context and latent sound. A scoped entry owns four
disjoint tensor slices through `offsets=[activation,parameters,routing,modulation]`, a local
`controls` document and `model`. Global/scoped overlap is rejected. Each model's allowance and
the composed `budget.compiler_model_bytes` must cover actual numeric storage. Predictions
cannot silently alter prepared presence/allocation. Failed publication restores that model's
values/context. This permits independent fragment compilers; it does not train a new joint model.

Numeric tensor, lock, X-value and matrix edits reconcile into the original fragment recipe.
Cross-fragment matrix overrides use `control_bindings` with stable source/destination fragment,
local elements, `layer_from_bottom` and `weight`; adding another fragment remaps them.
Recurrent binding coefficients likewise remain editable. Structural edits require explicit
“Use edited sound as base”. The panel's “Use current input policy” reconciles differing outer
performance/latent policies; local model latent inputs remain independent. Nested fragments
and arbitrarily deep control stacks are subject to explicit aggregate capacities.

Runtime predictions update base tensors before the authored control stack is mapped.
Held predictions remain the base between model events and host blocks, so X mappings are
not overwritten or accumulated repeatedly. Locked bottom parameters bypass model changes
and matrix contributions. Scoped compiler timing is measured only on due events. Explicit
execution reset restores authored base tensors, X defaults and GRU context before the next
(possibly delayed) compiler event.

The library, composition and native publication share complete lowering/policy validation.
Scoped model containers are arrays; IDs must be unique/nonempty and offsets exactly four
nonnegative integers. Global/scoped latent tensors count toward stored tensor capacity.
Model widths must fit native source capacities and available local latent storage. Explicit
global `compiler_model_bytes` is enforced; legacy independent model RAM allowances remain
supported. Output mixes require exact output/channel/gain triples with integer stereo indices.

## Four authored tier profiles

The plugin UI no longer offers tier switching or profile import (Sinin, 2026-10-01: the loaded
sound decides). A sound document may still carry `tier_profiles`, a JSON object with `Lite`,
`Standard`, `Pro`, `Studio` keys, for the programmatic `setTier` / `convertTier` path. Each value owns
a `budget` using the runtime capacity fields, including host alignment, compiler-model and
float64 boundary transport bytes where applicable. No new product default quotas are invented.
Old documents without profiles retain precision-only switching. Lite/Standard use float32;
Pro/Studio use float64. Padded ontology coordinates and every control remain available.

Native `convertTier` prepares candidates against the target budget, preserves parameter locks,
ontology, performance policy and declared latency, and publishes only a successful result.
The original representation is candidate zero. Additional authored documents may be supplied
in `tier_candidates[target]`. Optional `strategies=["merge_equivalent_instances"]` generates
a candidate that redirects identical eligible recurrences to one active instance, retaining
output-mix contributions and padded coordinates. It excludes models, composition, projections,
control stacks, locks and bound operands; only an explicit small pure-operation whitelist is
eligible. This is exact common recurrence elimination, not automatic arbitrary sound compression.

Changed candidates require `validation`:

- `cases`: each has `frames`, `timing_search_samples`, optional constant `inputs` for the
  six input categories, and optional `audio_frames`. Default `signal_type="pitched"` requires
  `fundamental_hz`; explicit `signal_type="unpitched"` supports noise/transient/FX fixtures.
- `maximum_errors`: nonnegative `relative_error` and `timing_error_samples`, plus
  `pitch_error_cents` for pitched cases.
- `evaluation_bytes`: allowance for source/candidate stereo comparison arrays.
- `evaluation_ticks`: total render-update allowance across all evaluated candidates/cases.

Evaluation renders the native execution off thread, measures normalized waveform RMS error,
autocorrelation pitch on the more energetic channel and stereo timing correlation, rejects
silence and unmeasurable pitched fixtures, and selects
the lowest-error admissible candidate. `tier_conversion` stores metrics and rejected reasons.
Insufficient capacity or failed identity constraints leave the active program/document unchanged.
Fixtures are reset-state explicit-input evidence, not guarantees for all future musical input.
Unpitched metrics report `pitch_measured=false` and `pitch_error_cents=null`, while still
protecting waveform and timing. Arbitrary trained topology search, silence fixture policy
and perceptual listening certification remain extensions; authored numerical capacities
still require product decisions.
