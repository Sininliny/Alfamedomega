# Alfamedomega: Universal Audio Tensor Compiler

Direction supplied by Sinin on 2026-09-30, including the recurrent runtime, tier specification
and dual-supervision clarifications. These are requirements; implementation status is recorded
in [runtime-contract.md](runtime-contract.md) and [migration.md](migration.md).

## Mission

Encode the accumulated history of digital audio synthesis and processing as a trainable
Audio Tensor Space. Learn how sounds are constructed, transformed and controlled. The scope
includes synthesizers, samplers, effects, physical models, granular and spectral processing,
modulation, routing, sound design workflows and future unknown audio devices.

The primary abstraction must not become a DSP graph, manually connected modules, fixed
signal chain, preset generator or a larger conventional synthesizer. Tensor representation
describes the computation space; deterministic DSP engineering makes it executable.

## Compilation and execution

Inputs are independent audio, MIDI, performance gesture, user preference, latent sound and
context tensors. Audio input is native. Instrument, FX and hybrid operation are required.
MIDI carries pitch, velocity, duration, timing, polyphonic relationships, MPE pressure, pitch
movement, gesture patterns and phrase context. Velocity is not hard-wired to volume.

The learned model predicts activation, parameter, routing and modulation relationships.
Compiler lowering produces an executable configuration containing:

1. Primitive activation tensor: which operations exist and their contribution.
2. Parameter tensor: their current parameters and temporal behavior.
3. Routing tensor: how information is shared across representations and destinations.
4. State allocation map: persistent memory needed by each operation.
5. Execution schedule: efficient state updates at the required timescales.

Modulation relationships and persistent temporal state are part of this contract, not
omitted because the compiler output has five principal categories.

At time step t:

```text
S_(t+1) = F(S_t, X_t, R_t)
Y_t     = G(S_t, X_t, R_t)
```

F and G are deterministic DSP operations. Feedback is recurrent state, not explicit graph
loops. A delay reads its stored state for output and updates memory from input and feedback;
a filter updates its memory and coefficients at its declared clock. The runtime must support
stateful, recurrent and feedback computation. Do not assume DAG execution, fixed slots,
method-based routing or a single global control rate such as 250 Hz.

Sample-rate state includes phase, filter memory and delay buffers. Audio-rate modulation
includes FM, fast envelopes and nonlinear feedback. Control tensors include routing decisions
and latent parameters; slow state includes preferences and performance context. These are
required timescale categories, not fixed frequencies imposed on every operation.

Shared latent/control state can influence multiple primitive updates simultaneously. The
initial runtime reads one immutable S_t and commits disjoint state updates together; see the
contract for the exact implementation boundary and limitations.

## DSP Primitive Universe

A primitive is a reusable mathematical audio operation. Its contract must identify input and
output representations, trainable and controllable parameters, modulation targets,
computational cost, supported rates and reusable count. The ontology is extensible; no fixed
method or slot enumeration defines it.

| Area | Required foundation |
|---|---|
| Oscillation and excitation | Sine, band-limited and wavetable oscillators; additive partials; noise; impulse, sample, granular and modal excitation |
| Harmonic and spectral | Partial manipulation, harmonic redistribution, spectral morphing, FFT, inverse FFT, spectral filtering, tilt and freeze |
| Filtering and resonance | Low-pass, high-pass, band-pass, state-variable and comb filters; resonators and modal resonators |
| Modulation | LFO, envelope, multi-stage envelope, random modulation, MIDI CC, MPE, gesture extraction and learned sources |
| Nonlinear | Saturation, waveshaping, distortion, clipping and feedback nonlinear systems |
| Time effects | Delay, feedback delay, chorus, flanger, phaser, diffusion, reverb and convolution |
| Physical/acoustic | Strings, body resonance, modal synthesis and excitation-response systems |
| Spatial | Stereo width, positioning, binaural processing, rooms and multichannel transformations |
| Pitch/time | Resampling, granular time stretch, phase vocoder, pitch transformation and formant preservation |

Listing an operation is not evidence of an implemented or validated kernel.

## Relationships and parameters

Routing has conceptual dimensions `source × destination × condition × time`. Its scope
includes audio, control, spectral and other intermediate representations, primitive
interactions, and any parameter capable of affecting humanly perceptible sound behavior.
One source can influence many destinations; one destination can receive many sources.
Parameter representation is conceptually `primitive × parameter × time`, not 32 fixed knobs.

Multiple primitives can contribute simultaneously. Activation weights need not form a
probability distribution. Sound identity need not belong to one synthesis method.

## Tiers and access

Lite, Standard, Pro and Studio share one tensor ontology. Each tier sets an executable
complexity budget: tensor dimensionality/resolution, active primitive count, routing
complexity, temporal state capacity, parameter precision, simultaneous processing capacity,
and model/runtime resources. No new numerical budgets are specified here.

The tiers are different computational resolutions of the same space. They must not be
different feature sets or impose artificial UI restrictions. Every executable preset can
expose its available controls. Lower tiers may approximate internal detail, while users
retain consistent indexed control layers, routing and expert tensor access.

All tiers share a padded training representation with explicit validity/capacity information.
Unused dimensions must not become real observations or consume active runtime resources merely
because they occupy padded storage. Shared representation enables common datasets, compatible
inference contracts, distillation and transfer of preset identity across tiers.

Parameter importance must be learned from sound state, preference, musical context,
performance and parameter relationships. Do not assign fixed importance to named controls.
Human constraints protect pitch correctness, timing stability, fundamental musical behavior
and user-locked controls. The model learns which remaining details preserve perceived identity.

Tier conversion is constrained optimization over the original representation and target
budget, maximizing perceived similarity and musical identity. Candidate strategies include
lower tensor resolution, merged activations, simpler routing, lower temporal detail, removal
of low-impact dimensions and approximated interactions. Do not simply delete modules or
substitute a predefined method. Conversion must respect protected musical properties and
locks; a failed budget check does not authorize silently changing the sound.

## Models and AOM Centre

Keep three model purposes distinct:

- Runtime model: predictable CPU use, low latency and DSP control generation.
- Preset learning model: offline WAV-to-editable-AOM reconstruction and deeper search.
- User preference model: local taste adaptation and exploration.

They do not share an assumed parameter budget. Offline import may take time, but must estimate
available CPU/GPU/memory capacity, stay within usable VRAM/RAM and avoid crashing the host.
Device categories such as laptops, 8-core desktops and high-memory/GPU workstations are
capacity-estimation examples, not mandatory hardware requirements.

WAV learning uses dual supervision: the primary target is the original canonical normalized
AOM representation **before perturbation**; the secondary target is the actual perturbed
rendered instance for reconstruction and parameter understanding. Do not train only for
perturbed recovery. Render controlled perturbations over multiple pitches and recording
conditions. Normalize irrelevant pitch, loudness and recording variation while preserving
timbre, articulation and temporal behavior. Detailed data/loss requirements are in
[training/METHOD.md](../../training/METHOD.md).

AOM Centre learns from audio together with construction representations, routing,
parameter behavior and performance context. Similar waveform matching alone is insufficient.

## User interaction and migration

Control layers are expandable and indexed as X0, X1 or X[layer,index]. Only bottom DSP
instance parameters carry semantic names such as filter cutoff. A higher control can affect
many lower controls through an editable matrix. Current configurations use explicitly authored
default mappings and ordering; future training can learn the mappings and perceptual ranking.
Do not label the current ordering as learned or add brightness/warmth macro definitions.
There is no fixed internal knob count: pagination and the finite host automation interface do
not limit the executable tensor dimensions. Raw tensor editing includes activation, routing,
state allocation, clocks and processing relationships. Locks and selected overrides must have
explicit authority over model suggestions. Training is deferred while the VST is completed.

Compatibility adjustments are authorized by Sinin's later response. Keep the legacy plugin
entry available through the compatibility build option while the main product uses the
tensor contract, recurrent runtime and shared DSP/UI components; authorization is not a
requirement to discard old data.
Do not claim an opaque legacy snapshot is a semantic preset conversion.

Unspecified directional choices must be raised with Sinin. Established DSP knowledge may
implement agreed mathematics; it does not authorize replacing the requested architecture
with familiar instrument assumptions.
