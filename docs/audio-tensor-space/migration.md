# Migration audit: Audio Tensor Space

Audit date: 2026-09-30. Synchronized base: `a7e7b71` on GitHub `main`.
The [mission](mission.md) defines the replacement direction; the [runtime contract](runtime-contract.md)
records the initial executable boundary. Legacy component documents remain historical implementation
references and cannot override the tensor requirements.

## Confirmed conflicts in the legacy implementation

| Evidence | Conflict |
|---|---|
| `src/dsp/include/alfa/dsp/SlotGraph.h` | Lower-triangular indexed processing slots express a DAG, rather than recurrent tensor execution. |
| `src/dsp/include/alfa/dsp/ControlFrame.h` | Fixed excitation arrays, slot widths and 250 Hz frames must not define the new representation or clocks. |
| `src/dsp/include/alfa/dsp/Primitive.h` | Voice/hop lifecycle does not by itself cover universal audio and intermediate-representation transformations. |
| `src/inference/MODEL_CONTRACT.md` and `InferenceHost.h` | 32 knobs, method selectors, named experts and two-part hybrids do not establish universal compiler inputs/outputs. |
| `training/alfamedomega_train.py` | Predefined method teachers, slot programs and legacy export contracts do not implement tensor construction learning. |
| `src/plugin/src/PluginProcessor.cpp` and plugin packaging | Existing instrument output rendering does not establish native audio-input FX/hybrid hosting. |
| Legacy UI and control-centre documents | Method selection and method-dependent knobs do not define layered semantic/tensor inspection or construction supervision. |

Mathematical kernels can be reused after validating their representations, states, parameter
semantics and clocks. A matching primitive name is not certification of compatibility.

## Decisions already supplied

- Routing spans any perceptually relevant audio/control/intermediate relationship and primitive
  interaction. It is not confined to velocity-to-volume or method-based destinations.
- Execution is S_(t+1)=F(S_t,X_t,R_t), Y_t=G(S_t,X_t,R_t), using deterministic DSP and persistent
  recurrent state. No DAG, fixed slots or global 250 Hz control rate is assumed.
- Tiers share one ontology and padded training representation. They limit internal capacity,
  not UI feature access. Context-dependent importance is learned; human constraints protect
  pitch, timing, musical behavior and user locks. Compression is optimization, not module deletion.
- WAV learning has a primary canonical target before perturbation and secondary rendered-instance
  target after perturbation. Pitch/loudness/recording invariance must retain articulation and
  temporal behavior. This answer is settled and must not be asked again.
- Offline preset learning, realtime compilation and preference adaptation have separate purposes
  and resource assumptions.
- Compatibility adjustments are allowed by the later explicit reply. Keep the parallel legacy
  plugin entry during stabilization and require an explicit semantic migration, rather than
  silently reinterpreting old data.

## Delivered foundation and remaining work

`src/tensor` adds a separate deterministic recurrent runtime with named tensor axes, independent
input categories, arbitrary configured instance counts, per-instance clocks, persistent memory,
allocation/schedule inspection, optional typed affine projections, parameter locks and resource validation. Simultaneous updates read one immutable
state snapshot and commit together. This replaces DAG assumptions in the new foundation while
leaving the legacy plugin path available.

Explicit presence masks distinguish absent padded dimensions from present zero-valued dimensions.
Represented parameter/routing/modulation capacity and full padded storage have separate limits.
The contiguous parameter backend does not permit active primitives to select absent padding.
These mechanisms support a shared representation; they do not implement cross-tier optimization.

The migration boundary preserves original bytes and requires a semantic translator to prepare a
candidate. It is not yet a converter for real legacy model/preset/automation formats. The new runtime
is connected to the native instrument/FX/hybrid host and supports imported GRU compiler models;
authored default sounds do not contain a trained universal model. Kernel coverage must be reported
from implemented code, not inferred from the required primitive-universe list.

Still required for the complete mission:

- Perceptual and musical lock authority beyond implemented numeric parameter locks and explicit
  indexed control layers.
- Complete primitive metadata, executable kernels and numerical/state validation across the universe.
- Evolution/compatibility policy for the current versioned tensor serialization and wider
  intermediate representations.
- Broad learned routing and multi-category performance/context generalization beyond imported
  models and explicit authored encodings.
- Shared padded construction datasets, dual-supervised WAV learning, validated normalization,
  distillation, learned importance and constrained tier compression.
- Product validation of layered controls, overrides and hardware-aware offline import at scale.
- Measured latency, CPU/memory limits and host-safety evidence beyond callback unit tests.

These are unfinished implementation areas, not unanswered versions of the decisions above.
New directional ambiguities must be raised with Sinin before committing to a universal meaning.

## Executable implementation update

The native `src/tensor_plugin` entry now hosts instrument, FX and hybrid audio
buses, MIDI/MPE input tensors, off-thread compilation/publication, recurrent DSP
and independently clocked GRU tensor predictions. The canonical product entry
uses indexed X control layers, editable many-to-many matrices and semantic names
only at the primitive parameter level. Tensor composition retains independently
authored fragments and explicit recurrent bindings. Model training is paused
during the current native plugin migration. The optional legacy build retains
its original state semantics; reused DSP/UI components are shared by the product.

`Compiler.h` defines the `aom.tensor/1` / `aom.math/1` JSON boundary; state allocation
and independent schedules are derived from validated programs. `Universe.h`
registers 64 operations: 48 reference, five production ring-state, two streaming
STFT and nine versioned nonlinear operations. Together with the four initial mathematical
kernels this is 68 operations. This count does not establish production parity with every
legacy algorithm or the complete history of audio computation. Explicit
immutable assets hold tables, IRs, frames, HRTFs and envelope segment definitions.
`Model.h` implements deterministic CPU inference for `aom.runtime.gru/1`, with
independent model budget, explicit clock, bounded predictions and user locks.

`training/aom` now provides actual dual-head training, native DSP rendering,
WAV-to-editable-template inference, runtime model training/export, local pairwise
preference learning, context-dependent importance and measured constrained tier
candidate search. AMO Centre exposes tensor workflows separately from legacy
method training. See [validation](validation.md) for reproducible execution and
remaining generalization/product limits. Earlier foundation-only observations
above are retained as the audit history, not the current implementation status.

Production extraction now includes constant-time ring history updates, explicit FDN8 and causal
FIR, streaming windowed FFT/overlap-add with typed numerical workspace, and explicitly selected
1x/2x/4x nonlinear operations. Existing IDs and factory sounds are not silently converted.
Preserve each operation's latency, initial state, assets and error semantics when migrating;
[production composition](production-composition.md) explains the concrete contracts. Optimized
partitioned convolution, anti-alias resampling and richer legacy physical/phase-vocoder behaviors
still require adaptation and validation. The primary editor's configuration display is read-only;
Sound library browsing/inspection leaves playback intact until an explicit Load.
