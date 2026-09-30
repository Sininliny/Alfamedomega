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
is not connected to plugin audio buses or a trained compiler. Kernel coverage must be reported from
implemented code, not inferred from the required primitive-universe list.

Still required for the complete mission:

- Host integration for streamed validated predictions and layered semantic lock authority beyond
  the implemented numeric publication/projection locks.
- Complete primitive metadata, executable kernels and numerical/state validation across the universe.
- Defined tensor serialization, intermediate representation semantics and extension compatibility.
- Instrument, FX and hybrid host integration with live audio preserved.
- Learned routing and multi-category performance/context processing.
- Shared padded construction datasets, dual-supervised WAV learning, validated normalization,
  distillation, learned importance and constrained tier compression.
- Layered UI inspection/overrides and hardware-aware offline import.
- Measured latency, CPU/memory limits and host-safety evidence beyond callback unit tests.

These are unfinished implementation areas, not unanswered versions of the decisions above.
New directional ambiguities must be raised with Sinin before committing to a universal meaning.
