# Recurrent tensor runtime contract

This describes the initial CPU implementation in `src/tensor/include/alfa/tensor/Runtime.h`
and `src/tensor/src/Runtime.cpp`. It is deterministic lowering and execution infrastructure,
not a trained universal compiler or a stable serialized model ABI.

## Representation boundary

`Tensor` owns named axes, shape, floating-point values and optional binary presence masks.
`TensorView` is a borrowed view. Mask 0 denotes padding whose numeric value must be zero;
mask 1 denotes represented capacity, including genuine zero-valued controls. Missing masks
mean all elements are represented. Parameter/routing/modulation capacity excludes padding;
`storedTensorElements` separately bounds full padded storage. Active contiguous parameter
slices cannot select absent padding.
The six `Inputs` fields independently carry audio, MIDI, gestures, preference, latent sound
and context. An entirely empty view explicitly means absent input. Concrete feature encodings
are not inferred from legacy MIDI widths or synthesis methods.

`Configuration` carries activation, parameters, routing, modulation, primitive instances,
shared initial state, local initial state, optional affine projections and a binary parameter-lock
mask. Each instance identifies a deterministic
primitive, activation element, parameter slice, explicit clock and between-update output
policy. Instance count and parameter width are configuration-dependent; they are not fixed
processing slots. Primitive descriptors declare state/output size, F/G callbacks, optional
sparse-journal capacity and typed numerical workspace capacity.

`Runtime::prepare` validates and copies a configuration off the audio thread, allocates
persistent state/output storage, and produces inspectable allocation maps and a schedule.
A rejected configuration leaves the previous prepared runtime intact. Resource budgets are
supplied policy, not predefined tier names or per-method capacity shares.

## Explicit affine projection lowering

`Projection` in `Runtime.h` and its execution in `Runtime.cpp` provide an optional mathematical
binding from selected information to parameters:

```text
effective_parameter[d] = base_parameter[d] + sum(weight[w] * source[s])
```

The producer selects a typed audio, MIDI, gesture, preference, latent, context or current-state
source, its element, a destination parameter element, and a coefficient element from either
the routing or modulation tensor. Flattened coefficient selection explicitly identifies the
condition/time contribution; the runtime does not infer musical feature encodings or choose
conditions/time windows. One source may contribute to multiple destinations and multiple
sources may accumulate at one destination. This is one supported lowering operation, not a
mandated interpretation of every future tensor or a learned router.

Prepare validates represented destination/weight indices, state indices and `projectionTerms`
capacity. Each tick evaluates projections from X_t and the immutable S_t before callbacks.
An unavailable or padded selected source either rejects the event or contributes zero, according to
explicit `MissingSource::Reject` or `MissingSource::Zero`. Nonfinite effective parameters
reject the event without committing state. Projection lists and source/destination selection
changes require off-thread preparation; coefficients can change through `updateTensors`.

`parameterLocks` is an optional binary mask matching parameter storage. A locked parameter
retains its compiled base value: projections do not modify it, and `updateTensors` rejects a
publication changing that value. Changing locks or their protected bases requires explicit
recompilation. This protects numeric parameter authority in the current backend; it is not
an implementation of the complete semantic UI or perceptual constraints optimizer.
## Tick semantics

At a host sample event t, each due primitive sees the same immutable full S_t, its local slice,
X_t, selected effective parameters after optional projection, routing and modulation. G produces its output and F writes only its
own next-state slice. Outputs do not become same-event inputs to other callbacks. All updates
commit together after validation. This supports mutual recurrent influence without a DAG or
topological execution order. Not-due state persists.

The shared initial-state prefix is readable by every primitive. In this baseline it has no
independent write/reducer API: recurrent shared interaction reads the full snapshot, including
other primitive local state via the allocation map. Extending shared writes requires explicit
ownership and combination semantics; calling this unrestricted shared-state mutation would
misdescribe the implementation.

Clocks specify `periodSamples` and `phaseSamples` per instance. Period 1 permits sample/audio
rate; different periods represent independent control and slow updates. The configuration
chooses frequencies. Between updates an output explicitly holds its previous value or becomes
zero. Fractional clocks, asynchronous events and timestamp interpolation are not implemented.

Dense transitions write a prepared next-state slice. Sparse transitions instead return a
list of local index/value writes into a prepared journal; untouched state persists. Every
returned write must be finite, in range and unique within its local slice. Outputs and all
due dense/sparse updates commit together after validation. `sparse_state_writes` budgets the
sum of allocated journal capacities; it is not the length of persistent history.
Duplicate validation uses a bounded Patricia trie with at most twice the declared sparse-write
capacity in structural nodes. Each index takes bounded word-bit walks; validation neither scans
nor clears the full persistent state. `sparseValidationStorageBytes()` reports this allocated
structural workspace separately from Scalar numeric storage. Structural indices are not audio
memory or a second precision choice.

`Step::scratch` is temporary typed numerical workspace. `scratch_elements` bounds the
maximum descriptor workspace, reused sequentially across callbacks rather than summed across
instances. Each callback must initialize every scratch value it reads. F and G must not
assume scratch persists between them, between events or after a rejected update. Workspace
is neither persistent S nor a communication channel between primitives. Its numeric storage
uses the same Scalar as state and is included in the runtime storage report.

`tick` allocates no memory. Invalid inputs or nonfinite next state/output silence the output,
preserve committed state and do not advance time. Callback implementations must themselves
be allocation-free, lock-free and noexcept. Finite values alone do not establish stability of
arbitrary nonlinear feedback. `reset` restores S_0 and clears outputs.

## Migration and evidence

`Migration.h` retains opaque original project bytes and format identity, and accepts an
explicit semantic translator to prepare a tensor candidate. It does not switch plugin authority,
rewrite host automation or automatically convert existing presets. A failed translation or
capacity check preserves the previous candidate and original bytes.

The test target `alfa_tensor_tests` exercises recurrence, simultaneous shared-state reads,
independent clocks, malformed configurations, budget refusal, inactive padded instances,
invalid-state atomicity, six input categories, projection/lock behavior, allocation-free ticking
and migration retention.
Foundation tests alone do not demonstrate trained inference, plugin integration or full primitive
coverage; later native/model evidence is recorded separately in validation.md.

## Current limits

`updateTensors` publishes activation, parameter, routing and modulation values into the existing
allocation without allocation. Shapes and masks must match; activating an unallocated primitive
requires off-thread recompilation. Disabling an allocated primitive holds state and silences output;
reactivation resumes it. Publication must be serialized with ticking; this is not a cross-thread
handoff implementation. The baseline does not learn routing, train neural compilation, adapt costs to measured
hardware, optimize cross-tier compression, or normalize WAV recordings. Its output is allocated
primitive output storage, not a completed host audio-bus mixing contract. Concrete primitive
kernels consume the parameter slices produced by the explicit projection lowering. Other
routing operations and intermediate-representation transformations remain to be defined.

Training padding/masks and tier budgets share the representation rather than define feature
locks. Their exact current validation rules belong to the source contract as it evolves.
Neither this implementation nor the documentation fixes a universal ontology shape, tensor
serialization version, global control frequency or complete mathematical routing encoding.

## Initial mathematical kernel coverage

`Kernels.h` and `Kernels.cpp` provide sine, one-pole low-pass, feedback-cell and integer-delay
reference operations, with parameter units/ranges, input/output/state descriptions, complexity,
clock support and reuse metadata. The integer delay uses an O(length) shift register and is
not claimed as a production-optimized delay. These kernels consume explicit parameter slices
and scalar audio input. Runtime affine projection can supply their effective parameters;
the kernels do not silently interpret arbitrary routing/modulation tensors themselves.
Their callbacks and tests are a small numerical foundation, not coverage of the full universe.

## Serialized compiler, learned execution and native hosting

`Compiler.h` accepts `aom.tensor/1`, ontology `aom.math/1`, with four named
prediction tensors, dynamically described primitive instances, assets, explicit
projection terms, clocks, initial state, locks and independent capacity limits.
The compiler validates them and derives state allocation and an execution
schedule. Instance descriptors own their primitive callbacks for the program's
lifetime. The current registry contains 69 Universe operations plus four original mathematical
kernels, for 73 total: 48 reference operations, five production ring operations, two streaming
STFT operations, nine versioned nonlinear operations and five advanced production operations.
This catalog count is implementation
coverage, not proof that universal production DSP is complete. Cubic and normalized
Kaiser-windowed sinc8 asset interpolation reuse the legacy numerical kernels; neither is a
ratio-dependent anti-alias downsampler. [Production composition](production-composition.md)
records state layouts, workspace, frame transport and migration differences.

The host's explicit `outputMix` rows are `[primitive_output_element, channel,
gain]`. Mixing is performed after simultaneous recurrent updates; it does not
introduce a module chain or establish general tensor routing semantics.

`aom.runtime.gru/1` embeds an independently budgeted neural compiler: six input
category widths, explicit GRU weights and persistent neural context, four output
schemas, producer-defined bounds and inference clock. Native inference matches
PyTorch sequential vectors. It predicts tensor values/presence and never renders
waveforms. In-place predictions must match prepared shapes and masks; structural
changes require off-thread recompilation. Failed tensor publication also retains
previous neural context. Reset restores prepared tensor controls, original DSP
memory and clock; the model resets its own original context.

The plugin stages immutable compiled programs off-thread and transfers ownership
through preallocated queues. Runtime ticking and tensor publication are serialized
on the audio thread. New audio/MIDI/MPE input encoding belongs to that native host,
not to an instrument-method selector. Its three modes share the same compiler
contract. The legacy host remains a separate compatibility entry.

Offline WAV import, local preference/importance learning, hardware probes and
constrained tier candidate optimization are implemented in `training/aom`. These
later capabilities supersede the foundation-only status descriptions above;
[validation](validation.md) records their actual limitations.

An explicit off-thread `PrimitiveResolver` can resolve extension operation IDs
while retaining tensor serialization and state allocation contracts. Unknown
operations without a registered implementation are rejected. This is an open
mathematical operation boundary, not a closed list of instrument types.

## Product control layers and recurrent audio operands

`control_layers` is an arbitrary-length list of indexed X layers. Each layer owns
`values`, per-coordinate `ranges`, and a matrix with named destination/source axes.
Matrix multiplication adds into the next layer's declared base values; the last
layer adds into the parameter tensor. Parameter locks restore the locked base value
after these transformations. Current matrices are authored defaults, not trained
semantic mappings. Eight stable host automation ports address the first eight
highest coordinates; all remaining coordinates and layers remain directly editable.

An instance can declare `input_channels` and `audio_bindings`. Each binding names
an input, persistent state or `output_history` element, an operand channel, and a
routing/modulation weight. Every primitive reads the same previous-state/output
snapshot. Its operand buffer is allocated during preparation and counted in the
budget. This permits summed audio operands and recurrent interactions without
graph sorting. Missing operands have explicit zero/reject policy.

Validated control edits publish a prepared configuration at the block boundary.
An off-thread structural fingerprint includes stable instance identity, callbacks, clocks,
initial state, assets, allocation/masks and recurrent operand layout. Compatible edits swap live
state/output buffers at publication in constant time with respect to history length. Compiler
models retain recurrent buffers only when model identity and authority layout match. Incompatible
DSP/precision changes start from the prepared initial state, while held MIDI remains when the
performance policy matches. Explicit Load and Panic clear performance; Panic prepares its reset
off thread. Compilation, file I/O and undo snapshots stay outside the audio callback.

Composition supports independent scoped fragment models with disjoint tensor slices, explicit
clocks, local latent inputs and aggregate `compiler_model_bytes`. Numeric tensor/X/matrix edits
reconcile back to the recipe using stable fragment coordinates. Structural divergence requires
an explicit new base. Native tier conversion applies authored `tier_profiles`, prepares every
candidate under its capacity, preserves locks and measures changed candidates against explicit
sound/pitch/timing cases. See [completion contracts](product-completion.md).

The canonical product target is `Alfamedomega` in `src/plugin`. It assembles the
tensor host and shared legacy DSP/UI components. `ALFA_BUILD_LEGACY_PLUGIN=ON`
retains the original Afmo compatibility entry; the product retains the tensor
entry's Aftn identity. These identities are distinct because their state formats
and host automation contracts differ.

Numerical execution uses one scalar type per tier, including assets, learned
compiler weights and recurrent state. Lite/Standard use float32; Pro/Studio use
float64. See [the measured precision policy](../tiers/runtime-precision.md) for
storage, accuracy evidence and explicit native host conversion boundaries.

The primary editor shows the current configuration and inspection without a loading selector.
Its Sound library browses and inspects candidates independently; only an explicit Load action
publishes the selected sound. Opening the browser, selecting a row or viewing details does
not mutate the current executable configuration. Parameter edits, X gestures, explicit imports
and loading remain distinct actions with their own state/publication semantics.
