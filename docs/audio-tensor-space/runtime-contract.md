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
processing slots. Primitive descriptors declare state/output size and F/G callbacks.

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
These tests do not demonstrate trained inference, plugin integration or full primitive coverage.

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
