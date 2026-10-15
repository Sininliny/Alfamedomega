# Production DSP operations: composition and explicit migration

`Production.h`/`Production.cpp` provide versioned mathematical operations that use the runtime's sparse transactional state journal. They supplement the existing primitive IDs. They do not replace old presets, reinterpret old state buffers, or introduce a new instrument/effect family selector.

The computation remains `S_(t+1)=F(S_t,X_t,R_t)` and `Y_t=G(S_t,X_t,R_t)`. Both callbacks read the same immutable old state. Sparse updates list only changed local indices; the runtime validates every candidate output/write and commits the complete due event together. A rejected event commits neither a partial feedback update nor another primitive's state. Unwritten history persists. Capacity and journal allocation occur during compilation, not on the audio callback.

## Primitive contracts

| Versioned operation | Parameters and persistent state | Work per update |
|---|---|---|
| `time.ring_delay.v1` | `[integer delay, feedback]`; `S=[cursor low16, cursor high16, C history values]`; `1<=delay<=C`, `abs(feedback)<1`. Reads the selected previous input/feedback value, writes `input+feedback*delayed`. | Constant-time read/update; 3 journal entries. |
| `time.ring_delay_cubic.v1` | Same state and feedback equation, `C>=4`, fractional `2<=delay<=C-2`. The four-point Catmull-Rom read reuses the typed legacy Hermite implementation. Delay may be changed on every scheduled update. | Four history reads; 3 journal entries. No internal LFO or inferred modulation. |
| `time.ring_allpass.v1` | Same state, `C>=2`, fractional `1<=delay<=C-1`, `abs(gain)<1`. Linear read `w`; writes `v=input+gain*w`; outputs `w-gain*v`. | Two history reads; 3 journal entries. Fixed integer delay has the Schroeder allpass response; fractional linear reads or changing delay do not imply a flat magnitude response. |
| `time.fdn_hadamard8.v1` | Eight explicitly declared operands and outputs; repeated `[delay_i,gain_i,damping_i]` triples. Each line stores `[cursor low16, high16, damping memory, C history values]`; total state `8*(C+3)`. Gains satisfy `abs(g_i)<1`; damping pole is in `[0,1)`. | 8 integer delay reads, 8 one-pole updates, shared typed normalized Hadamard8 butterfly, 32 journal entries. |
| `time.causal_fir_ring.v1` | No scalar parameters. Asset 0 is the editable causal coefficient tensor `h[0..K-1]`, `1<=K<=C+1`; same two-limb cursor and `C` input history values. | `O(K)` direct FIR readout, constant-time history update and 3 journal entries. No FFT partitions or shifting history. |

Delay units are **scheduled updates**. At a period of one host sample, a delay of `D` means `D` samples; at another clock period the corresponding physical duration changes. The compiler explicitly chooses each clock. These primitives do not impose a global sample/control rate.

The two cursor limbs each hold an exact unsigned 16-bit integer in the tier's scalar type. They represent the next write index without requiring a single float32 value to count beyond its exact integer range. History and damping memories use the same `Scalar` as the complete executable configuration. The maximum addressable ring capacity is the explicit 32-bit cursor range; practical capacity remains bounded by declared state and hardware memory budgets.

The FDN operation's arity of eight is a specific normalized-vector transform inherited from the old DSP code, not a global runtime slot limit. It contains no room preset, forced line-length pattern, modulation LFO, shelf-filter stack, freeze policy, or reverb chain. Those behaviors require explicitly represented parameters/state/operations. Multiple copies and other matrix transforms may coexist in the same tensor space.

The causal FIR readout is `h[0]*X_t + sum(h[k]*X_(t-k))`, so its direct tap does not acquire an extra implementation sample. Stereo/multichannel interpretation, input bindings and output mixing remain explicit. Padded asset taps represent zero coefficients.

## Composition

Primitive state slices are independent allocations in one recurrent state tensor. Input operands can be bound from audio, control-derived values, shared state and committed output history. A binding to committed output history means previous-event information; it does not read another primitive's uncommitted current output. Same-event coupling requiring a mathematical simultaneous solution must be represented by an operation that implements that solution, rather than by changing instance order.

Changing delay, feedback, damping, FIR assets or activation has explicit meanings in these operations. High-level X matrices can fan out to those parameter coordinates; the runtime does not invent semantic names or interpret arbitrary routing weights. State capacity changes require recompilation. A zero activation in the current runtime holds allocated state and silences output; reactivation resumes that history. Reset restores declared initial state and controls, rather than leaving cursor metadata behind.

Uniform precision applies to parameters, coefficients, interpolation and all persistent audio memories: Lite/Standard execute float32; Pro/Studio execute float64. JSON example capacities below are requirements of their small demonstrations, not invented tier limits. Changing tier must retain the same ontology and choose a matching `scalar_type`; it does not remove parameter dimensions or substitute a different primitive.

## Native composition workflow

Open **Tensor space → Compose fragments** in the canonical plugin. The current
sound becomes F0, retaining its tensors, assets, controls and clocks. Enter a
unique fragment ID before inserting a JSON fragment, cloning a fragment or
choosing **Create primitive**. The primitive authoring panel lists the complete
catalog and exposes state capacity, initial values, audio operands, parameter
arrays, immutable assets, routing/modulation coefficients, output mix and each
instance's period/phase. Its metadata describes the selected operation; there
are no guessed parameter values or implicit output connections.

Set the composition's explicit budget to fit the intended combination, then
prepare the recipe. Bind a fragment's committed output history or state to
another fragment's audio operand or parameter using an explicit coefficient.
Both directions are allowed; no DAG ordering is inferred. **Apply to sound**
publishes a validated executable configuration with Undo/Redo. Saving the sound
also retains its composition recipe and stable fragment identities. Independent
fragment control stacks are initially merged as block matrices and remain
editable through the X layer interface.

An edited flattened sound is protected by a fingerprint: opening its old recipe
cannot silently overwrite new values. **Use edited sound as base** explicitly
retains the edited executable as F0. If the live sound changes while the dialog
is open, publishing refuses and asks the user to reopen it with the current
sound. The importer also refuses incompatible performance/context policies
until they are explicitly reconciled in the recipe. It never guesses shared
musical meanings.

**Import WAV asset** reads an explicitly selected source channel, retains its
source sampling rate as metadata, and appends the decoded samples as an immutable
asset. It performs no resampling, channel mixing or parameter mapping. The JUCE
decoder is an explicit float32 source boundary; the compiled asset uses the
configuration's uniform Scalar. Declared storage capacity is checked before
allocating the PCM buffer. This is sample asset import, not the deferred WAV
learning model.

## Standalone reusable examples

The following are complete single-primitive executable JSON documents, usable as standalone FX examples or as explicit fragments copied into a larger tensor configuration:

- [Ring delay](../../programs/primitives/ring-delay.aom.json): Standard/float32, 64-sample history, delay 16, feedback 0.5, one input binding and 3 journal entries.
- [Hadamard8 FDN](../../programs/primitives/fdn-hadamard8.aom.json): Pro/float64, eight histories of capacity 16, independent delay/gain/damping triples, only operand 0 excited by input channel 0, 32 journal entries. The host example averages eight outputs and duplicates that scalar to two output channels; it makes no spatial rendering claim.
- [Causal FIR](../../programs/primitives/causal-fir.aom.json): Standard/float32, four explicit coefficients, four history slots, one audio operand and 3 journal entries. The current input tap is included in the readout.

When composing fragments, reassign activation indices, parameter slices, binding weight coordinates, initial-state slices and output indices. Recalculate the full budget. `stored_tensor_elements` includes tensor storage, asset values and allocated audio operand values; `projection_terms` also counts audio bindings. `sparse_state_writes` counts reserved journal entries, not history length. A float32 fragment cannot be silently inserted into a float64 executable through a hidden local buffer: parse/prepare the complete configuration with its uniform scalar type.

The float64 FDN example declares `input_capacity=[2,1280,48,5,0,8]` and `boundary_conversion_bytes=10744` for the native C ABI's audio/MIDI/gesture/preference/latent/context boundary; a seventh width, for the wave feature knobs, may follow and is zero when absent. The plugin converts its eight preference faders and eight wave feature knobs in fixed storage outside that allowance. Those capacities reserve float32 host-input conversion into double; they are host transport requirements outside the audio tensor ontology, not tier limits or mixed-precision DSP state. A smaller standalone caller can explicitly lower the capacities and corresponding budget to its actual input requirements. The float32 examples need no such conversion storage.

## Migration rules

Keep the old versioned IDs executable while consumers depend on their contracts. Select a new operation intentionally, rewrite its state/parameters/assets/bindings, and verify onset, feedback response, modulation behavior, reset and precision. Saving a new ID is an observable computational change, not a text substitution.

For the finite, non-overflowing `math.integer_delay` shift-register case, an explicit migration can preserve its length `N`, clock and feedback: allocate `N+2` ring state values; use parameters `[N, old_feedback]`; set both cursor limbs to zero and reverse the old newest-to-oldest history into the new ring buffer. At cursor zero, the ring's newest stored value lies at index `N-1`. Add the 3-entry sparse journal budget and one extra parameter coordinate. Check output onset and continuation before accepting the migrated sound. The new operation rejects invalid controls/overflow transactionally; this differs from old kernels that clamp or hold certain invalid calculations.

Do not map a complete old reverb family directly onto the FDN8 primitive: the old implementation includes additional absorption, modulation, glide and freeze behavior. Likewise, a previous FIR/convolution operation may have a different state order or readout latency. Preserve those distinctions with explicit composition/state conversion and verification. No automatic migration tool is claimed here.

## Streaming spectral operations

`SpectralProduction.h/.cpp` add two versioned operations. Both retain the existing tensor FFT's
full interleaved complex layout and forward sign; inverse FFT includes `1/N`, unlike the legacy
PFFFT inverse whose normalization was folded into the synthesis window. N is a declared
power of two, at least 2. An operation does not infer a musical spectrum, window or overlap.

- `spectral.stft_analysis.v1`: state `N+4` contains exact two-limb ring cursor, two-limb hop phase
  and N capture values. Parameter `hop` is an integer in `[1,N]`; asset 0 is the N-value analysis
  window. It captures one operand per scheduled update, emits the full `2*N` spectrum followed
  by a binary fresh-frame flag, and emits zeros between frames. Startup is explicitly
  zero-extended: the first frame arrives after one hop, with preceding history zero. Five sparse
  writes update capture/metadata. Output clearing is O(N), and each frame's FFT is O(N log N).
- `spectral.stft_synthesis.v1`: state `N+3` contains a two-limb cursor and an `N+1` overlap-add
  ring. It consumes the same spectrum/flag layout, with asset 0 the synthesis window and asset 1
  the N-value reciprocal COLA normalization. Each accepted frame adds its normalized inverse FFT
  beginning at the next update; G reads the old cursor. The maximum journal is `N+3` writes,
  and `scratch_elements=2*N` reserves typed inverse-FFT workspace separately from persistent S.

Both operations use the tier Scalar for coefficients, FFT arithmetic, capture, overlap-add and
temporary numeric workspace. The runtime reuses the maximum prepared workspace across callbacks;
callbacks initialize all workspace they read and cannot use it as persistent or shared state.
`sparse_state_writes` sums journal capacities; `scratch_elements` bounds this separate maximum.

With matching period-one clocks and committed-output-history transport, an authored COLA pair
reconstructs input delayed by `N+1` updates, including startup. At periods above one host sample,
analysis must Hold between updates so the frame survives intervening host ticks. Bin spacing is
`sampleRate/(periodSamples*N)`; delay units remain scheduled updates. Different clocks require
an explicit frame transport/resampling policy. Arbitrary windows, overlapping gains, irregular
frames or non-Hermitian edits do not automatically preserve a real unity reconstruction; synthesis
outputs the real part of the inverse transform.

This extracts streaming capture and weighted overlap-add ideas from legacy `SpectralStft`, with
explicit windows/hop and exposed state. It does not automatically port the old phase-vocoder,
identity phase locking, freeze/smear chain or optimized float-only PFFFT backend. Legacy packed
DC/Nyquist frames and normalized synthesis windows require explicit conversion before reuse.

## Explicit nonlinear oversampling

`NonlinearProduction.h/.cpp` add nine IDs: `nonlinear.{tanh,clip,table}.v1`, and the corresponding
`_polyphase2x.v1` / `_polyphase4x.v1` variants. These choose mathematical transfer functions and
processing factors explicitly rather than switching hidden quality state or selecting a synth
family. Basic parameters are drive, bias and output gain; table variants additionally declare
table_min/table_max and an editable uniformly spaced table asset with clamped endpoints.

1x operations are stateless. Each oversampling stage uses explicit dense halfband taps of even
length M in `[2,24]`. State contains three cursor/history rings per stage, totaling
`3+5*M/2` Scalar values; 4x adds a second independently declared stage. Halfband assets precede
the table asset. 2x reserves six journal writes; 4x reserves fifteen. F and G independently
evaluate the same old state and input, with bounded local temporary overlay values; only F's
validated writes persist. Histories and taps remain true float64 in Pro/Studio.

The shared legacy halfband design helper is an off-thread convenience, not an inferred sound
choice. For symmetric linear-phase FIR taps the declared delay is `M1-1` updates at 2x and
`M1-1+(M2-1)/2` at 4x. Arbitrary authored coefficients still execute but need not have that group
delay or useful alias rejection. Polynomial/table response, filter lengths and measured quality
remain per-operation decisions. These variants do not assert complete parity with every old
distortion mode, adaptive quality policy or arbitrary-rate resampler.

## Verification and measurement scope

Production tests exercise both scalar types, impulse onset, feedback echoes, ring wrapping, exact cursor progression past 65535, block segmentation, reset, fixed-delay allpass energy, a changing-delay independent Hermite reference, FDN tail decay, causal FIR taps, float64 coefficient distinctions and runtime budget/atomic rollback. Independent optimized Windows compilation with warnings as errors passed 10 cases and 9932 assertions before these documentation examples were added.

All three JSON examples were subsequently compiled by the built native runtime at 48 kHz and rendered for 128 impulse-input samples. The declared scalar widths were verified as 4/8/4 bytes; output remained finite, onsets were 16/3/0 samples respectively, ring echoes and FIR taps matched their contracts, and reset produced identical repeated renders. This fixture checks primitive outputs; host `outputMix` remains a separate explicitly declared readout.

The opt-in `.production-benchmark` fixture measures primitive callbacks through a small transactional harness, separately from the audio host. On one local optimized build, 192000 updates over a 384000-value ring took about 9.6 ms for each precision; state payload was 1536008 bytes for float32 and 3072016 bytes for float64, with 3 journal entries. These are observations from one machine, not guaranteed real-time host performance or a general speed comparison between tiers.

## Authored native host latency

An optional `host_latency_samples` integer declares native host compensation,
from zero through the host-supported signed integer maximum, additionally bounded by the
explicit `budget.host_alignment_bytes >= host_latency_samples * 2 * sizeof(float)`.
Absent means zero.
The runtime does not infer a longest graph path, add feedback delays, or count
artistic delays as latency. The composer preserves the first fragment's declaration
as an outer recipe property; subsequent additions and clones do not guess a new
value. The author may explicitly edit or remove that outer declaration. Invalid
declarations are rejected before any program or latency publication.

The host allocates a stereo dry ring off thread and delays the global dry contribution by the
declared latency. This applies at mix 0/50/100 percent, parameter bypass and the host's
`processBlockBypassed` entry. Compatible edits retain the ring cursor/history by swapping
ownership. Internal relationships between wet branches still require authored tensor alignment:
windowed spectral and linear-phase oversampling operations have contract-specific delays.

The five advanced production operations and their explicit layouts are documented in
[product completion](product-completion.md). Uniform partitioned convolution replaces
direct per-sample IR summation where authored; it does not claim the legacy nonuniform
zero-latency head or complete old-family parity.
