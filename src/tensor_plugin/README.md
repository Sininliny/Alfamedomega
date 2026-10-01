# Alfamedomega host assembly

This directory supplies the tensor host and editor library used by the canonical
`Alfamedomega` plugin in `src/plugin/CMakeLists.txt`. It does not build a second
product. The original Afmo plugin is available with `ALFA_BUILD_LEGACY_PLUGIN=ON`
for original projects. The current product keeps the tensor entry's Aftn identity.

The product combines recurrent tensor execution with shared legacy DSP and UI:
TPT SVF transitions, RBJ filter designs, cubic/Kaiser interpolation, theme and
look-and-feel, panel/knob gestures, waveform/spectrum/meter drawing and Hann FFT
analysis. The three legacy reuse audit documents describe implemented reuse and
remaining adaptations. Catalog coverage is not a claim of optimized legacy parity.

## Indexed controls

Controls are X[layer,index]. Expand arbitrary control layers, edit their many-to-many
matrices, or inspect all scalar DSP instance parameters and locks. Only bottom
instance parameters carry semantic names/units. Default matrices are authored;
there are no brightness/warmth macro definitions or fabricated learned ordering.
Eight stable X automation ports address the first eight highest coordinates.
Pagination and host port count do not limit internal tensor dimensions.

Settings selects one uniform execution type: Lite/Standard float32, Pro/Studio
float64. Raw parameter, matrix, asset and persistent state storage follow that
type. Native audio and highest host automation retain explicit float32 boundaries.
See `docs/tiers/runtime-precision.md` for measurements and allocation contracts.

Perform provides indexed knobs, explicit XY axes, complete-sound A/B snapshots,
waveform/spectrum, stereo meters and keyboard. Sound design exposes primitive
activations, all available instance parameters, control layers and relationships.
Tensor space provides the full editable contract, including state, clocks, assets,
audio operands and budgets. Settings retains the old themes, custom theme import,
accent and knob gestures. UI preferences persist separately from the sound.

## Audio and performance

Instrument, FX and hybrid modes use the same executable representation. They
choose available native input categories without replacing its DSP structure.
Input audio is mono/stereo. Each MIDI row has ten fields: Hz, velocity, gate,
elapsed seconds, pressure, normalized bend, CC74, onset seconds, channel and key.
A producer may request stable voice-indexed rows and an explicit voice capacity,
release retention and bend range; old compact-row reference documents still load.
Pedal holds gates; note-off preserves the row for its declared release interval.
Global panic clears DSP/performance state. Shared state has no inferred voice
ownership: channel all-sound-off releases that channel without resetting others.

Gesture axes describe bend, pressure and CC74 per MIDI channel. Context retains
time, sample rate and onset/release history. Indexed controls are not implicitly
copied into user preferences. Immutable `latentSound` is optional. A separately
imported runtime model predicts tensors; no factory program requires trained weights.
Training is paused while product work continues.

Each instance may sum `audio_bindings` from inputs, shared state or previous output
through routing/modulation weights. All updates read the same old state snapshot;
there is no DAG execution or fixed global control clock. `outputMix` explicitly
maps output elements to host channels.

## Host lifecycle

Off-thread compilation validates tensor shapes/masks, assets, allocation, clocks,
locks and explicit budgets. A prepared program publishes at the block boundary;
retired programs are destroyed off the callback. Compatible configuration edits
preserve state and notes and support undo/redo. Invalid edits retain the last valid
sound. Model context without a migration API is reset explicitly.

Input/output gain, mix, bypass and mode support host automation. Project state
contains complete sound JSON, automation values and a separate UI tree. Presets
contain complete editable sound documents, have stable identities/categories and
use atomic file publication and explicit overwrite handling. The MIDI staging
buffer is preallocated; oversized blocks are rejected rather than allocating.

## Local checks

Build `Alfamedomega_Standalone`, `Alfamedomega_VST3`, `alfa_tensor_host_tests` and
`alfa_tensor_tests`. `--tensor-smoke-test` initializes the actual editor/processor,
renders MIDI and native audio and verifies project restore without an audio device.
`ALFA_UI_SNAPSHOT`, `ALFA_UI_VIEW` and `ALFA_UI_SMALL` export real editor views.
The hidden `[.vst3-binary]` test loads the built VST3 in an independent JUCE host,
renders notes and creates its editor; set `ALFA_VST3_BINARY` to its bundle path.
