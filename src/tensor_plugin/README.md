# Native tensor host

`AlfamedomegaTensor` supplies a separate JUCE standalone/VST3 host for executable AOM tensor
configurations. It accepts live input audio, MIDI and simultaneous hybrid performance without
calling the old synthesis-method backend. The embedded MIDI projection reference loads at startup, with an explicit demonstration loader
for audio recurrence and hybrid interaction; changing input mode preserves the current program.
Configuration JSON is compiled off the audio callback;
stateful sample execution and explicit outputMix triples render the compiled program.

The editor offers the same musical intention controls, inspection and expert JSON editing
without tier-specific feature locks. Intention values are input information: only explicit
configuration projections assign their acoustic behavior. No learned model is claimed by the
reference examples. JSON parameter_locks protect compiled numeric parameter authority.

## Host feature boundary

MIDI input is a compact active-note tensor with axes [note, feature], variable note count and
10 host-defined features: pitch Hz, velocity, gate, elapsed note seconds, pressure, normalized
channel bend, CC74 timbre, onset seconds, channel number, MIDI key number. These input feature
encodings do not define the tensor universe or primitive parameter dimensions. Capacity is
128 simultaneous active notes in this host boundary, not a tier or ontology restriction.
At note-off the row is removed and context retains the latest release key/time. MPE channels,
poly pressure, bend and CC74 can influence multiple targets only through explicit tensor bindings.

Gesture axes are [channel, feature] with normalized bend, pressure and CC74 for 16 MIDI channels.
Context features are host time seconds, sample rate, most recent onset key, cumulative onsets,
inter-onset seconds, most recent onset time, most recent release key and release time. Intention
features are brightness, warmth, movement, aggression and organic character. Live audio is two
scalar channels; mono supplies zero for the second input. Empty input categories are explicit.
An optional top-level `latentSound` tensor carries immutable axes, shape, values and presence
information. It is validated off the audio thread, borrowed by callbacks and preserved with the
complete raw JSON. Absence remains empty; the host does not infer a latent default. Extended
MIDI/phrase encodings remain follow-up work.

Instrument mode omits audio input; FX omits MIDI/gesture; hybrid exposes both. The mode switch
does not select a fixed synthesis method or rewrite routing. `outputMix` is a required top-level
array of [outputElement, hostChannel, gain] triples; the host does not invent implicit output sums.

## Publication and persistence

Each successful load prepares a candidate program and atomically publishes it at the next audio
block. Old programs enter a bounded retirement queue and are reclaimed off the audio callback.
If the retirement queue fills, publication waits while the current program continues. Configuration
parsing, state allocation, JSON/UI persistence and destruction never occur in the audio callback.
State updates, MIDI events and meter counters do. A 64 KiB MIDI staging buffer is allocated
during preparation; oversized (>60000 byte) input blocks are refused and silenced. Unsupported
messages longer than three bytes are skipped before constructing owned MIDI message objects. Failed imports retain the previous program.

Project state retains complete input JSON, mode and intention controls. Open/save JSON dialogs
make the configuration editable and portable; changing protected parameters by explicit recompilation
is distinct from predicted value publication. Inspectable allocations, clocks, activations, routing,
locks, failures and measured block duration describe the actual host/runtime operation.

Tests render exact FX recurrence with native input, joint MIDI/audio hybrid interaction and project
persistence with refusal retention. This host is not evidence of trained compilation, universal WAV
inference, validated nuisance normalization or completed learned tier conversion.


## Runtime compiler model import

The editor can import `aom.runtime.gru/1` weights into the complete configuration JSON under
`runtimeModel`. Off-thread preparation validates the model against the tensor coordinates and
its own explicitly declared dimensions, bounds, masks, clock and RAM budget. `TensorModel::apply`
runs at the model's file-defined clock before DSP ticking and publishes predicted tensor values
without allocation. Absent required input or a refused structural prediction retains previous
DSP controls while audio continues; failures are visible in inspection. No universal input width,
control frequency or neural parameter budget is inferred from the host demonstration.

Raw runtime-model JSON and original host tensor configuration persist together. Embedded references
load no neural model. A host integration test loads the PyTorch golden GRU fixture and verifies
successful prediction publication and refusal retention for an incompatible RAM budget. This is
real neural inference infrastructure; golden fixture weights are not evidence of a fully trained
universal sound-construction model.

The standalone executable also accepts `--tensor-smoke-test`. This initializes its actual
processor and editor, renders embedded MIDI and native-audio examples, checks project-state
round-trip and returns success/failure without opening an audio device. Normal launch uses
JUCE's audio/MIDI device options and the tensor editor; the first view is musical intentions,
with detailed inspection and tensor JSON available explicitly.
