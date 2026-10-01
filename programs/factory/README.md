# Authored tensor programs

The installed factory library is produced by `FactoryPrograms.cpp`; the JSON files
in this directory are reviewable copies of those executable documents. These are
explicit producer-authored configurations, with no trained model requirement.

Every program declares its primitive activation, parameter and routing tensors,
persistent state, individual execution clocks, resource budget and output mix.
`audio_bindings` read native inputs or the previous output tensor. All primitive
updates commit together. The shared filter feedback in the instrument examples is
therefore recurrent computation, with no DAG execution requirement.

The polyphonic examples explicitly request eight stable MIDI rows through host-only
`performance` metadata. This capacity belongs to these authored programs. Their rows
contain frequency, velocity, gate, duration, pressure, normalized bend, CC74, onset,
channel and key. Note-off retains frequency and velocity while the declared release
interval runs. The configured pitch-bend range is two semitones.

Controls are described as X dimensions at each hierarchy level. Instrument examples
use six highest dimensions and ten intermediate dimensions; editable FX examples use
two and three. Each final matrix targets the declared DSP parameter base vector.
`parameter_metadata` gives each scalar its operation, instance, local index, units,
authored bounds and initial value. `driven` indicates an additional MIDI/state/audio
projection; its editable value is the base offset before that projection.

The factory documents expose exact properties of the reference algorithms:

- Additive frequency ratios and envelope timings are explicitly authored. Partial
  frequencies above Nyquist are not suppressed by the reference additive kernel.
- Moving Delay uses a short explicit buffer, valid across the tested 44.1–192 kHz
  host rates. Its delay memory and the diffusion network use reference shift registers.
- FIR Sculpt uses a short authored impulse response, with mean-stereo excitation.
- Short Diffuse Space uses four unequal, explicitly sized delay memories, with a
  native stereo dry contribution.
- Audio Motion Field combines note performance, previous primitive outputs and
  native audio-rate frequency influence in one tensor configuration.

Factory tests verify compilation and capacity accounting, simultaneous note spectra,
silence and release, actual FIR/stereo response, high-rate delay output, and every
corner of the authored highest-level control ranges.
