# src/plugin and src/ui

Owner: the "Build the plugin shell and UI" thread. Design: [docs/ui/ui-design.md](../../docs/ui/ui-design.md).

## Targets

| Target | What it holds |
|---|---|
| `Alfa::pluginCore` | `ParameterLayout` (every parameter's ID, range, unit and group, plus what each of the 32 knobs means for each method), `PresetLibrary`, `Telemetry`, `PreferenceLog`. The one place a parameter is added. |
| `Alfa::ui` | Theme tokens, look-and-feel, controls, visualisations and the four views (Perform, Signal flow, Sound design, Customize). Depends on `Alfa::pluginCore`, not on the processor. |
| `Alfa::pluginShell` | Processor, editor and the synth backends (neural and stub). Linked by the plugin and by the tests. |
| `Alfamedomega` | The JUCE plugin: VST3 and Standalone everywhere, AU on macOS, AAX when `ALFA_AAX_SDK_PATH` is set. |

All four are INTERFACE libraries so JUCE modules compile once per final binary.

## The sound

A sound has the shape the model was trained on (training/METHOD.md, src/inference/MODEL_CONTRACT.md): `method` (nine choices, in the contract's order), `hybrid_a` and `hybrid_b` (used when the method is hybrid), and `knob_01` to `knob_32` in 0..1. What a knob means depends on the method; `params::knobInfo` holds the names, display curves and minimum model sizes taken from the teacher programs in training/alfamedomega_train.py, and the UI relabels knobs when the method changes. A hybrid runs component A on the knobs and component B on the knobs rotated by 11; knob 32 crossfades. Additive with physical, or two of the same method, is silent.

Layout version 2 replaced version 1's Method blend, macros and per-engine controls, which the model cannot drive.

## Preference learning log

With "Learn from how I play" on (Customize view; stored in `<control-center home>/preference/plugin.json`, shared by every instance and the control center), each instance appends events to `<home>/preference/events/<date>.jsonl` in the schema of preference/alfapref/events.py: `load` (source browser, user, project or init), `play` once a second while notes sound, `edit` once the sound settles, `project_save` at most once per sound and only after 5 minutes, `rate` from the like and dislike buttons, `close`. The audio thread only counts notes in atomics; a shared background thread writes the files. With learning off nothing is written. Instance ids are random and saved with the project; no host exposes a project id, so the instance id also stands in for the project. There is no user preset saving yet, so `save_preset` is never written.

## How the pieces talk

- Parameters live in one `AudioProcessorValueTreeState`; every control binds through an attachment.
- The audio thread writes `Telemetry` (lock-free atomics); `ui::LiveData` reads it at 30 Hz. The UI never waits on audio or inference.
- Interface settings (`ui::UiState`) are saved with the instance and optionally as a per-user default, never in presets. Loading a preset skips the model size and any locked parameter.
- Model-size changes go to the backend at once; the reported latency changes only while the transport is stopped.

## The synth backend

`SynthBackend` is the seam to `Alfa::dsp` and `Alfa::inference`. `createBackend` reads the models folder (`ModelLibrary::defaultModelsDir()`: `%ProgramData%\Alfamedomega\Models` on Windows, `/Library/Application Support/Alfamedomega/Models` on macOS, `~/.local/share/Alfamedomega/Models` on Linux; `ALFA_MODELS_DIR` overrides it).

- **Models installed:** `RealBackend.cpp`. The inference host turns notes and the patch into control frames and `alfa::dsp::Synth` renders them. The plugin reports the tier's latency and hands each note to the DSP that many samples later, when the first frames computed from it arrive. The network has 16 voice slots; each slot is mapped to the DSP voice that took its note. The patch is sent to the model (20 Hz check, message thread) whenever the method, hybrid pair or a knob changes; an untrained hybrid pair is silent. Changing Model size loads the tier on the message thread and re-prepares the DSP with processing suspended. Bounces run the model non-real-time, so no frame is dropped.
- **No models:** `StubBackend.cpp` plays one cheap waveform per method, so the shell and visualisations still work. The Signal flow view's model step says "(stub)".

To try the neural path before trained models exist, point `ALFA_MODELS_DIR` at the inference tests' dummy models (`<build>/src/inference/tests/models`, exported at build time when Python has numpy and onnx). Dummy weights make sound, not music.

Not wired yet: glide, a sampler source for the sampling method (waiting on the sampling scope decision), reference-audio conditioning and a measured per-frame inference time.

## Checking the UI

```
cmake --preset dev && cmake --build build/dev
xvfb-run -a -s "-screen 0 2560x1600x24" ctest --test-dir build/dev    # Linux; needs a display
xvfb-run -a -s "-screen 0 2560x1600x24" build/dev/src/plugin/tests/alfa_ui_snapshot shots paper
```

`alfa_ui_snapshot` plays a chord through the processor and writes a PNG of every view.
