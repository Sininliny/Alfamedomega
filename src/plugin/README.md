# src/plugin and src/ui

Owner: the "Build the plugin shell and UI" thread. Design: [docs/ui/ui-design.md](../../docs/ui/ui-design.md).

## Targets

| Target | What it holds |
|---|---|
| `Alfa::pluginCore` | `ParameterLayout` (every parameter's ID, range, unit, group and minimum model size), `PresetLibrary`, `Telemetry`. The one place a parameter is added. |
| `Alfa::ui` | Theme tokens, look-and-feel, controls, visualisations and the four views (Perform, Signal flow, Sound design, Customize). Depends on `Alfa::pluginCore`, not on the processor. |
| `Alfa::pluginShell` | Processor, editor and the stub synth backend. Linked by the plugin and by the tests. |
| `Alfamedomega` | The JUCE plugin: VST3 and Standalone everywhere, AU on macOS, AAX when `ALFA_AAX_SDK_PATH` is set. |

All four are INTERFACE libraries so JUCE modules compile once per final binary.

## How the pieces talk

- Parameters live in one `AudioProcessorValueTreeState`; every control binds through an attachment.
- The audio thread writes `Telemetry` (lock-free atomics); `ui::LiveData` reads it at 30 Hz. The UI never waits on audio or inference.
- Interface settings (`ui::UiState`) are saved with the instance and optionally as a per-user default, never in presets. Loading a preset skips the model size and any locked parameter.
- Model-size changes go to the backend at once; the reported latency changes only while the transport is stopped.

## Stubs

`SynthBackend` is the seam to `Alfa::dsp` and `Alfa::inference`. Until they land, `StubBackend.cpp` plays one cheap waveform per synthesis method so the shell makes sound and the visualisations are live. Replace `createStubBackend` with a backend that drives the real engines; the processor and UI don't change.

## Checking the UI

```
cmake --preset dev && cmake --build build/dev
xvfb-run -a -s "-screen 0 2560x1600x24" ctest --test-dir build/dev    # Linux; needs a display
xvfb-run -a -s "-screen 0 2560x1600x24" build/dev/src/plugin/tests/alfa_ui_snapshot shots paper
```

`alfa_ui_snapshot` plays a chord through the processor and writes a PNG of every view.
