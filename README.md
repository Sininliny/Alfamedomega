# Alfamedomega

A universal audio tensor compiler: a trainable representation of how digital sounds can be
constructed, transformed and controlled. The intended learned compiler predicts primitive activation,
routing, parameter and modulation relationships, lowered into state allocation and schedules
for a deterministic recurrent DSP runtime.

The core abstraction is **Audio Tensor Space**. Instrument, FX and hybrid operation are native
requirements. This is not a traditional synthesizer, a preset generator or a modular DSP graph.

**Status: executable tensor compiler and native tensor plugin alongside the legacy entry.**
`src/tensor` supplies versioned tensor programs, recurrent deterministic execution, independent
clocks, padding masks, capacity validation, immutable assets, parameter locks and a migration
boundary. Fifty mathematical operations cover the required primitive families. The new
`AlfamedomegaTensor` Standalone/VST3 entry supports native audio, MIDI/MPE, instrument/FX/hybrid
execution, semantic intentions, inspection and expert tensor editing. A separately budgeted
recurrent neural compiler predicts tensor controls; DSP generates audio.

`training/aom` supplies dual-supervised WAV learning, native DSP rendering, runtime model
export, local preference learning and constrained tier candidate optimization. These are
executable reference implementations. The repository does **not** contain a universal pretrained
audio-construction model; the verified learning fixture is deliberately small. Structural
templates currently supply primitive identities and projection bindings for WAV import.

The [project direction](docs/audio-tensor-space/mission.md),
[runtime contract](docs/audio-tensor-space/runtime-contract.md),
[migration audit](docs/audio-tensor-space/migration.md) and
[training requirements](training/METHOD.md) describe the new direction and actual limits.
Older component documents describe the existing implementation and must not override it.
Compatibility adjustments are authorized; the legacy plugin entry remains during stabilization.

## Build

```bash
export CPM_SOURCE_CACHE=$HOME/.cache/CPM
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

Use `--preset core` to build the DSP, inference and tensor libraries with their tests, without
JUCE. See [CONTRIBUTING.md](CONTRIBUTING.md) for requirements, layout, conventions and the
git workflow.

## Layout

```
src/dsp/         existing DSP kernels and legacy runtime (no JUCE)
src/tensor/      recurrent tensor contract and deterministic runtime (no JUCE)
src/tensor_plugin/ native tensor Standalone/VST3 and layered editor
src/inference/   legacy model loading and real-time inference host (no JUCE)
src/ui/          look-and-feel and editor components
src/plugin/      the plugin target
training/        AOM data contract plus legacy method training and ONNX export
programs/        explicit demonstration tensor programs, not learned defaults
docs/            design docs and decision records
cmake/           build helpers and pinned dependencies
```

## Design documents

- [ADR 0001: plugin framework and inference stack](docs/decisions/0001-plugin-and-inference-stack.md)
- [Tensor tier requirements](docs/audio-tensor-space/mission.md#tiers-and-access)
- [Legacy model tiers](docs/tiers/model-tiers.md)
- [Packaging plan](docs/packaging/packaging-plan.md)
- [UI design](docs/ui/ui-design.md)

## Licensing

Alfamedomega's own code is under the Apache License 2.0 ([LICENSE](LICENSE)).

**JUCE licence not yet decided.** The plugin is currently built against JUCE under its
**AGPLv3** terms. Any plugin binary built today is therefore a combined work that can only be
distributed under AGPLv3, with its full source. Before any public release, either buy a JUCE
commercial licence (Starter, Indie or Pro) or accept AGPLv3 for the released plugin. The
`dsp`, `inference` and `tensor` libraries do not use JUCE and are unaffected.

The VST3 SDK comes with JUCE. The AAX SDK is licensed separately by Avid and is never
committed to this repository.
