# Alfamedomega

A neural synthesizer plugin (VST3, AAX, AU, Standalone) spanning subtractive, additive, FM,
wavetable, granular, physical-modelling, sampling, spectral and hybrid synthesis, with
several model-size tiers that trade fidelity for CPU.

Status: early scaffolding. Nothing is playable yet.

## Build

```bash
export CPM_SOURCE_CACHE=$HOME/.cache/CPM
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

Use `--preset core` to build only the engines and inference host with their tests, without
JUCE. See [CONTRIBUTING.md](CONTRIBUTING.md) for requirements, layout, conventions and the
git workflow.

## Layout

```
src/dsp/         synthesis engines (no JUCE)
src/inference/   model loading and real-time inference host (no JUCE)
src/ui/          look-and-feel and editor components
src/plugin/      the plugin target
training/        model training and ONNX export
docs/            design docs and decision records
cmake/           build helpers and pinned dependencies
```

## Design documents

- [ADR 0001: plugin framework and inference stack](docs/decisions/0001-plugin-and-inference-stack.md)
- [Model tiers](docs/tiers/model-tiers.md)
- [Packaging plan](docs/packaging/packaging-plan.md)
- [UI design](docs/ui/ui-design.md)

## Licensing

Alfamedomega's own code is under the Apache License 2.0 ([LICENSE](LICENSE)).

**JUCE licence not yet decided.** The plugin is currently built against JUCE under its
**AGPLv3** terms. Any plugin binary built today is therefore a combined work that can only be
distributed under AGPLv3, with its full source. Before any public release, either buy a JUCE
commercial licence (Starter, Indie or Pro) or accept AGPLv3 for the released plugin. The
`dsp` and `inference` libraries do not use JUCE and are unaffected.

The VST3 SDK comes with JUCE. The AAX SDK is licensed separately by Avid and is never
committed to this repository.
