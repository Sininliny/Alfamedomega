# Foundation validation, 2026-09-30

Base: GitHub `main` at `a7e7b71`. Working branch: `codex/audio-tensor-refactor`.
This evidence applies to the independent recurrent CPU foundation, not completion of the
universal primitive universe, trained inference or new plugin hosting.

## Completed local checks

- Release builds of `alfa_tensor_tests` and retained `alfa_plugin_tests` using MSVC 2026,
  with the configured warnings-as-errors policy.
- 21 tensor-runtime/kernel CTest cases: recurrence, simultaneous state reads, independent
  clocks, explicit affine routing, parameter locks, masks/capacity, atomic refusal,
  no allocation during ticking/tensor publication, migration byte retention and numerical
  oscillator/filter/feedback/delay behavior.
- 9 dependency-free Python training-contract tests, also registered as one CTest:
  canonical/instance dual targets, grouping, shared padding/masks and independent RAM/VRAM
  estimates.
- 18 retained plugin CTest cases, including project-state restoration, preset/lock behavior,
  neural/stub note rendering and editor opening.
- All 40 selected CTest registrations passed. The Python registration runs the 9 cases above.
- New C++ files were formatted and checked with the locally available clang-format 22.1.3;
  whitespace checks passed. CI uses clang-format 18 and has not run on these changes.

Reproduce in this existing Windows build:

```powershell
cmake --build build --config Release --target alfa_tensor_tests alfa_plugin_tests --parallel 4
ctest --test-dir build -C Release -R '^(alfa_tensor_tests:|alfa_aom_training_contracts$|alfa_plugin_tests:)' --output-on-failure
```

The new tests are also included by the normal core/dev CMake configuration. No Linux,
sanitizer, DAW-host timing or measured CPU-budget result is claimed by this local check.

## Remaining decisions for the next implementation phase

- Whether the new tensor tiers retain the old model/runtime resource totals or use newly
  defined budgets. The new runtime accepts explicit capacities and does not import legacy
  method-allocation percentages.
- A normalization policy that distinguishes recording/pitch/loudness nuisance variation
  from timbre, articulation and temporal identity. Current normalization records validate
  declarations; they do not implement or establish perceptual invariance.

The [migration audit](migration.md) tracks the remaining implementation scope, including
semantic translators, learned compilation/compression, full primitive coverage and native
instrument/FX/hybrid host integration.
