# Local validation, 2026-09-30

Synchronized GitHub base: `a7e7b71`. Branch: `codex/audio-tensor-refactor`.

The new implementation is independent of the legacy method/slot engine. Release
builds produce the tensor C ABI, tensor tests, native host tests, Standalone and
VST3, while preserving the existing targets.

## Executed checks

- Tensor runtime, compiler, recurrent neural inference and primitive signal tests.
  Native GRU outputs are compared against sequential PyTorch golden vectors.
- Native host tests cover audio input, simultaneous MIDI control, MPE pressure,
  note-off, persisted tensor programs and runtime-model import/refusal.
- The actual Standalone executable's `--tensor-smoke-test` initializes its processor
  and editor, renders MIDI instrument and native audio FX, restores saved state,
  and returns zero only when all checks pass. This does not require an audio device.
- A native DSP rendered dataset trains both canonical and perturbed-instance heads.
  Save/load, WAV inference, bounded locked tensor export, native recompilation and
  WAV rendering execute in `python -m training.aom.smoke build/aom-smoke`.
- The controlled canonical harmonic coefficient was recovered as 0.35335 against
  the original 0.35. Canonical loss fell from 0.05110 to 0.0000462; rendered-instance
  loss fell from 0.09097 to 0.00000224. These are fixture results, not evidence of
  arbitrary-WAV generalization.
- AOM Centre CLI invokes the new learner and compiler; the GUI uses an independent
  process for tensor training/import, leaving the audio host outside offline work.
- C++ formatting uses locally installed clang-format 22.1.3; CI's version 18 and
  other operating systems have not been run here.

## Reproduce

```powershell
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
& 'build/src/tensor_plugin/AlfamedomegaTensor_artefacts/Release/Standalone/Alfamedomega Tensor.exe' --tensor-smoke-test
python -m training.aom.smoke build/aom-smoke
.venv/Scripts/python.exe -m pytest apps/control-center/tests -q -p no:cacheprovider --basetemp build/cc-tests-fresh
```

Use a fresh test directory each time: preexisting sandbox-owned temporary
folders can cause unrelated permission failures.

## Scope and remaining product decisions

The executable representation, stateful DSP, model categories and hosting paths
are implemented. Fifty operations cover the requested families; parameter/asset
contracts are explicit. FFT/vocoder operations consume complex frames; windowing
and overlap-add must be supplied by an appropriate producer. Assets are immutable
per prepared configuration. New structures and masks are compiled off the audio
thread; in-place realtime tensor publication preserves the prepared topology.

The repository does not include a universal pretrained construction model,
validated arbitrary-recording normalization, a perceptual dataset or quality
certification. WAV reference import uses explicit structural templates. Tier
candidate optimization and preference/importance learning require supplied
measurements and constraints. Product tier capacities remain explicit inputs;
legacy resource totals are not silently adopted. Legacy data remains available
in its original entry; the byte-retaining migration boundary is not an automatic
semantic converter for every legacy preset/model/automation format.

No DAW playback, listening judgment, Linux/macOS result or universal latency
certification is inferred from the local build and deterministic checks.

## Final local results

- Complete Release build: succeeded, including both new and retained plugin targets.
- Tensor suite: 47 cases, 42,291 assertions passed.
- Native tensor host: 6 cases, 27 assertions passed.
- Combined tensor/training/retained-plugin CTest selection: 72/72 passed.
- AOM Centre in its PySide6 environment: 54 passed, 6 platform/environment skips.
- Actual Standalone smoke process: exit code 0 after its processor/editor checks.
- Actual trained runtime GRU rendered 16,000 samples at 16 kHz. Its learned routing
  coefficient produced peak 0.159958 versus baseline 0.080000; reset replay was
  bitwise identical. A one-byte model allowance was refused. See
  `build/aom-smoke/runtime-report.json`.

CMake's `ALFA_AOM_TRAINING_PYTHON` selects an interpreter containing NumPy/PyTorch,
independently of the GUI environment. For this host it is
`C:/Users/Sinin/miniconda3/python.exe`. This is a local cache choice, not a portable
repository default.

The broad retained-suite run passed all other registrations but was interrupted
in the old offline latency test after 194 seconds. Follow-up diagnosis showed
its parameter combinations were progressing: Windows' roughly 15.6 ms sleeps
amplify its per-frame scheduler polling. The tensor runtime does not use that
legacy scheduler. The complete follow-up result is recorded separately below
when available.

Follow-up retained latency check completed successfully: 380 assertions across
all 48 generated parameter combinations, exit code 0. The individual manifest
path-refusal check also passed. No legacy source changes were needed. The initial
interrupted run is retained above as diagnostic history; no unresolved regression
is inferred from its interruption.

The learned runtime smoke is reproducible from a clean checkout after native build:

```powershell
python -m training.aom.runtime_smoke build/aom-runtime-smoke
```

It generates explicit test targets, trains/exports a model, renders native A/B
audio, proves bitwise reset replay and verifies budget refusal.

After explicit latent-input and compiler-extension checks, the final tensor suite
passed 48 cases / 42,297 assertions and the native host passed 7 cases / 37 assertions.
The final combined CTest selection contains 75 registrations: tensor, host,
dependency-free contracts/hardware, PyTorch learners and retained plugin checks.
The native Standalone smoke was rerun against the final binary and returned 0.

The CI workflow now has a dedicated CPU tensor-learning job and retains the new
Standalone/VST3 artifacts. Remote CI has not run; local success does not claim
cross-platform or remote validation.

## Unified product and scalar precision, 2026-10-01

The canonical `Alfamedomega` assembly now combines the tensor host with reused
legacy DSP math and UI components. Current generated products are
`build/src/plugin/Alfamedomega_artefacts/Release/Standalone/Alfamedomega.exe` and
`build/src/plugin/Alfamedomega_artefacts/Release/VST3/Alfamedomega.vst3`.
Earlier `AlfamedomegaTensor` build outputs are stale prototype artifacts.

Local Release builds succeeded for the canonical Standalone, VST3, native C
runtime, tensor tests, host tests, retained plugin tests and precision benchmark.
The current tensor suite passed 59 cases / 142,771 assertions. The host suite
passed 30 cases / 204,488 assertions, including actual note execution in float64,
same-coordinate tier transfer, editor tier selection, double parameter preservation,
invalid-edit transactional refusal, preset/project restore, MIDI/pedal/release,
phase-preserving edits, explicit many-to-many layers, themes and functional bypass.
The retained plugin suite passed 18 cases / 482 assertions. Earlier shared filter
regressions passed 56 cases / 19,830 assertions.

The final VST3 bundle was scanned and loaded by an independent JUCE plugin host,
rendered MIDI audio, saved/restored state and constructed its editor: 5,127
assertions passed. The actual Standalone's instrument, FX and state round-trip
smoke passed for all four editor pages at 1040 x 700. PNG snapshots were inspected
for layout, visible waveform/spectrum and meters, indexed controls, lowest-level
parameter names, editable matrix/JSON and the precision selector. Numeric display
uses shortest exact round-trip strings instead of unnecessary binary decimal tails.

Native C interface checks rendered finite nonzero float32 and float64 audio,
verified bitwise reset replay, refused over-capacity inputs and refused inadequate
conversion-memory allocation. No neural training was run during this product pass.
Precision/resource measurements and their scope are in
[runtime precision](../tiers/runtime-precision.md).

These checks establish local Windows operation, not validation in every DAW or
on every platform. Authored factory configurations contain no trained model.
Reference FFT/STFT, convolution, long delay and several physical operations still
need the optimized legacy implementations adapted to the typed state contract;
the reuse audits distinguish these from the components already shared. Catalog
coverage of 52 operations does not establish production CPU bounds for arbitrary
configurations or every requested historical audio technique.
