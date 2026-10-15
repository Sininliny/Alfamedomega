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
  WAV rendering execute in `python -m training.amo.smoke build/amo-smoke`.
- The controlled canonical harmonic coefficient was recovered as 0.35335 against
  the original 0.35. Canonical loss fell from 0.05110 to 0.0000462; rendered-instance
  loss fell from 0.09097 to 0.00000224. These are fixture results, not evidence of
  arbitrary-WAV generalization.
- AMO Centre CLI invokes the new learner and compiler; the GUI uses an independent
  process for tensor training/import, leaving the audio host outside offline work.
- C++ formatting uses locally installed clang-format 22.1.3; CI's version 18 and
  other operating systems have not been run here.

## Reproduce

```powershell
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
& 'build/src/tensor_plugin/AlfamedomegaTensor_artefacts/Release/Standalone/Alfamedomega Tensor.exe' --tensor-smoke-test
python -m training.amo.smoke build/amo-smoke
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
- AMO Centre in its PySide6 environment: 54 passed, 6 platform/environment skips.
- Actual Standalone smoke process: exit code 0 after its processor/editor checks.
- Actual trained runtime GRU rendered 16,000 samples at 16 kHz. Its learned routing
  coefficient produced peak 0.159958 versus baseline 0.080000; reset replay was
  bitwise identical. A one-byte model allowance was refused. See
  `build/amo-smoke/runtime-report.json`.

CMake's `ALFA_AMO_TRAINING_PYTHON` selects an interpreter containing NumPy/PyTorch,
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
python -m training.amo.runtime_smoke build/amo-runtime-smoke
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
At that checkpoint, reference FFT/STFT, convolution, long delay and several physical operations still
needed the optimized legacy implementations adapted to the typed state contract;
the reuse audits distinguish these from the components already shared. Catalog
coverage of 52 operations at that checkpoint did not establish production CPU bounds for arbitrary
configurations or every requested historical audio technique.

## Standalone interaction regression, 2026-10-01

The preceding screenshot smoke did not exercise a sustained event loop. A user
reported an unusable Standalone. The new idle-editor regression reproduced five
unrequested preset reloads in 200 ms (revision 2 became 7). Programmatic ComboBox
clears queued asynchronous user-selection notifications; after rebuilding the
choices, the preset callback reloaded the sound and repeated the refresh forever.
All programmatic choice rebuilds now suppress these notifications. A user preset
selection still sends its notification and loads exactly once.

The host suite now passes 31 cases / 204,493 assertions, including sustained idle
dispatch and exactly-one preset publication. The final VST3 binary host check
again passes 5,127 assertions. The Standalone additionally supports
`--standalone-interaction-test`: it opens the actual native window and device,
runs the regular message loop and audio callbacks, verifies idle revision stability,
selects a preset, changes an X control, triggers keyboard audio, changes pages,
releases notes with Panic and selects Pro through the editor. This actual-device
test passed after the fix; `ALFA_INTERACTION_RESULT` records its result. It needs
a working output device and is separate from the device-independent smoke.

A full-duplex device initialization can also leave no device available. Standalone
now attempts output-only playback in that case and displays the fallback or actual
device error in the editor. Audio input remains configurable in the native Settings
dialog. An unavailable input must not silently prevent instrument playback.

## Production extraction contract update, 2026-10-01

The current registry contains 68 operations: four foundation kernels and 64 Universe operations
(48 reference, five production ring, two streaming STFT, nine nonlinear). Sparse transactional
updates preserve large ring history without full copying. Separate typed numerical workspace
is explicitly budgeted by `scratch_elements`; it has no persistent or inter-primitive meaning.
The [production contracts](production-composition.md) document latency, normalization, state
layout and migration differences. This source update is not a claim that every legacy optimized
algorithm or universal DSP coverage has been completed.

Independent Windows optimized warnings-as-errors checks passed the production ring fixture
(10 cases / 9932 assertions) and streaming STFT fixture (6 cases / 320981 assertions), including
both scalar types and actual recurrent runtime transport. Three standalone ring/FDN/FIR JSON
examples compiled/rendered/reset using the built native runtime. These focused checks are
distinct from the centrally rebuilt combined suite and real host stress testing; this documentation
update does not infer new central counts or nonlinear test results from source alone.

The editor now displays current configuration without an implicit-loading selector. Sound library
selection/search/inspection only changes the candidate preview; explicit Load publishes a sound.
Earlier selection-driven browser behavior above is historical. Training remains paused during
this product/production extraction work; authored default sounds contain no trained universal model.

The subsequent central host run passed 53 cases / 218688 assertions before the last added latency
cases; the independently loaded VST binary check passed 5127 assertions. The updated actual
Standalone interaction test also passed, including library browsing without sound mutation.
These results were reported by the coordinating build run; this docs-only pass did not rerun them
or infer results for later added tests. Sparse duplicate checking now uses bounded structural
Patricia-trie workspace, reported separately from Scalar numerical storage; no full-state scan
or clear is needed for large sparse histories.

## Final integrated Windows validation, 2026-10-01

The frozen source built Release targets `Alfamedomega_Standalone`,
`Alfamedomega_VST3`, `alfa_tensor_c`, `alfa_tensor_tests`,
`alfa_tensor_host_tests` and `alfa_tensor_sparse_bench` successfully.

- Combined tensor suite: 89 cases / 602424 assertions.
- Combined native host suite: 55 cases / 218745 assertions, including authored
  host latency transactions, composition latency declarations and reopened
  nested editor dialogs.
- Final VST3 loaded in the independent JUCE plugin host: 5127 assertions.
- Shared legacy delay/nonlinear suite: 76 cases / 20532228 assertions.
- Native C ABI: Standard float32 and Pro float64 render/reset/capacity checks;
  insufficient boundary budget rejected.
- Minimum-size editor and sound-library snapshots checked for layout; library
  inspection never publishes a sound, only explicit Load does.

The sparse journal benchmark with bounded duplicate validation measured roughly
94–107 ns per update for histories of 128, 65536 and 384000 values on this machine.
This isolated fixture does not establish CPU deadlines for arbitrary compositions.
Training, every DAW/platform qualification, optimized partitioned convolution,
large asset deduplication and full DSP universe completion remain separate work.

The final rebuilt Standalone actual-device interaction also passed again, including
idle stability, nonmutating library/instance inspection, explicit loading, keyboard
audio, indexed X controls, Panic and Pro tier selection.

## Priority completion validation — 2026-10-01

The [completion contracts](product-completion.md) supersede the prior outstanding statements
about uniform partitioned convolution, ratio-aware playback, mixed host alignment, numeric
composition reconciliation, scoped compilers, compatible state publication and native tier
capacity conversion. The original [design audit](design-compliance-audit-2026-10-01.md) is
retained as a before-change record.

Local Windows Release, warnings-as-errors:

- Core tensor suite: **95 cases / 720904 assertions**, passed.
- Native host suite: **59 cases / 222932 assertions**, passed.
- Rebuilt product VST3 in the independent JUCE host: **1 case / 5127 assertions**, passed.
- Rebuilt Standalone actual-device interaction: passed idle stability, nonmutating instance
  and library inspection, explicit load, keyboard/X audio, Panic single prepared reset and
  single Pro tier publication.
- Product Standalone/VST3 and tensor C ABI library built successfully.
- Full first-party clang-format 18 check passed after correcting historical formatting in
  twelve files. These additional changes are formatting only.

PR checks additionally exposed strict GCC/Clang signed conversion and complete-type issues,
two pre-existing STFT test index conversions, and the pre-existing UI use of floating-point
`to_chars` above the macOS 11 deployment target. These were repaired explicitly; UI formatting
now uses the portable shortest-round-trip JSON formatter. The complete core source/test set
was also checked with the local Clang frontend. CI results are authoritative for platform builds.

New signal fixtures independently compare multi-partition convolution with direct summation,
verify playback alias suppression/passband, modal long decay, waveguide tuning with damping
and dispersion, and frame phase-locking/flags. Migration verifies buffer ownership, clock,
identity, initial state and active allocation. Host fixtures cover aligned dry/wet mix at
0/50/100 percent, parameter/DAW bypass across blocks, scoped model composition, preserved
matrix relationships, resource refusal, locked-candidate rejection, measured bad pitch and
equivalent-recurrence tier reduction. Held MIDI survives a precision change.
An additional waveguide boundary regression reproduced an undamped alternating residue when
a fractional allpass reached exact integer delay. Explicit bypass/memory replacement removes
that residue; it is verified with zero excitation and zero feedback.
Inverting feedback also compensates its half-cycle phase so the requested fundamental
does not drop an octave; an independent impulse-onset fixture verifies the half-cycle loop.

At the user's request, GitHub CI was stopped due to quota limits. Final acceptance and merge
use the local builds, tests, Clang frontend checks and actual-device/VST3 host evidence above.
Subsequent push/merge commits explicitly skip CI. Remote macOS/Linux completion is not claimed.

This pass did not start training. Perceptual listening, all DAWs/platforms, large-program
deadline/allocation stress and complete legacy algorithm parity remain separate qualification.
Tier numeric capacities are authored inputs, not product default promises.


## Core readiness follow-up — 2026-10-01

The user will perform FL Studio project acceptance. This pass prioritizes deterministic DSP,
free tensor composition, complete loading validation and layered control authority. No GitHub
CI was used, and training remained paused.

Final local Windows Release results:

- Core tensor DSP: **95 cases / 720904 assertions**, passed.
- Native host: **68 cases / 307457 assertions**, passed.
- Final rebuilt VST3 in the independent JUCE host: **1 case / 5127 assertions**, passed.
- Final rebuilt Standalone actual-device interaction: passed idle, nonmutating inspection,
  explicit load, keyboard/audio, X controls, Panic and Pro tier.
- Standalone/VST3 and native tensor C ABI built successfully.
- Full first-party clang-format 18 and whitespace checks passed.
- Local Clang frontend checks passed for execution, Tier evaluation and the three changed
  model/loading/allocation test files under strict warnings. JUCE's legacy Windows template
  constructor requires delayed template parsing for this standalone frontend invocation;
  its C++20 deprecation warning was disabled. This is not a remote platform build claim.

New numerical fixtures compare model-driven execution with an independent no-model reference
through three control layers, saturation, locked bottom parameters, unequal-depth composition,
float32/float64 and model clocks across block boundaries. Reset/replay is compared with a
fresh execution before and after a delayed first model event. Loading rejects malformed
scoped containers/IDs/offsets, unavailable model input widths, latent over-capacity, negative
or fractional host allowances, invalid output triples and malformed native performance policies.
Tier fixtures cover right-only pitched output and explicitly unpitched stochastic FX, including
rejection of a changed candidate and an undeclared missing pitch fixture.

Thread-local replacement C++ new/delete operators (normal/array/aligned) surround **288 complete
native callbacks** without a warm-up exemption: 192 factory callbacks across all eight sounds
and both precisions, plus 96 callbacks in two independent hosts with 384000-sample ring
histories, CPU models, layered controls, live edits/locks/reset and 32/64/128-sample blocks.
All observed C++ allocation and deallocation counts were zero; output remained finite and
host failure counters remained zero. This probe does not intercept direct C malloc/free,
allocations inside external DLLs or operating-system allocation APIs. It is bounded test
coverage, not long-duration or arbitrary-configuration CPU deadline certification.

Asset deduplication/compression, preference-profile input, additional relationship visualization,
silence evaluation, full historical algorithm parity and cross-platform/DAW qualification remain
separate work. FL Studio acceptance is assigned to the user, not claimed from these host tests.
