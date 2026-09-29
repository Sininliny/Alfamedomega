# Contributing to Alfamedomega

## Layout and ownership

| Path | Target | Owner | Notes |
|---|---|---|---|
| `src/dsp/` | `Alfa::dsp` | DSP engines | Real-time synthesis engines. Plain C++20, **no JUCE**. |
| `src/inference/` | `Alfa::inference` | Inference host | Model loading, anira + ONNX Runtime. **No JUCE**. |
| `src/ui/` | `Alfa::ui` | Plugin shell and UI | Look-and-feel, theming, components. INTERFACE library compiled into the plugin. |
| `src/plugin/` | `Alfamedomega` | Plugin shell and UI | `juce_add_plugin`, processor, editor, parameters. |
| `training/` | – | Training | PyTorch training and ONNX export. |
| `docs/` | – | Anyone | Design docs and ADRs (`docs/decisions/NNNN-title.md`). |
| root, `cmake/`, `.github/`, `scripts/` | – | Repo and workflow | Build system, CI, conventions. |

Only a directory's owner edits it. If you need something from another component, ask its owner
(or open an issue) rather than editing their files. Component-specific dependencies are added in
that component's own `CMakeLists.txt` with `CPMAddPackage` and a pinned version; shared ones
live in `cmake/Dependencies.cmake`.

Dependency direction: `plugin → ui, dsp, inference`. `dsp` and `inference` do not link
against each other or JUCE, so they build and test in seconds and can be reused outside the
plugin. The one shared piece is the control-frame contract, `alfa/dsp/ControlFrame.h`, a
header owned by `dsp` that `inference` uses through `Alfa::dsp`'s include path only. Change it
only through the DSP owner, since both components and the training export depend on it.
The glue between the model's control frames and the engines lives in `src/plugin`.

## Building

Requirements: CMake 3.25+, Ninja, a C++20 compiler (Clang 15+, GCC 12+, MSVC 2022),
clang-format 18. The inference tests export small ONNX models at build time and need
Python 3 with `pip install numpy onnx`; without it those tests report as skipped. With Clang 20
the RealtimeSanitizer test target (`alfa_inference_rtsan_tests`) is built as well. On Linux the plugin also needs the JUCE dependencies
(`libasound2-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxcomposite-dev libfreetype-dev libfontconfig1-dev libgl-dev`).

```bash
export CPM_SOURCE_CACHE=$HOME/.cache/CPM   # share downloads between build trees

cmake --preset core      # dsp + inference + tests, no JUCE (fastest)
cmake --build --preset core
ctest --preset core

cmake --preset dev       # everything, including the plugin (downloads JUCE)
cmake --build --preset dev
```

Presets: `core`, `dev`, `asan` (core with AddressSanitizer + UBSan), `release`.
Build trees go in `build/<preset>/`, which git ignores. When several people or agents work in
the same checkout at once, give each its own tree outside it:
`cmake --preset core -B /tmp/alfa-dsp`.

AAX is built only when `-DALFA_AAX_SDK_PATH=<path>` is set; the SDK is not redistributable and
never goes in git.

## Tests

Each component has a `tests/` directory. Register a test executable with:

```cmake
alfa_add_test(alfa_dsp_tests
    SOURCES OscillatorTests.cpp FilterTests.cpp
    LIBRARIES Alfa::dsp)
```

Catch2 v3. Every `TEST_CASE` is registered with CTest. Tag slow or benchmark tests with
`[.slow]` so they don't run by default. DSP code should be tested against reference values
(known frequencies, impulse responses, energy), not only "it doesn't crash".

## Code style

- C++20. Formatting is `clang-format` 18 with the repo's `.clang-format`; run
  `bash scripts/format.sh` before committing, CI runs `--check`.
- Namespaces `alfa::<component>`; types `PascalCase`, functions and variables `camelCase`,
  constants `kPascalCase`, files named after their main type.
- Public headers under `include/alfa/<component>/`, sources under `src/`.
- Warnings are errors in CI (`-DALFA_WARNINGS_AS_ERRORS=ON`); build with it locally too.
- **Real-time rules** for anything reachable from `processBlock` or a DSP `process()`:
  no allocation, no locks, no system calls, no exceptions, no logging. Allocate in
  `prepare()`. Communicate with other threads through lock-free queues or atomics.

## Version control

### Commits

[Conventional Commits](https://www.conventionalcommits.org/), with the component as scope:

```
feat(dsp): add band-limited wavetable oscillator
fix(inference): avoid allocation when switching tier
build: pin anira to v1.x
docs(adr): record frame-rate decision
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`.
Scopes: `dsp`, `inference`, `ui`, `plugin`, `training`, `adr`, or none for repo-wide changes.
Keep commits small and building.

### Branches

`main` is always green. Work happens on one branch per component or feature,
merged through a pull request once CI passes:

```
dsp/wavetable-oscillator
inference/anira-session
plugin/parameter-layout
training/export-check
```

The GitHub repository is https://github.com/Sininliny/Alfamedomega.

Several agents share the one repository in `/mnt/project-files/Alfamedomega`. Each works in
its own `git worktree` on its own branch, so nobody switches branches under anyone else:

```bash
git -C /mnt/project-files/Alfamedomega worktree add ~/wt-dsp -b dsp/wavetable-oscillator main
```

- Stage **only your own directory** (`git add src/dsp`); never `git add -A` from the root.
- Merge `main` into your branch to pick up changes; never rewrite history
  (`rebase`, `amend`, `reset --hard`, force-push) on a branch someone else may have.
- If `.git/index.lock` exists, another commit is in progress: wait and retry. Never delete it.
- The repo-and-workflow thread pushes branches and opens pull requests; a branch merges to
  `main` only with green CI.
- In a fresh container, trust the folder once:
  `git config --global --add safe.directory /mnt/attach/project-files/Alfamedomega`
  (and the same for `/mnt/project-files/Alfamedomega`).

### Versions and releases

Semantic versioning in `project(VERSION ...)` of the root `CMakeLists.txt`. Releases are
tagged `vMAJOR.MINOR.PATCH`; signing, notarization and installers run only on tags
(see `docs/packaging/packaging-plan.md`). Model weights are versioned separately
(`<tier>-<version>`) and never committed to git.

## Decisions

Anything that is hard to reverse (a dependency, a file format, plugin identity codes, the
model's output representation) gets an ADR in `docs/decisions/`, numbered in order, with
status `Proposed` until Sinin confirms it.
