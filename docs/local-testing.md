# Testing Alfamedomega locally

These commands mirror the CI jobs in `.github/workflows/ci.yml`. The first configure downloads JUCE, Catch2 and ONNX Runtime, so it needs internet access and takes a few minutes. Later builds are incremental.

## 1. Install the prerequisites (once)

| | Windows | macOS | Linux |
|---|---|---|---|
| Compiler | Visual Studio 2022, "Desktop development with C++" workload | Xcode or `xcode-select --install` | GCC 13+ or Clang 17+ |
| CMake 3.25+ | Included with VS, or cmake.org | `brew install cmake` | `sudo apt install cmake` |
| Ninja | Not needed | `brew install ninja` | `sudo apt install ninja-build` |
| Python 3.11+ | python.org | `brew install python@3.12` | `sudo apt install python3 python3-venv` |
| git | git-scm.com | Included with Xcode | `sudo apt install git` |

The inference tests export small ONNX models, so install these Python packages too:

```
python -m pip install numpy onnx
```

On Linux, the plugin build also needs the JUCE libraries:

```
sudo apt install libasound2-dev libfreetype-dev libfontconfig1-dev libx11-dev libxcomposite-dev libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev libxrender-dev
```

## 2. Get the code and pick what to test

```
git clone https://github.com/Sininliny/Alfamedomega.git
cd Alfamedomega
git fetch origin
```

Then check out one of these:

- To test main: `git checkout main && git pull`
- To test a PR branch: `git checkout --detach origin/<branch-name>` (the branch name is shown on the PR page)

## 3. Build and run the C++ tests and the plugin

Use a new build folder, or delete `build`, whenever you switch branches.

**Windows.** Open the "Developer PowerShell for VS 2022" from the Start menu:

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DALFA_BUILD_PLUGIN=ON -DALFA_BUILD_TESTS=ON -DALFA_WARNINGS_AS_ERRORS=ON
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

**macOS:**

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DALFA_BUILD_PLUGIN=ON -DALFA_BUILD_TESTS=ON -DALFA_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

To build a universal binary the way CI does, add `"-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64"` to the first command. It's slower.

**Linux.** Use the presets:

```
cmake --preset core  && cmake --build --preset core  && ctest --preset core
cmake --preset asan  && cmake --build --preset asan  && ctest --preset asan    # AddressSanitizer run
```

To build the plugin on Linux as well, use the `release` preset.

A pass ends with `100% tests passed`. If a test fails, copy the failing test's output and send it in the thread.

The plugin is written to `build/src/plugin/Alfamedomega_artefacts/Release/VST3/Alfamedomega.vst3`. On macOS there is also `.../AU`. Copy it into your plug-in folder to try it in a DAW.

## 4. Validate the plugin with pluginval (optional, recommended for plugin changes)

Download pluginval for your OS from https://github.com/Tracktion/pluginval/releases/latest, unzip it, and run:

```
pluginval --strictness-level 5 --validate-in-process --skip-gui-tests --validate build/src/plugin/Alfamedomega_artefacts/Release/VST3/Alfamedomega.vst3
```

On macOS, the binary is `./pluginval.app/Contents/MacOS/pluginval`. On Windows it is `.\pluginval.exe`.

## 5. Run the Python tests (control center and preference learning)

Run these from the repo root. On Windows, use `.venv\Scripts\activate` instead of the `source` line.

```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e "apps/control-center[gui,test]"
python -m pytest apps/control-center
cd preference && python -m unittest discover tests && cd ..
```

## 6. Check formatting (optional)

This needs clang-format 18 and bash, so on Windows use Git Bash:

```
bash scripts/format.sh --check
```

## What matters most

While GitHub Actions is unavailable, the Windows and macOS builds and `ctest` (step 3) are the checks nobody else can run. Run pluginval (step 4) on any change to `src/plugin`.
