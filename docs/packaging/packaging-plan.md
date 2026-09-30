# Alfamedomega packaging plan: VST3 and AAX (draft v1, 2026-09-29)

Scope: how Alfamedomega gets from source to a signed, installable VST3 and AAX plugin on
Windows and macOS. It builds on [ADR 0001](../decisions/0001-plugin-and-inference-stack.md)
(JUCE 8 + ONNX Runtime CPU + anira, CMake everywhere) and the
[model tiers](../tiers/model-tiers.md). No code exists yet, so this is a plan, not a runbook.

Items marked **(verify)** are facts about third-party programs that change often; check them
against the vendor's current page before paying for or signing anything.

## 1. Targets

| Format | Windows | macOS | Notes |
|---|---|---|---|
| VST3 | x64 | Universal (arm64 + x86_64) | Main format. Linux VST3 is possible later at little cost. |
| AAX | x64 | Universal | Pro Tools only. Requires Avid and PACE programs (section 5). |
| AU | – | Universal | Not asked for, but JUCE builds it from the same target almost for free and Logic users need it. Recommended. |
| Standalone | x64 | Universal | Useful for testing and demos; optional to ship. |
| Control center | x64 | Universal (arm64 + x86_64) | The Python/Qt app in `apps/control-center`, frozen with PyInstaller. See section 7. |

Minimum OS: Windows 10 x64, macOS 11 (first release with Apple Silicon). Windows on ARM is
out of scope for v1 (Pro Tools does not run there).

## 2. Build pipeline

One CMake project; `juce_add_plugin(... FORMATS VST3 AAX AU Standalone)` produces every
format from one target, and the shared `Alfamedomega_SharedCode` static library holds the
DSP, anira, and ONNX Runtime.

- **Dependencies:** JUCE 8 (git submodule or CPM, pinned tag), anira (pinned), ONNX Runtime
  (see 2.1), AAX SDK (not redistributable publicly; path passed with
  `juce_set_aax_sdk_path`, fetched from a private location in CI). The VST3 SDK ships
  inside JUCE.
- **CI:** GitHub Actions, two jobs per commit:
  - `macos-14` (Apple Silicon): `-DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"`,
    `-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0`, Xcode generator or Ninja.
  - `windows-2022`: MSVC x64 with the DLL runtime (`/MD`). anira's prebuilt static ONNX
    Runtime is compiled `/MD` and does not link into `/MT` targets, so the installer ships the
    VC++ runtime (section 6).
- **Every commit:** build, unit tests, pluginval, and the ONNX exportability check the ADR
  already requires. **Tagged releases only:** signing, notarization, installers.
- **Plugin identity, fixed forever before the first public build:** 4-char manufacturer
  code (at least one uppercase letter), 4-char plugin code, bundle ID
  (`com.<you>.alfamedomega`), VST3 class ID (JUCE derives it), AAX product and plugin IDs.
  Changing any of these later breaks every saved session and preset.

### 2.1 Shipping ONNX Runtime inside a plugin

This is the riskiest packaging item, because a plugin shares its process with every other
plugin the host has loaded.

- **Link it statically**, with hidden symbols (`-fvisibility=hidden`; on Windows no
  `onnxruntime.dll` at all). Windows ships its own older `onnxruntime.dll` in System32
  for Windows ML, and other plugins may bring different versions; a shared DLL lookup
  can silently load the wrong one and crash the host.
- **Use a reduced-operator build** (ORT's `--include_ops_by_config` from the op list of all
  tier models). This cuts the library from tens of MB to a few MB per architecture. Cost:
  a new op in a future model needs a plugin rebuild, so the export CI job should fail if a
  model uses an op outside the config.
- Bundle size estimate: plugin binary 10–25 MB per architecture, roughly double for the
  macOS universal binary, per format. Measure once the first build exists.

## 3. Shipping the model weights

Tier weights from the tier plan: Lite 2 MB, Standard 24 MB, Pro 80 MB, Studio 320 MB.

**Recommendation: weights live outside the plugin bundles, in one shared folder**, so
VST3, AAX, AU and standalone all use one copy and the bundles stay small for signing:

- macOS: `/Library/Application Support/Alfamedomega/Models/`
- Windows: `C:\ProgramData\Alfamedomega\Models\`

Layout, one folder per tier and model version:

```
Models/
  manifest.json            # tier, model version, file, SHA-256, ORT opset, latency samples
  standard-1.0.0/model.onnx
  lite-1.0.0/model.onnx
```

- **What installs:** the base installer carries Lite and Standard (about 26 MB). Pro and
  Studio are optional installer components, ticked by default only if the machine meets
  the tier's hardware line; a later "download tier" button in the UI can fetch them too.
- **Loading:** off the audio thread, verify SHA-256 against the manifest, then build the
  anira session. One loaded model per tier per process, shared between instances, so
  twenty Standard instances in a session do not load twenty copies of the weights.
- **Sessions and presets** store tier and model version. If a project opens on a machine
  without that tier, it falls back to the nearest installed tier and says so in the UI,
  never silently.
- **Latency per tier** is reported with `setLatencySamples`. Switching tier changes latency;
  some hosts (Pro Tools especially) only re-apply delay compensation after transport stops,
  so tier switching should be a "stop, switch, play" action, not live.
- **The Models folder is writable by users.** The control center imports, removes and
  re-verifies weights there without asking for an admin password, so the installer sets
  permissions when it creates the folder, and again on every upgrade:
  - Windows (Inno Setup): `[Dirs] Name: "{commonappdata}\Alfamedomega\Models";
    Permissions: users-modify`. The default ProgramData ACL is not enough on its own: it lets
    users add files but not replace or delete the ones the installer put there.
  - macOS (`.pkg` postinstall script): `chgrp -R staff` plus `chmod -R g+rwX` on
    `/Library/Application Support/Alfamedomega/Models`. Every local user is in `staff`. The
    script runs on every install, because a package upgrade resets the payload to root-owned.
  - **Trade-off:** any process running as any local user can now change the models that
    every DAW loads. ONNX files are data, not code, but a malformed one still reaches ONNX
    Runtime's parser. The SHA-256 manifest catches corruption, not tampering, because the
    manifest is writable too. That is acceptable on a personal music machine. Signing the
    manifests of official releases, with a public key built into the plugin, would close the
    gap if it ever matters.
  - **Changing files under a running plugin.** The control center writes each model into a
    temporary folder inside `Models/`, renames it into place, and replaces `manifest.json`
    atomically (write to a temporary file, then rename). The plugin reads the manifest again
    when its tier menu opens. ONNX Runtime reads a model into memory when it loads, so
    removing a tier that a DAW has loaded does not crash the DAW.
- **Weights as IP:** any file on a user's disk can be copied. Light encryption of the
  model files deters casual extraction but not a determined user; decide whether it is
  worth the complexity (section 8).

## 4. VST3

- **SDK licence:** Steinberg released the VST 3 SDK under the MIT licence with version 3.8
  (October 2025), replacing the old proprietary/GPLv3 dual licence, so no signed
  agreement is needed to ship VST3. The VST logo still has usage guidelines if you want to
  display it. **(verify)**
- **Install locations:** `C:\Program Files\Common Files\VST3\Alfamedomega.vst3` and
  `/Library/Audio/Plug-Ins/VST3/Alfamedomega.vst3`.
- **Validation:** pluginval at strictness 10 plus Steinberg's `validator` from the SDK, in CI.
  Manual smoke tests in at least Reaper, Ableton Live, Cubase and FL Studio before release.

## 5. AAX

AAX is the one format that is gated by third parties. Getting it to users takes three things
that only you can set up:

1. **Avid developer account and AAX SDK.** Register at Avid's developer site, accept the
   AAX SDK terms and download it. Recent AAX SDKs are offered under a dual licence (Avid's
   own, or GPLv3); read which one fits your licensing plan (section 8). **(verify)**
2. **Pro Tools Developer build.** Avid gives AAX developers a Pro Tools build that loads
   unsigned AAX plugins. Everything in development is tested there first.
3. **PACE Eden signing.** Retail Pro Tools refuses unsigned AAX plugins. You request the
   Eden SDK / `wraptool` from PACE (contact through Avid's developer program), get an iLok
   account, and PACE issues you a signing identity (a "WCGUID"). There is no fee for AAX
   signing-only use at the time of writing. **(verify)**

Build notes:
- JUCE builds the AAX target from the same code; Alfamedomega registers as an AAX
  instrument with MIDI input.
- `wraptool sign` both PACE-wraps and Apple-codesigns the `.aaxplugin` on macOS (pass the
  Developer ID with `--signid`), and signs it on Windows with your Authenticode certificate.
  So on macOS the order is **wraptool first, then notarize**; running `codesign` after
  wraptool would break the PACE signature.
- `wraptool` needs a logged-in iLok account, which is awkward in hosted CI. Plan: sign AAX
  on a self-hosted runner (or manually on your machine) for the first releases, and move
  it into CI once PACE's current cloud-signing option is confirmed. **(verify)**
- Install locations: `C:\Program Files\Common Files\Avid\Audio\Plug-Ins\` and
  `/Library/Application Support/Avid/Audio/Plug-Ins/`.
- Validation: Avid's AAX Validator (ships with the SDK tools) plus a session test in the
  Pro Tools Developer build and then a signed build in retail Pro Tools.

## 6. Code signing and notarization

### macOS

- Needs an **Apple Developer Program** membership (US$99/year), which gives a
  "Developer ID Application" certificate (bundles) and a "Developer ID Installer"
  certificate (the `.pkg`).
- VST3 and AU: `codesign --options runtime --timestamp` with the Developer ID Application
  identity. Hardened runtime entitlements should stay minimal; the plugin does not JIT, and
  ONNX Runtime's CPU provider does not need `allow-jit`.
- AAX: `wraptool` as above.
- Installer: a `.pkg` built with `pkgbuild`/`productbuild`, one component per format plus a
  models component per tier, signed with the Installer certificate.
- Notarize the final `.pkg` with `xcrun notarytool submit --wait` using an App Store Connect
  API key (stored as a CI secret), then `xcrun stapler staple`.

### Windows

- Needs an **Authenticode code-signing certificate**. Since 2023 private keys must live in
  hardware or a cloud HSM, so a plain `.pfx` file is no longer an option. Choices:
  - **Microsoft Azure Trusted Signing (Artifact Signing):** cheapest (around US$10/month)
    and CI-friendly, but eligibility depends on country and whether you sign as an
    individual or an organisation. **(verify eligibility for you)**
  - **OV certificate on a cloud HSM** (SSL.com eSigner, DigiCert KeyLocker, and similar):
    a few hundred US$ per year, works anywhere.
  - EV no longer buys instant SmartScreen reputation, so it isn't worth the premium here.
- Sign the `.vst3` DLL, the AAX binary (through wraptool), the standalone `.exe`, and the
  installer, all with RFC 3161 timestamps.
- Installer: Inno Setup (simplest) with components per format, per model tier, and the control center.
- **VC++ runtime:** the installer runs the Microsoft Visual C++ 2015–2022 x64
  redistributable (`vc_redist.x64.exe /install /quiet /norestart`), in a version at least as
  new as the toolset that built ONNX Runtime 1.26 and the plugin. It skips the step if an equal
  or newer version is already installed (check the `Version` value under
  `HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64`).
- **Runtime version risk inside a DAW:** with `/MD` the plugin uses whichever
  `msvcp140.dll` the host process already loaded. A host that ships its own older copy can
  crash plugins built with VS 2022 17.10 or later, the known `std::mutex` constexpr-constructor
  break. Define `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` for all Windows targets, and include
  one DAW that bundles an old runtime in the Windows smoke tests.

## 7. Control center app

`apps/control-center` is a Python 3.10+ app built on PySide6 (`alfa-cc-gui`), with a CLI
(`alfa-cc`). It installs next to the plugins, is signed with the same identities, and reads
and writes the same Models folder (section 3).

**Freezing: PyInstaller in `--onedir --windowed` mode.** Briefcase also works and has signing
built in, but it expects to own the installer and would produce a second one. PyInstaller
output drops into the same `.pkg` and Inno Setup installers as the plugins, as one more
component. Don't use `--onefile`: it unpacks to a temp folder on every launch, starts slowly,
and antivirus tools flag it far more often.

- **Outputs:** `Alfamedomega Control Center.app` in `/Applications`, and
  `C:\Program Files\Alfamedomega\Control Center\` with a Start-menu entry. The CLI goes in the
  same bundle, as a second console executable on Windows. An optional `alfa-cc` symlink in
  `/usr/local/bin` can come later.
- **macOS architectures:** build with the universal2 python.org Python and
  `--target-arch universal2`. PySide6 wheels are universal2. If any dependency ships only
  single-architecture wheels, fall back to separate arm64 and x86_64 builds. **(verify on
  first build)**
- **Size:** roughly 80–150 MB with PySide6-Essentials, before trimming unused Qt modules.
  This is an estimate; measure it on the first build.
- **PyTorch is not bundled.** Training runs in a separate Python that has PyTorch, set in the
  `training_python` setting. The frozen app stays small, and its signature does not have to
  cover a gigabyte of CUDA libraries.

**Running training from a packaged app.** A frozen app has no Python of its own and no source
checkout around it. `alfacc/settings.py` already handles this: when frozen, it finds the
training script and the preference learner through the `repo_dir` setting (the user's
Alfamedomega checkout), and the training Python through `training_python`, then a `.venv` in
that checkout, then the Python on `PATH`. Packaging adds nothing here, except a first-run prompt
for `repo_dir` if it is empty.

**Signing:**
- macOS: PyInstaller's `--codesign-identity "Developer ID Application: …"` with hardened
  runtime signs every bundled `.so`, `.dylib` and Qt framework inside-out, then the `.app`.
  Start with no entitlements on the app itself and add one only if a notarized build fails at launch (the render host, below, is the exception). CPython
  does not JIT, and every library is signed with our own identity, so neither `allow-jit`
  nor `disable-library-validation` should be needed. The app is notarized as part of the
  same `.pkg` as the plugins.
- Windows: sign both executables and every bundled `.dll`/`.pyd` that isn't already signed.
  Smart App Control and some antivirus tools check each file, not just the `.exe`. PyInstaller
  bootloaders are a common antivirus false positive even when signed. If one is reported,
  submit the build to Microsoft's malware analysis portal, and consider rebuilding the
  bootloader from source.

**Plugin assimilation additions** (branch `control-center/plugin-assimilation`,
`docs/control-center/plugin-assimilation.md` section 10):
- Bundle the `alfa-render-host` binary (built from `apps/control-center/render-host/` with
  CMake) next to the frozen executable. The app looks for it in `sys._MEIPASS` or in the
  executable's own folder. Use PyInstaller's `binaries=` so the host is signed with everything else.
- Include `alfacc/assimilate/analyze_worker.py` as a PyInstaller data file. It runs under
  the training Python, not inside the frozen app.
- **The render host is the one signing exception on macOS.** It loads other vendors'
  plugins, so it is signed with our identity and hardened runtime plus the
  `com.apple.security.cs.disable-library-validation` entitlement. Without that
  entitlement, macOS refuses to load any plugin signed by another team. Some older plugins also
  need `com.apple.security.cs.allow-unsigned-executable-memory`. Add it only if a real
  plugin fails to load, and keep both entitlements on the render host only, never on the app or
  the plugins.
- Ship the render host as a universal binary and **keep its x86_64 slice**. On Apple
  Silicon, when a plugin fails to scan, the control center retries it under Rosetta
  (`arch -x86_64 alfa-render-host ...`) and remembers that for the plugin. Stripping the host
  to arm64 only would break Intel-only plugins.
- Windows: sign `alfa-render-host.exe` like the other binaries. Third-party VST3s it
  loads keep their own signatures (or none), which does not affect ours.
- The render host builds in the same CI job as the plugins, and is frozen into the app
  afterwards, so the app job depends on it.

**CI:** add a job to the same workflow on both runners that runs the app's `pytest`
suite, freezes the app, and launches the frozen app once with `alfa-cc info` as a smoke test.
Signing and packaging happen on tags, together with the plugins.

## 8. Decisions needed

1. **Licensing model.** The repo is Apache-2.0, and JUCE is AGPLv3 or commercial. Shipping
   closed-source needs a paid JUCE licence tier; shipping open-source means the plugin
   binaries are AGPLv3. The AAX SDK licence choice follows from this.
2. **Include AU?** Recommended yes.
3. **Weights protection:** plain files (recommended for v1) or encrypted.
4. **How Pro and Studio reach users:** optional installer components (recommended for v1)
   or in-app download (needs a download server and update checks).
5. **Where imported models go.** Recommended: the shared Models folder, made writable at
   install (section 3). Alternative: a second, per-user Models folder that the plugin also
   searches. That needs no permission changes, but two places to look is harder to explain.
6. **Legal entity name** for certificates. Apple, the Windows certificate and PACE all show
   the signer's name to users; individual or company should be decided before any of them
   is requested, because changing later means reissuing all three.

## 9. Steps only Sinin can do

| # | Step | Cost (verify) | Needed for |
|---|---|---|---|
| 1 | Decide individual vs company name for signing | – | everything below |
| 2 | Enrol in the Apple Developer Program; create Developer ID Application + Installer certs; create an App Store Connect API key for notarization | US$99/yr | macOS release |
| 3 | Get a Windows code-signing option (Azure Trusted Signing if eligible, else an OV cloud-HSM cert); identity verification takes days | ~US$120/yr to a few hundred | Windows release |
| 4 | Register as an Avid developer, download the AAX SDK, request the Pro Tools Developer build | free | AAX development |
| 5 | Contact PACE for Eden / wraptool, create an iLok account, receive a WCGUID | free for AAX signing | AAX release |
| 6 | Pick the JUCE licence tier (or commit to AGPLv3) | from free | any public release |
| 7 | Add the signing secrets (Apple certificates and API key, Windows signing credentials) to the GitHub repository | free | release builds in CI |

Steps 2–5 involve identity checks and can take one to several weeks, so they are worth
starting well before the first build is ready.

## 10. Repository skeleton

The repository now exists: `main` has the CMake skeleton, `cmake/` dependency and compiler
helpers, CI in `.github/workflows/ci.yml`, and the control center. The packaging pieces are not
created yet. As agreed, they wait until the stack and licensing are settled:

```
cmake/                      # add: ORT reduced-op build, AAX SDK lookup, signing helpers
packaging/macos/            # pkg scripts and postinstall (Models permissions), notarize.sh
packaging/windows/          # Inno Setup script (Models permissions), sign.ps1
packaging/control-center/   # PyInstaller spec for the GUI and CLI
.github/workflows/release.yml  # sign, notarize and build installers on tags
```
