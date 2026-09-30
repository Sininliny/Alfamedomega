# Local Plugin Assimilation

Status: first version implemented (`apps/control-center/src/alfacc/assimilate`, `render-host/`).
Owner: control center.

The control center learns the timbres of instruments the user already owns. It scans installed
instrument plugins, lets the user pick some, plays every preset across its playable range with
standard notes, and analyses the recordings on this computer into a personal timbre knowledge
base. That knowledge base feeds sound matching, preference learning, synthesis-space expansion
and user-specific adaptation of the model.

A preset is treated as a function of pitch, velocity and note length: `timbre(key, velocity,
duration)`. Assimilation samples that function on a grid inside the preset's playable range.

## 1. Pipeline

```
discover files ─► open (scan) ─► list presets ─► find range ─► render grid ─► analyse ─► knowledge base
  folders          render host     programs,       probes +       chosen         NumPy       features.jsonl
  per OS           per file,       .vstpreset,     key-by-key     density        worker      presets.json
                   own process     default         edges
```

Every step that touches a plugin runs in `alfa-render-host`, a separate process started per call.
A plugin that crashes or hangs fails that call, which is recorded, and the control center moves on.

## 2. Which plugin formats a third-party host can load

| Format | Loadable by our host | Notes |
|---|---|---|
| **VST3** | Yes: macOS, Windows, Linux | Steinberg's VST3 SDK is open source (MIT since SDK 3.8), and JUCE bundles the hosting parts. The main path. |
| **AU** (Audio Unit v2/v3) | macOS only | Apple's own API; every Mac instrument ships one. |
| **VST2** | Only with a licensed VST2 SDK | Steinberg stopped licensing the VST2 SDK in 2018 and it can't be redistributed. The render host is built without VST2 unless `ALFA_VST2_SDK_PATH` points to a copy the builder is licensed for. Almost every VST2 instrument still sold also ships VST3. |
| **AAX** | No | Only Pro Tools and Avid's own tools host AAX. The SDK licence doesn't allow third-party hosts, and most AAX plugins are PACE/iLok wrapped and refuse to run elsewhere. Nearly every AAX instrument also ships VST3 or AU, so the page shows AAX files as "use the VST3 or AU version" and links the twin when it is installed. |
| CLAP | Not yet | Open and easy to host; JUCE 8 doesn't host it. A later addition if users ask. |
| LV2, LADSPA | Off | Linux effects mostly; instruments are rare. |

So "supported AAX instruments" in practice means the same instrument through its VST3 or AU twin.
Discovery matches twins by normalised name (`discovery.py`).

## 3. Render host

`apps/control-center/render-host` is a JUCE 8 console app, built separately from the plugin. JUCE
is already a project dependency (AGPLv3 for now), so this adds no new licence. Pedalboard and
DawDreamer were considered and rejected: both are GPLv3, and a separate small host gives us crash
isolation and exact control of note timing anyway.

Commands (all output is JSON or `progress i/N` lines on stdout):

| Command | Does |
|---|---|
| `formats` | Formats this build can host, e.g. `["VST3","AudioUnit"]`. |
| `scan [--programs] FILE...` | Plugins inside each file: name, vendor, format, uid, version, category, is_instrument, channel counts, and program names. |
| `render JOB.json` | Renders every note for every preset in the job to 32-bit float WAV, writes `renders.json`. |

A render job gives the plugin file and uid, sample rate (48 kHz), block size, and lists of presets
(`program` index, `.vstpreset` file, raw `state`, or `default`) and notes (`id, midi, velocity,
dur_s, tail_s, tail_mode`). For each preset the host loads it, lets the plugin settle in silence,
then for each note: note-on, hold for `dur_s`, note-off, and capture the release until it has
stayed below −90 dBFS for 50 ms or `tail_s` passes. Between notes it sends all-notes-off and
all-sound-off and resets the plugin, so one note's tail never leaks into the next. Rendering is
offline (`setNonRealtime(true)`), usually much faster than real time.

On Apple Silicon the render host is built universal and runs natively, so it can't load an
Intel-only plugin. When opening a plugin fails there, the scan tries again under Rosetta
(`arch -x86_64 alfa-render-host ...`); if that works, the plugin is marked and every later render
of it runs under Rosetta too.

`render-host/test-synth` builds **AlfaTestSynth**, a tiny VST3 with three known programs (a sine on
every key, a saw that is silent outside C1–C4, and decaying noise). The end-to-end test renders it
and checks the ranges found are exactly those.

## 4. Presets

For each opened instrument the preset list is, in order:

1. The plugin's programs (`getNumPrograms`), keys `p0000…`.
2. On VST3, `.vstpreset` files in the standard VST3 preset folders for that vendor and plugin
   (`~/Library/Audio/Presets/<vendor>/<plugin>`, `Documents/VST3 Presets/...`, `~/.vst3/presets/...`),
   keys `f00000…`, with the subfolder as the category.
3. If neither gives anything but "Default" or "Init", one `default` preset: the state the plugin
   opens in.

Many commercial instruments keep presets in their own browser and format, not as programs or
`.vstpreset` files. Those need a per-vendor reader, or the user saving presets as `.vstpreset` from
a DAW. A proposal for this is under "Later" below.

## 5. Finding the playable range

A preset is played only where it makes a usable sound.

1. **Coarse probes**: MIDI 12–120 every 6 semitones (19 notes), velocity 100, held 0.4 s, 1 s tail.
   All selected presets go in one host call; if it fails, each preset is retried in its own call, so
   one bad preset doesn't cost the others.
2. **Decide**: a key is *audible* if its peak is above −60 dBFS and within 40 dB of the preset's
   loudest probe. A preset is *pitched* if at least 60% of its audible keys sound at the requested
   pitch, octave shifts allowed, within ±50 cents (autocorrelation, confidence ≥ 0.6). For a pitched
   preset a key is usable only if it tracks pitch; for an unpitched one (drums, noise, FX), any
   audible key is usable. The range is the longest unbroken run of usable keys.
3. **Edges**: every key between each end of the range and the next probe outside it is rendered and
   judged the same way, so the range ends on the exact key. The range never grows past the probed
   span (C0 to C9).

Presets that are silent or have no usable key are marked and skipped. On AlfaTestSynth this gives
A0–G8 pitched for the sine, exactly C1–C4 for the bass, and C0–C9 unpitched for the noise.

## 6. Sampling densities

Grids are anchored on C (C1 = MIDI 24, C4 = 60) so octaves line up across presets, and always
include both ends of the range.

| Density | Keys | Velocities | Note lengths | Max tail | Renders for C1–C6 |
|---|---|---|---|---|---|
| **Fast** | every 12 semitones | 100 | 1 s | 2 s | 6 |
| **Normal** | every 5 semitones | 40, 80, 120 | 1 s | 3 s | 39 |
| **Detailed** | every semitone | 24, 48, 72, 96, 112, 127 | 1.5 s | 4 s | 366 |
| **Unreasonable** | every playable key | 16, 32, 48, 64, 80, 96, 112, 127 | 0.25, 1, 4 s | 10 s | 1,464 |

The request said Normal is every 4–6 semitones with 3 velocities and Detailed 4–8 velocities; the
defaults take 5 and 6, and `--step`, `--velocities` and `--durations` on the CLI make a custom
density. Unreasonable also turns on release analysis: three note lengths show how the release
depends on how long the key was held. Before starting Unreasonable the window asks for
confirmation, since it can take hours per preset. The page shows an estimate for the selected
presets before ranges are known.

## 7. Analysis and the knowledge base

`analyze_worker.py` runs in the training Python (it needs NumPy only), like the preference learner,
so the control center itself stays standard-library only. For each rendered note it writes one JSON
line:

| Field | Meaning |
|---|---|
| `preset, note, midi, velocity, dur_s` | What was played |
| `peak_db, rms_db` | Level (RMS over the active part of the held note) |
| `attack_ms` | Time to 90% of the held peak |
| `release_ms, release_truncated` | Note-off to −60 dB below the note-off level |
| `f0_hz, cents, octave, f0_confidence, pitched` | Pitch tracking against the requested key |
| `centroid_hz, flatness` | Spectral brightness and noisiness of the held part |
| `descriptor_id, descriptor` | The preference learner's 134-number `logmel-stats-v1` descriptor, from `preference/alfapref/descriptor.py`, when the checkout is available |

Using the preference descriptor means harvested presets live in the same timbre space as the
user's sample libraries, so preference learning and sound matching can use them without a new
embedding.

Files, under the control center's data folder:

```
assimilation/
  catalog.json                      plugin files found; plugins opened; files that failed to open
  <vendor>-<plugin>-<format>/
    plugin.json                     the plugin as the host reported it, and its preset list
    presets.json                    per preset: status, range, pitched, octave shift, density, counts
    features.jsonl                  one line per rendered note
    work/                           only while a job runs, or with "Keep audio"
```

Rendered audio is deleted after analysis unless **Keep audio** is ticked. A new harvest of a preset
replaces its earlier features. **Forget this plugin** deletes that plugin's folder; the plugin itself
is never touched.

## 8. Using it in the control center

Window: **Plugin assimilation** page. *Find plugins* lists every plugin file in the standard folders
(and the extra folders from Settings) with whether it can be opened. *Open selected* scans the ticked
files, one process each. Then pick an instrument, tick presets, choose a density, and *Start*. The job
runs two steps, *Find playable ranges* and *Render and analyse*, with progress, log and cancel on the
Jobs page.

CLI:

```bash
alfa-cc plugins discover
alfa-cc plugins scan                          # every loadable plugin found, or name files
alfa-cc plugins list
alfa-cc plugins presets Pigments
alfa-cc plugins estimate Pigments
alfa-cc plugins assimilate Pigments --density detailed --presets p0000,p0003
alfa-cc plugins show Pigments --preset p0003  # the analysed notes as JSON lines
alfa-cc plugins forget Pigments
```

Settings: `render_host` (else `$ALFA_RENDER_HOST`, a copy next to the packaged app, or `PATH`) and
`plugin_folders` (extra folders, separated by the OS path separator).

## 9. Privacy and licences

Everything stays on the user's computer: plugins are loaded locally, audio is analysed locally and
deleted, and nothing is uploaded. Presets and plugin audio belong to their vendors; the knowledge
base holds analysis features, not audio, and is for the user's own model. If it is ever used to
train shared models, that needs a separate opt-in and a licence review. It is off by design here.

## 10. For other threads

- **Preference learning**: `features.jsonl` lines with a `descriptor` can be read as extra items with
  weak labels from preset names and categories (the preference tokenizer applies). Proposed: the
  learner's batch command accepts `--assimilation <home>/assimilation`.
- **Training**: the knowledge base is a list of (plugin, preset, key, velocity, duration) → timbre
  features. It can drive synthesis-space expansion (find regions no family covers well) and a per-user
  fine-tune. With **Keep audio** on, the WAVs plus `presets.json` are a ready rendering set.
- **Packaging**: bundle `alfa-render-host` next to the frozen control center (it is found there), and
  include `alfacc/assimilate/analyze_worker.py` as a data file. On macOS the render host must be
  signed with the same identity and needs the `com.apple.security.cs.disable-library-validation`
  entitlement to load third-party plugins.
- **Repo and CI**: a CI job that builds `render-host/` on Linux and runs
  `pytest tests/test_assimilate.py` with `ALFA_RENDER_HOST` and `ALFA_TEST_SYNTH` set.

## 11. Later

- Vendor preset readers for instruments that don't expose presets as programs or `.vstpreset`.
- CLAP hosting.
- Running several render hosts at once (one per core) for Detailed and Unreasonable.
- Multi-sampling round robins and legato transitions.
