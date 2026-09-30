# Batch Adaptive Preference Learning in the control center

- **Status:** Proposed
- **Date:** 2026-09-29
- **Owners:** the control center runs batches; the learner itself belongs to `preference/`
  ("Add adaptive preference learning").

## What a batch does

1. **Index the libraries.** The user picks sample or preset folders. The control center walks
   them and writes one JSON line per audio file to `<home>/preference/library_index.jsonl`:

   ```json
   {"path": "/Users/me/Samples/Pads/warm_pad_03.wav", "library": "Samples", "rel": "Pads/warm_pad_03.wav",
    "format": "wav", "bytes": 1234, "mtime": 1790000000, "sr": 48000, "channels": 2, "bits": 24,
    "duration_s": 2.5, "name_labels": ["warm", "pad"], "folder_labels": ["pad"]}
   ```

   Labels are lower-case words from the file and folder names, with numbers, note names, versions
   and filler words ("final", "loop", "wav") dropped and simple plurals folded. Rescans reread
   only files whose size or modification time changed. A label count goes to
   `<home>/preference/vocabulary.json`.

2. **Run the learner** on the index and the plugin's behaviour log:

   ```
   <learner> batch --index <home>/preference/library_index.jsonl
                   --events <home>/preference/events
                   [--assimilation <home>/assimilation]
                   [--consent <home>/preference/plugin.json]
                   [--labels <home>/preference/labels.jsonl]
                   --out <home>/preference/profiles/<timestamp>
   ```

   Each optional flag is passed only when `learner.py` or the `alfapref/cli.py` next to it
   mentions it, so an older learner still runs. `--consent` names the switches file (below);
   `--labels` is written from the user's hand labels, one `{"path", "labels"}` per line.

   `--assimilation` is added when plugin assimilation has analysed notes and the learner script
   supports the flag ([plugin-assimilation.md](plugin-assimilation.md), section 10).

   `<learner>` is `python <repo>/preference/learner.py`, found from the repository folder in
   Settings and run with the training Python (it needs numpy), or any command set there
   explicitly. The learner prints `progress N/M` lines for the progress bar, exits non-zero on
   failure, and writes `profile.json`, `library.jsonl` and `summary.json` into `--out`.
   `--help` exits 0. Agreed with the preference thread on 2026-09-29; checked end to end
   against `preference/adaptive-learning`.

   The learner recomputes words from each record's `rel`, so `name_labels` and
   `folder_labels` in the index are for display. The control center computes them with
   `preference/alfapref/tokens.py` from the repository folder (loaded by file path), and
   falls back to its own copy of those rules only when no checkout is set.

## Learning from how the user plays (opt-in)

Sinin decided on 2026-09-30 that behaviour learning is strictly opt-in, visible and removable
(`docs/preference/design.md` section 7.1). The control center's Preference page shows the same
switches the plugin reads, in `<home>/preference/plugin.json`:

```json
{"learn": true,
 "categories": {"performance": true, "parameters": true, "suggestions": true, "workflow": false},
 "personalization": 0.6}
```

- The switch is off unless the user turns it on; a missing file means off. `{"learn": true}` on its
  own (what the plugin writes) turns every category on. Keys the plugin or learner add are kept.
- **What's learned** shows the latest profile's `summary.json` `explain` lines.
- **Reset learning** runs `<learner> reset --profile <latest>` (with `--all` for "Reset and drop
  examples"), deletes older profiles and the behaviour log. Libraries, the index and the user's own
  words stay.
- **Tag or label files** stores tags, a description and hand labels per file in
  `<home>/preference/annotations.json`. Scans copy them into the index records as `tags`,
  `description` and `labels`, and hand labels also go to `labels.jsonl` for `--labels`.

CLI: `alfa-cc pref learn on|off [--category workflow=off] [--personalization 0.6]`,
`alfa-cc pref explain`, `alfa-cc pref reset [--all]`,
`alfa-cc pref label FILE --tags pad --labels warm,airy --description "slow swell"`.

The learner's `--presets` (the user's own presets) is not fed yet: the control center has no
preset list of its own until the plugin's preset format is settled.

## Privacy

- Nothing leaves the machine. The control center makes no network calls.
- The index and `annotations.json` hold absolute paths, sizes and words, never audio. It is a local cache; learned
  profiles don't contain paths.
- **Forget everything** deletes the index, vocabulary, behaviour log and all profiles. The list
  of chosen folders stays so the user can re-run.

## Proposals for other threads

- **Plugin**: append implicit feedback (load, play time, keep, save, reuse, small edit, quick
  discard) as JSON lines to `<home>/preference/events/<date>.jsonl`, with `<home>` from
  `alfacc.settings.default_home()`. The event schema belongs to `preference/`.

If the learner can't be found, a batch indexes the libraries and stops, and says so in its log.
