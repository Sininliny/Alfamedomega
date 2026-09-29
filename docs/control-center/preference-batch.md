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
                   --out <home>/preference/profiles/<timestamp>
   ```

   `<learner>` is the command set in Settings, for example `python -m alfa_preference`. The
   learner prints `progress N/M` lines for the progress bar and exits non-zero on failure. It
   writes whatever it learns (embeddings, preference weights, the personal text-to-timbre map)
   into `--out`. `--help` must exit 0; the control center uses it to check the command works.

## Privacy

- Nothing leaves the machine. The control center makes no network calls.
- The index holds paths, sizes and labels, never audio.
- **Forget everything** deletes the index, vocabulary, behaviour log and all profiles. The list
  of chosen folders stays so the user can re-run.

## Proposals for other threads

- **preference/**: provide the `batch` entry point above, or tell the control center thread
  what arguments it wants instead.
- **Plugin**: append implicit feedback (load, play time, keep, save, reuse, small edit, quick
  discard) as JSON lines to `<home>/preference/events/<date>.jsonl`, with `<home>` from
  `alfacc.settings.default_home()`. The event schema belongs to `preference/`.

Until a learner exists, a batch indexes the libraries and stops, and says so in its log.
