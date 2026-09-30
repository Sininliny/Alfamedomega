# Model store: the shared Models folder

- **Status:** Proposed (schema 1)
- **Date:** 2026-09-29
- **Writer:** the control center (`apps/control-center`, `alfacc.models`). **Reader:** the plugin's
  inference host (`src/inference`).
- Builds on the packaging plan, section 3, and the training export in `training/METHOD.md`, section 7.

## Location

| OS | Folder |
|---|---|
| macOS | `/Library/Application Support/Alfamedomega/Models/` |
| Windows | `C:\ProgramData\Alfamedomega\Models\` |
| Linux | `$XDG_DATA_HOME/Alfamedomega/Models/` (default `~/.local/share/...`) |

`ALFA_MODELS_DIR` overrides it for both apps (tests, CI, portable installs).

## Layout

```
Models/
  manifest.json            # the index below; the only file the plugin needs to read first
  lite-1.0.0/              # <tier>-<version>, one folder per installed model
    manifest.json          # the training export's own manifest, copied unchanged
    patch_encoder.onnx
    instance_step.onnx
    voice_core.onnx
    expert_<method>.onnx   # eight methods + expert_hybrid
    audio_encoder.onnx     # optional
    grain_sources.npy
    *.int8.onnx            # optional int8 copies (Lite)
```

`dsp_golden.npz` (the C++ test vectors) is left out of installs unless asked for.

## manifest.json, schema 1

```json
{
  "schema": 1,
  "default_tier": "standard",
  "models": [
    {
      "id": "standard-1.0.0",
      "tier": "standard",
      "version": "1.0.0",
      "dir": "standard-1.0.0",
      "files": [{"path": "voice_core.onnx", "sha256": "…", "bytes": 1234567}],
      "bytes": 25000000,
      "sr": 48000, "frame_hz": 250, "hop": 192,
      "source": "training export export",
      "installed_at": "2026-09-29T20:00:00+00:00",
      "notes": ""
    }
  ]
}
```

- `tier` is one of `lite`, `standard`, `pro`, `studio`. `version` is `MAJOR.MINOR.PATCH`.
- Several versions of a tier can be installed at once. The plugin uses the highest version of the
  tier a preset asks for, unless a preset pins an `id`.
- `default_tier` is the size a new plugin instance starts with.
- A reader must refuse a `schema` higher than it knows.

## Rules for writers and readers

- **Writers** stage a model in `.staging-<id>/`, rename it into place, then replace `manifest.json`
  atomically (write `manifest.json.tmp`, fsync, rename). Removing a model unlists it first and
  deletes the folder second. So a reader never sees a listed model with missing files.
- **Readers** ignore folders that start with `.`, load only what `manifest.json` lists, and check
  each file's SHA-256 off the audio thread before creating a session (packaging plan section 3).
  A file that fails the check means the model is skipped and the UI says so.

## Bundles (`.alfamodel`)

Moving a model between machines. A zip with:

```
alfamodel.json    # the manifest entry without dir/source/installed_at, plus schema and exported_at
files/<name>      # every file in the entry
```

Import checks every hash before anything is installed, and rejects entry names with path parts.

## Open points

1. The inference host thread should confirm it reads this schema, or say what it needs changed.
2. The system folders need admin rights to write on macOS and Windows. The installer should
   create `Models/` writable by the user's group, or the control center will ask for elevation.
   Until that is decided, pointing Settings at a user folder works for development.
