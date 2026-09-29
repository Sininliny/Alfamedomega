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

## Layout and manifest

The inference host owns this format (`src/inference/MODEL_CONTRACT.md` on the inference
branch), and `training/alfamedomega_train.py export` already writes it. The control center
follows it exactly and only adds keys the host ignores.

```
Models/
  manifest.json
  lite-1.0.0/              # <tier>-<version>
    export_info.json       # the training export's description of the graphs
    patch_encoder.onnx  instance_step.onnx  voice_core.onnx
    expert_<method>.onnx   # eight methods + expert_hybrid
    audio_encoder.onnx     # optional
    grain_bank.npy
    *.int8.onnx            # optional int8 copies (Lite)
```

```json
{
  "schema": 1,
  "default_tier": "standard",
  "models": [
    {
      "tier": "standard",
      "version": "1.0.0",
      "folder": "standard-1.0.0",
      "opset": 17,
      "frame_rate_hz": 250,
      "files": {"voice_core.onnx": "<sha-256 hex>", "...": "..."},

      "sizes": {"voice_core.onnx": 1234567},
      "bytes": 25000000,
      "source": "training export export",
      "installed_at": "2026-09-29T20:00:00+00:00",
      "notes": ""
    }
  ]
}
```

- The first block of keys in each model is the host's contract. `default_tier`, `sizes`,
  `bytes`, `source`, `installed_at` and `notes` are the control center's additions; a manifest
  written by the training export without them is still read correctly.
- `tier` is one of `lite`, `standard`, `pro`, `studio`. `version` is `MAJOR.MINOR.PATCH`. The
  newest version of a tier wins in the host.
- `dsp_golden.npz` (the C++ test vectors) is left out of installs unless asked for.
- A reader must refuse a `schema` higher than it knows.

**Importing.** The control center accepts a Models folder written by the training export
(every model it lists is installed), a single `<tier>-<version>` folder, a zip of either (the
Colab notebook's last cell makes one), or a `.alfamodel` bundle. It never installs in place:
files are copied, re-hashed and published as below.

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
alfamodel.json    # the manifest entry without folder/source/installed_at, plus schema and exported_at
files/<name>      # every file in the entry
```

Import checks every hash before anything is installed, and rejects entry names with path parts.

## Open points

1. The system folders need admin rights to write on macOS and Windows. The installer should
   create `Models/` writable by the user's group, or the control center will ask for elevation.
   Until that is decided, pointing Settings at a user folder works for development.
