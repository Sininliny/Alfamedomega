# CC-0001: Control center app stack

- **Status:** Proposed
- **Date:** 2026-09-29
- **Scope:** `apps/control-center/` (the standalone control center). The plugin stack is
  [ADR 0001](../decisions/0001-plugin-and-inference-stack.md) and is not affected.

## Context

Alongside the VST3 and AAX plugins, Alfamedomega needs a standalone app for work that is not
musical performance: training models, importing and exporting weights, and running Adaptive
Preference Learning in batch over a user's library. What it touches:

| Job | Lives in | Language |
|---|---|---|
| Training, distillation, ONNX export | `training/alfamedomega_train.py` (METHOD.md) | Python + PyTorch |
| Batch preference learning | `preference/` | Python (expected) |
| Installed weights | the shared `Models/` folder from the packaging plan, read by the plugin | files + `manifest.json` |

None of these run on the audio thread, and none need low latency. They are long jobs,
file management and a few forms.

## Options

1. **JUCE 8 app (C++).** Same look-and-feel code and theme tokens as the plugin, same
   installer, one language for the shipped product. But every real job is Python: the app
   would spawn Python for training and preference learning anyway, and do its own JSON,
   hashing, zip and job bookkeeping in C++, which is slow to write and to change while the
   training contract is still moving. It is also one more AGPLv3 binary while the JUCE
   licence is undecided.
2. **Python core + PySide6 (Qt 6) desktop GUI.** Imports nothing from training at start-up but
   runs it with the same interpreter and conventions the training thread uses, reads the
   export folder directly, and gets hashing, zip and subprocess handling from the standard
   library. Qt is LGPLv3, native on macOS and Windows, and styles well enough to match the
   plugin's Graphite theme. Cost: a second UI toolkit and a larger download (about 60 MB
   for the Qt runtime, not counting PyTorch).
3. **Python core + local web UI** (a small HTTP server and a browser page). The easiest to
   style, but it opens a local port, depends on the user's browser, and is a worse fit for
   long-running local jobs and file pickers.

## Decision

**Option 2.** A Python package, `alfacc`, with three layers:

- **Core** (standard library only): the model store, job runner, training job builder,
  preference library indexer and settings. Everything the app does is here and is tested
  without Qt or PyTorch.
- **CLI** (`alfa-cc`), so every action can be scripted, used over SSH or in CI, and run on a
  machine without a display.
- **GUI** (`alfa-cc-gui`, PySide6), a thin layer over the core. Colours and type come from
  the plugin's Graphite tokens in `docs/ui/ui-design.md`, so the two apps look like one
  product.

PyTorch is never a dependency of the control center. Training runs in a separate Python
interpreter chosen in Settings (a local venv with PyTorch), or on Colab through a
generated notebook, which matches how training is already run.

## Consequences

- The control center ships as its own app, built with PyInstaller or Briefcase and signed
  with the same Apple and Windows identities as the plugin. That adds one step to the
  packaging plan but no new accounts.
- The control center and the plugin share data, not code. The contract between them is
  the `Models/manifest.json` schema (`docs/control-center/model-store.md`) and, later, the
  preference event log. Both need to stay in step with the inference host.
- Nothing in the control center makes a network call. Libraries, indexes and preference
  profiles stay on the user's machine, as the preference spec requires.

## Revisit if

- Model management has to happen inside the plugin too (for example an in-app "download
  tier" button). The manifest format stays the same, so only the UI would move.
- Shipping two toolkits becomes a real support cost. The core and CLI would survive a move
  to JUCE; only the GUI layer would be rewritten.
