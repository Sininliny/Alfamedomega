# Alfamedomega interface and AOM Centre design

- **Status:** Implemented in the tensor product editor (`src/tensor_plugin`) and AOM Centre (`apps/control-center`)
- **Date:** 2026-10-01
- **Supersedes:** the 2026-09-29 method-first draft (Perform / Signal flow / Sound design / Customize,
  Musician / Student / Developer levels, 32 method knobs). It is in git history.
- **Requirements:** [mission](../audio-tensor-space/mission.md), "User interaction and migration".
  Sinin's UI feedback of 2026-09-30: no Musician / Student / Developer switch, no "explain each step"
  panel, simple, useful and powerful.

## Principles

1. **One instrument, three depths, no audience modes.** Play, Structure and Tensor are views of the
   same executable sound. Depth is reached by opening a view, not by declaring who you are.
2. **Show relationships, not just numbers.** The direction's central idea is that one source affects
   many destinations through matrices. Every view that edits a matrix also draws it.
3. **Nothing invented.** Controls stay `X[layer,index]`; only bottom instance parameters carry the
   semantic names in `parameter_metadata`. The UI adds no brightness/warmth macros and does not call
   authored ordering "learned". The influence display is computed from the authored matrices.
4. **Viewing never changes sound.** Selecting instances, layers, cells or tensors is inspection.
   Edits happen only through a value entry, a knob, a lock or an explicit Apply.
5. **What changes the sound's running context lives in the header.** Input mode (Instrument / FX /
   Hybrid) and tier were hidden in Settings; they are now always visible.

## Plugin layout (1280 × 800, minimum 1040 × 700)

| Area | Contents |
|---|---|
| Header | Brand; Play / Structure / Tensor; input mode Instrument · FX · Hybrid; tier; Bypass; Panic; Settings |
| Sound bar | Previous / next sound, current name (opens the library), Save as, Open, A / B (Shift-click captures), Capture, Undo, Redo |
| Workspace | The selected view |
| Footer | Input, Mix, Output; a live flow line `input → X controls → instances → output` with counts for the current sound; meters |
| Keyboard | Optional, toggled in Settings |

### Play

- **Controls:** the selected layer's X knobs, eight per page, with readable values (at most four
  significant decimals) and a pager whose arrows point the right way.
- **Influence:** touching a knob focuses it. A diverging bar list shows the bottom parameters that
  control reaches, ranked by the composed linear weight `M_k ⋯ M_1 M_0` (clamps between layers are
  not modelled; the caption says so). Locked parameters are marked because locks override mappings.
- **XY:** two explicit X axes, as before.
- **Output:** waveform and spectrum.

### Structure

- **Instances:** activation per instance as a bar.
- **Parameters:** the selected instance's parameters. Time and frequency sliders are logarithmic,
  so short attacks and low frequencies are reachable (the minimum stays the authored minimum).
- **Relationships:** routing and modulation entries that target the selected instance, each with a
  diverging weight bar and an editable coefficient. "Add relationship" opens the form only when
  asked, so it no longer overlaps the list.

### Tensor

- **Control matrix:** a heat map of the selected layer's matrix, sources as columns and destinations as
  rows (named parameters on the last layer). Click a cell to select it and type its coefficient.
- **Expert:** the tensor value inspector, Apply / Export JSON, runtime model import, composition and
  the compiled inspection report. JSON is the last resort, not the main surface.

### Settings

Theme, accent, knob drag, keyboard, preset library folder, tier profiles and precision notes, laid
out in two columns without overlapping text. Settings never change the sound.

## AOM Centre

The window is named **AOM Centre**. The Audio Tensor Space work comes first; the four legacy pages
(models, method training, preferences, plugin assimilation) are grouped under a **Legacy** heading at
the bottom of the navigation and stay fully working.

The old Audio Tensor Space page was eight unlabelled path fields over eight buttons, without Browse
buttons or the app's styling, so any button could be pressed with the wrong fields filled. It is now
one page per task, each asking only for what that task needs:

| Page | Task | Inputs |
|---|---|---|
| Import sound | WAV → editable tensor preset | WAV model, recording, tensor template, parameter bounds, output; recovery target, presence threshold, sample rate |
| Datasets | Build a controlled dataset, then check a manifest | Dataset plan, new output folder; manifest, model category |
| Train | One of four separate model roles | Role, manifest, output model, checkpoint, resume |
| Evaluate | Native reconstruction report | Model, manifest, report path |

The repository comes from Settings (falling back to the source tree). Results chain forward: a
built dataset fills the manifest fields, a trained WAV model fills Evaluate and Import. Every page
shows its latest job's status, progress and log tail and links to Jobs.

## Open questions for Sinin

1. Should users be able to give X controls their own names (stored in the sound, never auto-generated)?
2. Should the Play view show a learned ranking of controls once a trained importance model exists?
