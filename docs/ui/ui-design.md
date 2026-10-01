# Alfamedomega UI design

> Legacy UI design. The [Audio Tensor Space direction](../audio-tensor-space/mission.md)
> supersedes method-first selection and the fixed 32-knob model as target requirements.
> The replacement needs semantic interaction, advanced tensor inspection/locks/overrides
> and developer visualization; detailed interaction design remains open.

- **Status:** Draft
- **Date:** 2026-09-29
- **Mockup:** https://claude.ai/artifact/EEkyiciMVE9cVzsWzwaHGH (seven boards: Perform, Signal flow, Sound design, Customize, Learning settings, Generate, Control states)
- **Revised:** 2026-09-30, method-first: Sinin replaced the nine-engine blend with one method (or a trained hybrid) per sound; implemented in PR #20.
- **Depends on:** [ADR 0001](../decisions/0001-plugin-and-inference-stack.md) (JUCE 8), `docs/tiers/tier_budget.py` (tier numbers shown in the UI are provisional and come from there)

## The problem

Alfamedomega puts a neural model that has learned nine synthesis methods behind one instrument. Each sound is one method, or a trained hybrid of two, played through 32 knobs whose meaning depends on the method. The UI has to make the common case (pick a sound, shape it with a few knobs, play) obvious to someone who has never read the manual, while still reaching every parameter for sound designers, and letting each person arrange the interface their own way.

## Principles

1. **Two depths, one instrument.** A **Perform** view for playing and broad shaping, a **Sound design** view for everything, and a **Signal flow** view that shows how the current sound is being made. All three edit the same patch; nothing exists in one view that the other breaks.
2. **Every value is readable without touching it.** Each control shows its name and its value in real units (kHz, ms, dB, st), not an unlabelled arc. Hovering adds the exact value; dragging shows it live.
3. **Colour means one thing.** Amber is a value you set. Blue is modulation. Red-orange is a warning. Nothing else is coloured. The pair is chosen to stay distinct for common colour-vision deficiencies and differs in lightness as well as hue.
4. **The model is visible, not magic.** The UI shows which method the sound uses, how much model capacity that method has, what each of its knobs does, the current model size and its CPU and latency cost.
5. **Customization never leaks into sound.** Layout, theme, size and panel choices are saved per user and per instance, separately from presets. Loading a preset never rearranges the interface.

## Views

### Perform (default)

| Area | Contents | Why |
|---|---|---|
| Header | View switch, preset browser (prev/next, category, "made on" tier), model-size switch with params, CPU and latency | Everything that changes *which* sound or *how much* it costs lives in one bar |
| Method | The nine choices in the model's order (subtractive, wavetable, additive, spectral, FM, physical model, granular, sampling, hybrid), each with its share of model capacity and a dot on the one playing. **Hybrid** opens A and B pickers for the two components and warns when the pair is not one the model was trained on | A sound is one method or one trained hybrid, which is what the legacy model takes as input (`training/LEGACY_METHOD.md`, `src/inference/MODEL_CONTRACT.md`). Engines are never layered independently |
| Knobs | The 32 model knobs, paged, showing only the knobs the current method uses, each named and given units for that method (for example Filter cutoff in Hz for subtractive, Op5 ratio for FM). The amp envelope and unison knobs are shared by every method. Knobs a larger model is needed for carry a PRO+ or STUDIO badge | Names come from the method's teacher program, so a knob means the same thing the model was trained on |
| Timbre space | XY pad that moves the current method's two most characteristic knobs, two snapshots, "randomize nearby" | A fast way to explore a method without reading every knob |
| Voice strip | The shared amp envelope knobs, voices, glide, bend, per-voice model activity, output level and meter | Always-needed performance basics |
| Keyboard | On-screen keys with played notes highlighted | Auditioning without a controller |

### Signal flow

Added after feedback on 2026-09-29: key parameters should be visualised, the current synthesis workflow should be clear to someone with a computer science background, and people should choose which parts of it are shown.

- **The pipeline:** seven steps left to right: Notes in, Knobs, Neural model, Method, Sound engines, Envelope + FX, Output. Each step is a card with a live mini-visualisation (piano roll, the 32 knob values, trunk and experts sized by capacity with the current method's expert lit, the chosen method or the hybrid A/B crossfade, waveform, envelope, output spectrum) and a one-line summary of its current state.
- **Inspector:** selecting a step shows a large version of its visualisation, its key parameters as value bars, and a plain-language note explaining what the step does.
- **Detail levels:** *Musician* shows the visualisation and parameters. *Student* adds what data goes in and comes out of each step. *Developer* adds which thread it runs on, its rate (the model's frame rate from the model manifest, for example 250 frames/s, and 48 kHz for the engines) and its cost budget, taken from `tier_budget.py`.
- **Explain each step** turns the notes on or off.
- **Choose what is shown:** each step can be hidden from the display with a chip in the toolbar; the sound still passes through it, and a line says how many steps are hidden. A compact version of the strip can also be pinned to the Perform view (off by default, switched on in Customize).
- **Interpretation:** "a workflow that handles quotation marks" was read as annotated steps, where each step carries a short explanatory note. To be confirmed with Sinin.

### Preference learning (Generate, Learning settings)

Implements the UI items in section 8.1 of [`docs/preference/design.md`](../preference/design.md). Everything below runs locally; the UI says so wherever learning is mentioned.

- **"Learn from how I play" switch** (Customize › Learning). **Off by default.** Turning it on starts the event log; turning it off stops new events but keeps what was learned until the user forgets it. While on, a line shows how many sessions have been noted.
- **Library folders** (Customize › Learning). "Add folder…" opens the host OS folder picker; each folder row shows its indexing state and a **Forget this folder** button, which deletes that root's entries from `library.jsonl`. **Forget everything** deletes the whole profile folder and is styled as the one destructive action; it needs a confirm dialog in the build. **Show profile folder** reveals it in the file manager. A side panel restates the "stored / not stored" table from section 7 of the preference doc in plain words.
- **Generate** (a Generate… button in the Perform view's Timbre space panel opens it; in the build it is a sheet over Perform, not a separate window):
  - a **words box** ("warm, airy") with suggestion chips that append a word;
  - a **Familiar ↔ Surprising** slider. Left is high β (close to the user's taste); the far right is β = 0, the plain random patch generator, and the helper text says taste is ignored there. With learning off, "familiar" means close to the words and library only, and a note says so with a link to Learning settings;
  - a **seed** readout with **Lock seed**, because the same profile, words, setting and seed reproduce the same proposals;
  - **Generate 4 sounds**, then four proposal cards, each with its main methods, a waveform, **Play** (audition without replacing the current patch), **Use this sound** (loads it and logs `load` with source `generate`), like and dislike, and a **Why this sound** readout of its top words. Words the user typed are drawn in the accent colour; words the model added are neutral.
- **Like and dislike on the preset bar** (Perform header). Optional: shown only when "Show like and dislike on the preset bar" is on (off by default in the build; on in the mockup so it can be seen). Pressing one again clears it, and changing preset clears it. Each press logs a `rate` event. To make room, Save moved into the preset menu.
- **Events the UI emits** (schema `preference/alfapref/events.py`): `load` with its source (browser, generate, project restore, init), `edit` (debounced), `save_preset`, `rate`. The UI never writes the log itself; it posts to the background writer the preference doc describes.

### Sound design

- **Methods** list on the left: the nine methods with a one-line summary each; picking one changes the sound's method (hybrid opens its A/B pickers).
- **Method panel** in the centre: a live output display (dashed trace = with modulation), then all 32 knobs for the current method in real units, grouped as the teacher program groups them, and a **Neural** group (latent drift, seed, follow model).
- **Modulation** on the right: a route list (source, target, bipolar amount bar) with **Add route**, and a source editor (LFOs, envelopes, MPE).
- Every parameter's right-click menu offers **Add modulation**, **MIDI learn**, **Lock when browsing presets** and **Type a value**. This is the main customization path for power users, because it works anywhere.

### Customize

Theme (Graphite, Paper, High contrast), accent colour, interface size (75 to 200%), density (comfortable, compact), knob drag gesture (vertical, horizontal, circular), value display (units, percent, on hover), knobs per page, and which panels Perform shows, including an optional Signal flow strip. A live preview applies each choice as it is made. "Use for new instances" makes the current setup the default; "Export theme file" writes the token file described below.

## Interaction rules

- **Knobs:** drag in the chosen gesture; Shift for fine steps; double-click resets to default; click the value to type one; scroll wheel nudges. All controls are keyboard-focusable, arrow keys step, and focus is drawn as a visible ring.
- **Modulation** is drawn as a blue ring outside the value arc, covering the range the source moves it through.
- **Locked** controls show a lock badge and survive preset changes.
- **Controls a smaller model cannot drive** stay visible, dimmed, with a badge naming the smallest size that supports them (for example "PRO+"). Hiding them would make presets look different on different computers.
- **Changing model size while playing:** the new model loads in the background and crossfades in; held notes finish on the old one. Latency reported to the host changes only when the transport is stopped, because hosts handle a mid-playback latency change badly. On first load a short benchmark marks sizes this computer is unlikely to sustain (red outline); they stay selectable.
- **Preset tier portability:** each preset records the size it was made on ("made on Pro"). Loading it on a smaller size plays the closest approximation and says so, rather than refusing.

## Visual system

| Token | Graphite (default) | Use |
|---|---|---|
| `bg` | `#141518` | Window |
| `panel` | `#1C1E22` | Panel surfaces |
| `line` | `#2A2D33` | Borders and tracks |
| `text` | `#ECE8E1` | Primary text |
| `text2` | `#A7A39B` | Labels and secondary text (about 6.5:1 on `panel`) |
| `accent` | `#E9A23B` | Value arcs, selected state |
| `mod` | `#6AA9E0` | Modulation |
| `warn` | `#E07A5F` | CPU and compatibility warnings |

Type is IBM Plex Sans for labels and IBM Plex Mono for every number, so values line up and do not jitter while changing. Minimum text size is 11 px at 100% scale; touch-sized targets (44 px) for primary controls. The design is drawn at 1280 × 800 and scales uniformly from 75% to 200%.

## Implementation notes (JUCE 8)

- **Rendering:** native JUCE components with a custom `LookAndFeel` that reads a theme token JSON (the table above). Themes become data files, like the model tiers, and users can share them. Uniform scaling via `AffineTransform` on the top-level editor plus `setResizable` with a fixed aspect ratio.
- **Parameters:** every control binds through `AudioProcessorValueTreeState` attachments so host automation, undo and MIDI learn share one path.
- **UI state:** layout, theme and scale live in a separate `ValueTree` saved with the plugin instance and in a per-user settings file, never in the preset.
- **Meters and model activity** are read from lock-free atomics written by the audio thread and repainted on a 30 Hz timer; the UI never waits on the audio or inference threads.
- **Considered and not chosen:** JUCE 8's WebView UI. It would let this mockup's HTML ship almost directly and makes theming trivial, but it adds a WebView2 runtime dependency on Windows, a second rendering stack to debug inside every host, and more friction in Pro Tools (AAX). Worth revisiting if the native path proves slow to iterate on.

## Open questions

1. Which two knobs Timbre space moves for each method is a first guess and should be checked by ear once real models exist.
2. Tier names, CPU and latency figures follow `tier_budget.py` and will change with the tiers thread.
3. Whether the on-screen keyboard supports MPE gestures, or only clicks.
