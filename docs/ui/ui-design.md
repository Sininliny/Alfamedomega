# Alfamedomega interface and AOM Centre design

- **Status:** Implemented in the tensor product editor (`src/tensor_plugin`) and AOM Centre (`apps/control-center`)
- **Date:** 2026-10-01, Play and Controls redesigned 2026-10-14
- **Supersedes:** the 2026-09-29 method-first draft (Perform / Signal flow / Sound design / Customize,
  Musician / Student / Developer levels, 32 method knobs). It is in git history.
- **Requirements:** [mission](../audio-tensor-space/mission.md), "User interaction and migration".
  Sinin's UI feedback of 2026-09-30: no Musician / Student / Developer switch, no "explain each step"
  panel, simple, useful and powerful. Sinin's 2026-10-14 brief: as beautiful as possible, simple by
  default with deep operation on request, Pigments 7 as the reference for *why* to visualise: draw
  individual sound elements so they are easier to shape, and project the control vector space to a
  low-dimensional view. Do not try to draw the whole wiring ("this tangle can never be read
  clearly"). Knobs and faders should look like instruments, not VB6 controls; colours soft and
  readable.

## Principles

1. **One instrument, several depths, no audience modes.** Play, Controls, Instances and Tensor are views of the
   same executable sound. Depth is reached by opening a view, not by declaring who you are.
2. **Show relationships, not just numbers.** The direction's central idea is that one source affects
   many destinations through matrices. Every view that edits a matrix also draws it.
3. **Nothing invented.** Controls stay `X[layer,index]`; only bottom instance parameters carry the
   semantic names in `parameter_metadata`. The UI adds no brightness/warmth macros and does not call
   authored ordering "learned". The influence display is computed from the authored matrices.
4. **Viewing never changes sound.** Selecting instances, layers, cells or tensors is inspection.
   Edits happen only through a value entry, a knob, a lock or an explicit Apply.
5. **What changes the sound's running context lives in the header.** Input mode (Instrument / FX /
   Hybrid) was hidden in Settings; it is now always visible. There is no tier selector: precision
   and capacity come from the loaded sound (Sinin, 2026-10-01).

## Plugin layout (1280 × 800, minimum 1040 × 700)

| Area | Contents |
|---|---|
| Header | Brand; Play / Controls / Instances / Tensor, each with its own colour dot; input mode Instrument · FX · Hybrid; Bypass; Panic; Settings |
| Sound bar | Previous / next sound, current name (opens the library), Save as, Open, A / B (Shift-click captures), Capture, Undo, Redo |
| Workspace | The selected view |
| Footer | Input, Mix, Output; input mode and "N X in L layers, M instances"; state, notes, rate and block; meters |
| Keyboard | Optional, toggled in Settings; the full MIDI range fills the width |

### Identity colours

Every primitive family and every source kind has one hue, used wherever it appears: the dot on an
element card, the stripe on an instance, the dot on a relationship, the live mark on a parameter.
Families: oscillators pink, excitation magenta, physical green, filters teal, nonlinear red-orange,
spectral blue, pitch/time violet, time cyan, spatial green-teal, granular rose; control primitives
(envelopes) stay neutral. Sources: MIDI cyan, audio green, gesture magenta, context amber,
preference and latent violet. Hues are fixed; saturation and value adapt to the theme. Amber stays
"positive", blue "negative" on every signed bar.

### Play

The default view is about the sound, not the graph.

- **Sound elements.** Each primitive that has a natural picture is drawn the way it sounds, and
  instances of the same primitive share one card (eight voices draw eight curves):

  | Primitive | Drawing | Handles |
  |---|---|---|
  | `control.adsr` | the envelope curve (A, D, S hold, R) | attack, decay + sustain, release |
  | `filter.biquad_lowpass` | RBJ magnitude response on a log frequency axis | cutoff (x), Q (y, dB) |
  | `osc.additive` | live partials (Hz, amplitude) of sounding voices | none (driven by notes) |
  | `time.modulated_delay` | echo train with feedback decay | none |
  | `spatial.stereo_width` | stereo wedge | none |
  | `time.fdn_reverb` | g^n decay to −60 dB | none |

  Curves use the parameters the runtime actually used in the last block (so modulation and
  velocity show), and the authored value where nothing reaches. Dragging a handle edits the
  authored value of every instance in the card at once, as one undoable change. Double-click opens
  the instance in the Instances view. Other primitives have no card; they are reached from Instances.
- **Output:** waveform and spectrum in the same grid.
- **Sound space** (right). The selected X layer projected onto the two directions that change the
  sound most. With `J = M_K ⋯ M_l` the composed authored matrices (parameters × controls; clamps
  between layers are not modelled), `Dp` the parameter range widths and `Dx` the control half ranges,
  the normalised Jacobian is `J̃ = Dp⁻¹ J Dx` (rows without a range or locked are dropped). The two
  axes are the eigenvectors `v1, v2` of `J̃ᵀ J̃` with the largest eigenvalues (cyclic Jacobi), i.e.
  the right singular vectors of `J̃`. A control state `x` is drawn at `(v1·u, v2·u)` with
  `u = Dx⁻¹ (x − mid)`. Each axis says what share of the sound's change it carries (`λᵢ / Σλ`) and
  which parameters it moves most, in their own names. Arrows show where each control alone would
  move the sound. Dragging the dot moves `u` by `V Δp`, the smallest control change with that
  projection, clamped to the ranges; host-automatable controls are written with host gestures,
  deeper ones as one undoable change on release. A short trail and the A / B snapshots are drawn.
- **Controls** (bottom): the first page of the selected layer's knobs, with user names and the
  learned order, as before.

### Controls

The deep control surface: the layer's knobs on top, then **Influence** (what the focused control
reaches, as before), **Layer state** (every layer's live values as bars, the page outlined; click a
bar to focus that control) and **XY** (two explicit axes). User names, learned order and influence
are unchanged from the 2026-10-01 design.

### Knobs, faders and values

Rotary controls are a lit dome with a soft shadow, a thin track and a glowing value arc that starts
from zero on bipolar ranges. Linear sliders are a 4 px track with a filled run from zero and a lit
disc thumb. Values are quiet monospaced text under the control rather than boxed fields; double-click
to type. Parameter rows draw the authored value and, when the runtime is using a different value,
a band from authored to live in the colour of the strongest source reaching it.

### Live telemetry

The processor publishes, once per audio block and without locks, each instance's output peak, the
effective parameters the runtime used (`Runtime::effectiveParameters()`) and every layer's values.
The editor uses them only while blocks keep arriving; otherwise it shows authored values.

### Instances

- **Instances:** one card per instance with its family stripe, title, primitive, activation and a
  live output level.
- **Parameters:** the selected instance's parameters. Time and frequency sliders are logarithmic,
  so short attacks and low frequencies are reachable (the minimum stays the authored minimum).
- **Relationships:** routing and modulation entries that target the selected instance, each with its
  source colour, a diverging weight bar and an editable coefficient. "Add relationship" opens the form only when
  asked, so it no longer overlaps the list.

### Tensor

- **Control matrix:** a heat map of the selected layer's matrix, sources as columns and destinations as
  rows (named parameters on the last layer). Click a cell to select it and type its coefficient.
- **Expert:** the tensor value inspector, Apply / Export JSON, runtime model import, composition and
  the compiled inspection report. JSON is the last resort, not the main surface.

### Settings

Theme, accent, knob drag, keyboard, preset library folder, and the loaded sound's precision and resource notes, laid
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
| Rank controls | Learned control order for one X layer of a sound | Sound, context importance model, context features, layer, output sound |

The repository comes from Settings (falling back to the source tree). Results chain forward: a
built dataset fills the manifest fields, a trained WAV model fills Evaluate and Import. Every page
shows its latest job's status, progress and log tail and links to Jobs.

## Decisions

1. Users can name X controls (Sinin, 2026-10-01: part of "full customisation").
2. The Play view shows a learned control ranking when a trained importance model has produced one
   (Sinin, 2026-10-01). The ranking is computed for an explicit context file; re-ranking live from
   the playing state needs a defined encoding of that state into the model's features, which does
   not exist yet.
3. Visualise sound elements and the control space, not the whole graph (Sinin, 2026-10-14). The
   signal-graph centrepiece first proposed that day was dropped.
