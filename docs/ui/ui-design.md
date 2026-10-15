# Alfamedomega interface and AOM Centre design

- **Status:** Implemented in the tensor product editor (`src/tensor_plugin`) and AOM Centre (`apps/control-center`)
- **Date:** 2026-10-01, Play and Controls redesigned 2026-10-14, visual style and heading 2026-10-15
- **Supersedes:** the 2026-09-29 method-first draft (Perform / Signal flow / Sound design / Customize,
  Musician / Student / Developer levels, 32 method knobs). It is in git history.
- **Requirements:** [mission](../audio-tensor-space/mission.md), "User interaction and migration".
  Sinin's UI feedback of 2026-09-30: no Musician / Student / Developer switch, no "explain each step"
  panel, simple, useful and powerful. Sinin's 2026-10-14 brief: as beautiful as possible, simple by
  default with deep operation on request, Pigments 7 as the reference for *why* to visualise: draw
  individual sound elements so they are easier to shape, and project the control vector space to a
  low-dimensional view. Do not try to draw the whole wiring ("this tangle can never be read
  clearly"). Knobs and faders should look like instruments, not VB6 controls; colours soft and
  readable. Sinin's 2026-10-15 style notes: flat like Massive X and Pigments 7 but not purely flat,
  with subtle raised and recessed edges; restrained but beautiful, tight spacing; visualisations may
  be rich, so the user feels the overall direction the whole sound is changing in, not one filter.
  Mockup v4 was approved ("没毛病").

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
| Header | Brand; Play / Controls / Instances / Tensor as underlined tabs; input mode Instrument · FX · Hybrid as a segmented group; Bypass; Panic; Settings |
| Sound bar | Previous / next sound and the current name (opens the library) in one recessed field; Save as, Open, A / B (Shift-click captures), Capture, Undo, Redo |
| Workspace | The selected view |
| Footer | IN, MIX and OUT faders; status; state, notes, rate and block; IN / OUT meters as thin bars |
| Keyboard | Optional, toggled in Settings; the full MIDI range fills the width |

### Visual language

- **Surfaces.** Panels are raised: a flat fill, a one-pixel top highlight and a soft shadow, 12 px
  radius. Screens (every plot, the sound field, segment groups, fields) are recessed: a darker fill
  with an inner top shadow and a bottom edge light, 8 px radius. Buttons are small raised plates;
  choices inside a group are segments. 10 px gaps everywhere. No metal, no bevels.
- **Tokens.** Graphite: background `#111317`, panel `#1B1E24`, line `#2A2E36`, text `#EBE9E4`,
  secondary `#8B9099`, accent amber `#F2A35E`, blue `#6AA2F5`, warning `#E27B62`. Paper: `#E9E7E2`,
  `#F4F3EF`, `#DDDAD3`, `#23252A`, `#6B6E75`, `#D9822F`, `#3B78D8`, `#C0502F`.
- **Now and heading.** One grammar on every picture: a neutral solid line is the sound now; a dashed
  line is where the current movement takes it; the band between them is amber where the heading is
  higher and blue where it is lower. Module hues appear only as the small dot in a panel's title.

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
- **Sound** (top left). The whole output: its spectrum now (120 log bands, 30 Hz to 16 kHz) and,
  dashed, where its measured change is heading: the difference between a fast and a half-second
  average carried on for as long again. Beside it, the measured change of level, centroid, lows
  (< 250 Hz), mids and highs (> 4 kHz) over that half second. Everything here is measured output,
  never predicted.
- **Heading.** The heading is a step of 0.35 half-ranges along the direction the projected layer's
  controls moved in the last 1.5 s (by hand, automation or the sound space), or, with no recent
  movement, along the strongest direction `v1`. The step is clamped to the control ranges, mapped
  linearly through `J Dx` and clamped to the parameter ranges, starting from the live parameters.
  Element cards draw it as their dashed curve; the **Heading** panel lists, per primitive group, the
  parameter that moves most as a signed fraction of its range; the sound space draws it as a dashed
  amber arrow from the current point.
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
  The amber shading is how much one small step changes the sound at each point of the plane, from
  the current point with control and parameter clamps applied (finite differences of `Dp⁻¹ Δp`), so
  it fades where controls or parameters run out of range.
- **Controls** (bottom): the first page of the selected layer's knobs, with user names and the
  learned order. Under each knob, one small square per primitive group (up to 12) shows what a
  positive turn of that control alone does to the group's most affected parameter, amber up, blue
  down, brighter for more (`Dp⁻¹ J Dx`, normalised over the page).

### Controls

The deep control surface: the layer's knobs on top, then **Influence** (what the focused control
reaches, as before), **Layer state** (every layer's live values as bars, the page outlined; click a
bar to focus that control) and **XY** (two explicit axes). User names, learned order and influence
are unchanged from the 2026-10-01 design.

### Knobs, faders and values

Rotary controls are a recessed well with a raised cap and a 2 px pointer, a thin track and a value
arc that starts from zero on bipolar ranges (amber up, blue down). Linear sliders are a recessed
4 px groove with a neutral run (amber or blue from zero when bipolar) and a small raised cap. Values are quiet monospaced text under the control rather than boxed fields; double-click
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
