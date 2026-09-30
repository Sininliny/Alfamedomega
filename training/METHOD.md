# Alfamedomega training method (draft v4, 2026-09-30)

This is the method `alfamedomega_train.py` implements. It builds on the tier spec
(`docs/tiers/model-tiers.md`, v5 per-voice budgets), the capacity plan (Claude Doc "Alfamedomega synthesis capacity
plan"), ADR 0001 (JUCE 8 + ONNX Runtime through anira), and the control-frame contract in
`src/dsp/include/alfa/dsp/ControlFrame.h` (layout version 2, exported as `src/dsp/layout.json`),
which is the source of truth for section 2. `python alfamedomega_train.py layout` checks the
script's offsets and tier limits against `layout.json`. Nothing has been trained at scale yet, so every loss weight, size and schedule is a
starting point to be tuned.

v2 aligned the DSP primitives with ControlFrame.h: one OSC layer, amplitude-only PARTIALS with
the modal integrator moved into the graph, per-primitive mix gains, the hop-ramp convention, no
grain spray, and hybrids merged into one frame. v3 renders additive on PARTIALS as its own family
(Sinin's ruling), adopts the tier spec v5 per-voice budgets (Pro 0.65 M, Studio 2.8 M), and renames
the export tensors to the inference host's contract. v4 moves to layout v2: 2627-float
VoiceFrames with dry gain defaulting to 1, OSC unison reads for subtractive and wavetable, the
output safety stage in the render, and the design for processing slots and modulation
(section 2b), which is not trained yet.

## 1. What the network learns

The network never produces audio samples. Every 4 ms (250 Hz) it produces a **VoiceFrame** of
controls for six DSP primitives that the plugin renders on the audio thread. Training renders the
same primitives in PyTorch, so the loss is computed on audio while the network stays at frame
rate.

```
knobs (32) + method ──► PatchEncoder ──► patch embedding e          (once per patch change)
e + global mod (4) ──► InstanceNet (GRU) ──► context c_t              (once per instance, per frame)
note features (8) + c_t ──► VoiceCore (GRU) ──► h_t                   (per voice, per frame)
h_t ──► method expert (routed MLP + lookup tables) ──► method params
method params ──► builder (fixed math, inside the graph) ──► VoiceFrame fields
VoiceFrame ──► DSP primitives ──► audio                                (per voice, per sample)
```

This is where the tier constraint from the tier spec is met. Only VoiceCore, one routed sub-expert
per layer and the head run per voice per frame, which is the 1.4 to 3% "active" budget. The bulk of
the trunk is the PatchEncoder, which runs only when a patch changes, and the bulk of each expert is
unused sub-experts and lookup tables that are gathered rather than multiplied.

### Note features (per voice, per frame)

`[pitch/127, velocity, gate, log1p(time since note-on)/log1p(4 s), log1p(time since note-off)/log1p(4 s), mod wheel, aftertouch, pitch bend]`.
f0 comes from pitch and bend (±2 semitones) and goes straight to the DSP; the network does not
have to predict it.

## 2. DSP primitives (ControlFrame.h, layout version 2)

The PyTorch render functions in the script are the reference implementation of the header. The
export writes golden vectors (section 7) that the C++ engine's tests reproduce. Method-specific
logic lives in the network graph as fixed math, so C++ implements only these six primitives.

**Timing.** The hop is round(sr/250), which is 192 at 48 kHz. A voice consumes frame j at the
start of hop j. Across hop j, "linear" controls ramp per sample from frame j−1 to frame j
(a = i/hop), and "held" controls use frame j. For hop 0, frame −1 is frame 0. Phases start at 0 on
note-on, and f0 ramps the same way.

| Primitive | Controls per frame | Render |
|---|---|---|
| **OSC** | `osc.re`, `osc.im`: 1024 harmonics; `osc.detune_ratio`, `osc.detune_level`: 7 unison reads | frame j's wave is Σ re cos(2π(k+1)φ) − im sin(2π(k+1)φ), stored as a 2048-point table read with linear interpolation, and hop j crossfades from table j−1 to table j. Table j drops harmonics at or above Nyquist for the f0 at the start of hop j times the largest ratio of any unison read sounding in frame j−1 or j (at least 1). Unison read r (the first oscUnison − 1 of 7 run) has its own phase, starting at (r+1)/8 on note-on and advancing at f0·ratio_r (both linear across the hop, so the frequency is quadratic within a hop), reads the same crossfaded tables, and adds level_r times the result |
| **PARTIALS** | `partials.ratio`, `partials.amp` (M used) | sinusoids at f0·ratio with amplitude amp, both linear. A partial at or above Nyquist is silent |
| **FM** | `fm.ratio`, `fm.level` (N used), `fm.index` (8×8 row-major, j>i) | y_i = sin(θ_i + Σ_{j>i} index[i][j]·y_j), evaluated from the last operator to the first, with output Σ level_i·y_i. There is no feedback |
| **NOISE** | `noise.band`: 64 gains | each hop starts a 2-hop grain: uniform noise in [−1, 1), shaped by frame j's band curve with DC zeroed, then Hann-windowed and overlap-added |
| **GRAIN** | per layer: `grain.source` (2 bank indices as floats), `grain.weight` (2), `grain.position`, `grain.rate`, `grain.gain` | each hop starts a 2-hop Hann grain per layer reading w0·bank[s0] + w1·bank[s1] from position·8192, advancing rate per sample and wrapping. All values are held |
| **SAMPLER** | `sampler.rate` (held), `sampler.start` (note-on frame), `sampler.gain`, `sampler.eq` (64) | reads the sample at (f0/root)·rate from start·length, applies a short-time EQ (2-hop Hann frames, the frame starting at hop j using frame j's curve), then applies gain |

The dry voice is `attack(t) · Σ_p mix[p] · primitive_p`, with `mix` = 6 linear gains in the order
[osc, partials, fm, noise, grain, sampler], and `attack(t) = 1 − exp(−t/attack_tau)`, with
attack_tau taken from the note-on frame. The per-sample attack keeps transients sharper than one
4 ms frame, as the tier spec asks. Layout v2 then gives

    voice out    = pan(dry_gain · dry + Σ_k out_gain_k · slot_k)          (equal power, unity at centre)
    instance out = safety(master_gain · width(dry_gain · Σ voices + Σ_k out_gain_k · slot_k))
    safety(x)    = x for |x| ≤ 2 (+6 dBFS), else sign(x) · (2 + 2·tanh((|x| − 2)/2))   (never past +12 dBFS)

`dry_gain` defaults to 1 in both frames and `osc.detune_ratio` to 1, so every producer starts from
a default-constructed frame (`VF_DEFAULTS` in the script, `voiceframe_defaults` in
`export_info.json`). Until slots are trained (section 2b), the training render is one voice
through an otherwise empty Synth: mono at centre pan, no slots, instance at its defaults, so
`y = safety(dry_gain · dry)`. The teachers stay under full scale, so the safety stage only acts on
a bad prediction, and it bounds the loss there.

Band curve: band b is centred at 20·(nyquist/20)^(b/63) Hz, linear in log-frequency between
centres, and flat below band 0. The DC bin takes band 0's gain in the SAMPLER EQ, so unity
bands are transparent, and is zero in NOISE.

Tier sizes: wavetable harmonics K_h = 32 / 64 / 128 / 256 (subtractive and spectral fill all 1024
analytically), partials 16 / 32 / 48 / 64, FM operators 2 / 4 / 6 / 8, grain
layers 2 / 4 / 6 / 8, and OSC unison 2 / 4 / 6 / 8 reads (1 / 3 / 5 / 7 detuned reads). These
match `TierLimits` and `layout.json`.

## 2b. Processing slots and modulation (design; not trained in this version)

Layout v2 adds up to 4 voice slots and 4 instance slots (2 / 3 / 4 / 4 by tier), each
`[family, out_gain, params[16] in 0..1, input[10]]`, with a lower-triangular routing DAG, and a
modulation block (24 source params, 4 / 6 / 8 / 8 routes). The families and their param tables are
in `layout.json`: 9 voice families (filter to body) and 6 instance-only ones (delay, reverb,
convolution, spectral, pitch shift, spatial). The script writes zeros there today (every slot
Off, no routes), which the DSP renders as the v1 instrument. The plan, for review before code:

1. **Family choice is per patch, not per frame.** A slot head on the patch encoder picks each
   slot's family (softmax over the families allowed in that chain, Off included) and the mod
   routes' source and destination. The host writes them into every frame. A per-frame choice
   would crossfade processors every hop, and it would make the router the hardest part of
   training for no musical gain.
2. **Everything continuous is per frame.** A slot head on the voice hidden state writes out_gain,
   the 16 params (sigmoid), the input weights (masked to the lower triangle and the tier's slot
   count) and the route depths. instance_step writes the InstanceFrame the same way.
3. **Teachers get slot programs.** Each teacher patch draws 0 to the tier's slot count of
   families, with params from the neutral defaults in `layout.json` plus knob-driven offsets, and
   a few classic routings (serial chain, parallel into the dry sum, one send). Losses: cross
   entropy on the family, L1 on normalised params weighted by out_gain, and the audio loss.
4. **Render through the real Synth, gradients through proxies.** Most families are recursive
   (SVF, comb, waveguide, FDN reverb) or nonlinear, and a PyTorch port would drift from the C++.
   The loss uses `y = proxy + stopgrad(synth − proxy)` instead: the forward value comes from the
   C++ Synth (through a small Python binding), and gradients flow through a differentiable proxy
   per family. The proxies are frame-wise magnitude responses for the linear families (filter,
   comb, formant, resonator, body, waveguide) and a memoryless waveshaper for nonlinear.
   mod_delay, phaser and the instance-only families get no audio gradient, only the control loss.
5. **Cost.** Slot and route counts are hard caps from the tier, masked in the heads. On top of
   that, a penalty λ_cost · Σ_slots cost(family) · P(family), with costs from the per-family
   `alfa_dsp_bench` timings, keeps slots Off unless they earn their CPU against the
   2 / 3 / 3 / 3.5 ms budgets.
6. **Modulation** is for what the 250 Hz frames cannot do: LFOs above 125 Hz, envelope segments
   shorter than a frame, and sample-and-hold. The network already writes slow movement directly,
   so routes start empty and are trained from teacher programs that use them (fast vibrato,
   snappy filter envelopes).

This needs a Python binding of `Synth` (asked of the DSP thread) and new export outputs on
patch_encoder, voice_core and instance_step (to be agreed with the inference thread).

## 3. Method experts and their builders

Each expert is two residual **routed MLP layers** (top-1 switch routing over K sub-experts),
an optional **lookup table** with product-key top-2 selection, a linear head and a builder.
Builders are fixed differentiable math, exported inside the ONNX graph, and they emit VoiceFrame
fields directly.

| Expert | Head outputs (method params) | Builder → VoiceFrame | Lookup table |
|---|---|---|---|
| Subtractive | saw/square/triangle mix, pulse width, cutoff, resonance, LP/BP/HP mix, noise level, amp, unison ratios (±1 semitone) and levels | analytic oscillator spectra × 4-pole state-variable filter magnitude at each harmonic → OSC; the same filter on the band centres → NOISE; unison → OSC reads | none |
| Wavetable | table selection (top-2 over the bank), position, brightness tilt, amp, unison ratios and levels | interpolate table frames at the position → OSC; unison → OSC reads | tables of 32 frames × K_h complex harmonics (60% of the expert) |
| Additive | M partial amps (codebook + correction), per-partial detune, stiffness stretch B, amp | partials at k·√(1+Bk²)·(1+detune_k) → PARTIALS. Additive is its own family and never uses OSC | partial-envelope codebook (30%) |
| Spectral | spectral-envelope codebook + correction, harmonic/noise balance | envelope at each harmonic → OSC, the same envelope → NOISE | spectral-frame codebook (40%) |
| FM | ratios (snapped toward integers by a learned blend), levels, modulation indices | → FM | none |
| Physical (modal) | mode-set selection, per-mode decay τ and excitation, breath noise | per-mode integrator `a_t = a_{t−1}·exp(−1/(τ·250)) + exc_t` → `partials.amp`; breath → NOISE. The integrator state is a graph input and output (`modal_state`) | mode-ratio codebook (5%) |
| Granular | per layer: source selection (top-2), position, rate (pitched or free), gain | → GRAIN | grain bank of 8192-sample buffers (50%), shipped as `grain_bank.npy` |
| Sampling | rate fine-tune, start offset, EQ curve (codebook + correction), harmonic residual | → SAMPLER, plus a small OSC residual | EQ codebook (5%) |
| Hybrid | two shifts of h (one per component expert) and a crossfade xf | runs two component experts on the shifted h, then **merges them into one VoiceFrame** | none |

**Hybrid merge** (the host applies the same rule): OSC spectra add, weighted by xf and 1−xf;
NOISE bands add by power; every other primitive belongs to exactly one component and takes that
component's mix scaled by its weight; attack_tau is the smaller of the two. Components are drawn
from {subtractive, wavetable, additive, FM, physical}, except the additive + physical pair (both
use PARTIALS), so two components never share anything but OSC and NOISE. Unison reads act on the
merged table, so one set serves both components: the OSC-weighted mean of their levels, and the
level-weighted mean of their ratios. That is exact when only one component plays OSC, and an
approximation for subtractive + wavetable.

**Unison in the teachers.** Knob 22 sets how many of the tier's detuned reads sound (none below
0.5, all at 1), and knob 23 sets the spread (3 to 50 cents, symmetric around the main read).
Sounding reads have level 0.8, and the amp is divided by √(1 + Σ level²). The network's levels
pass through `relu(· − 1e-4)`, so a read it does not want is exactly 0 and costs the DSP nothing. Ring modulation was dropped because one VoiceFrame cannot express it.

Expert sizes follow the capacity shares: expert budget = 0.70 × tier total × share, where the
share formula is copied from `tier_budget.py`. The table takes the fraction shown above, and the
rest sets the number of sub-experts K. If K would fall below 2, which happens for Lite's small
experts, K = 1 and the table absorbs the rest. That is the tier spec's "small control map"
fallback.

Routing uses a switch-style top-1 router with a straight-through gate (output × router
probability), a load-balance loss `K · Σ_k f_k · P_k`, and a router z-loss. In ONNX the chosen
sub-expert's weights are gathered per voice (`Gather` + `MatMul`), so compute per voice is one
sub-expert whatever K is.

`python alfamedomega_train.py report` prints the resulting sizes. All four tiers land within 10% of
the tier spec:

| Tier | Total params (target) | Active per voice (target) | Instance per frame (target) |
|---|---|---|---|
| Lite | 3.93 M (4 M) | 0.121 M (0.12 M) | 0.09 M (0.10 M) |
| Standard | 15.2 M (16 M) | 0.346 M (0.35 M) | 0.52 M (0.50 M) |
| Pro | 63.6 M (64 M) | 0.674 M (0.65 M) | 1.50 M (1.50 M) |
| Studio | 249 M (256 M) | 2.73 M (2.8 M) | 5.80 M (6.0 M) |

## 4. Data

There is no public dataset of "the sound of each synthesis method", so training starts from
**procedural teacher programs**. For each method, a classic parametric synth maps 32 knobs plus a
performance (note, velocity, length, mod wheel, bend, aftertouch) to method params. The params go
through the same builders and primitives, which gives paired (inputs, VoiceFrames, audio) examples
on the fly with no download. Factory content for the teachers (64 wavetables, 16 grain sources,
plate mode sets, 8 samples rendered from other teachers) is generated from fixed seeds, so the
"factory bank" is reproducible.

Real audio comes in at stage 3 (section 5): any folder of labelled notes, with NSynth's
`examples.json` + `audio/*.wav` layout supported directly and resampled to 48 kHz.

Method sampling per batch: each batch is one method, drawn with probability ∝ share^0.5. That
keeps the trunk weighted toward common methods while the 4% experts still train often enough.
Gradients are accumulated over 4 batches, so each optimizer step mixes methods.

## 5. Training stages

| Stage (`--stage`) | Trains | Data | Loss |
|---|---|---|---|
| 1. `pretrain` (Studio, or any tier from scratch) | everything | teacher programs, 5% hybrid | MRSTFT(audio) + λ_ctrl · control distance + 0.01 · balance + 0.001 · z-loss; λ_ctrl decays 1 → 0.1 over the first half |
| 2. `polish` | everything, 0.3 × LR | teachers, 30% hybrid | same, λ_ctrl = 0.1 |
| 3. `finetune` | everything, 0.3 × LR; AudioEncoder comes in | labelled notes: the reference is another note from the same instrument, and the target is this note | MRSTFT only, cut above the source Nyquist; sampling plays the reference note |
| 4. `distill` (Pro, Standard, Lite from Studio) | the student | teacher programs, plus real notes when `--data` is given | control distance to the Studio frames + MRSTFT to the Studio render + 0.25 × the stage-1 loss |
| 5. `export` | none | none | ONNX parity checks; int8 dynamic quantisation for Lite (`--int8`), fp16 for the rest (not yet in the script) |

**MRSTFT** is the multi-resolution spectral loss, over FFT sizes 2048/1024/512/256/128/64 with 75%
overlap: `‖|S|−|Ŝ|‖₁ / ‖|S|‖₁ + mean |log(|S|+1e-5) − log(|Ŝ|+1e-5)|`.

**Control distance** compares VoiceFrames in perceptual units: the mix gains, log-magnitude of the
OSC spectrum weighted by 1/k, log NOISE and EQ bands, log FM ratios and levels, log partial ratios
and amplitudes, log grain and sampler rates and gains, and log attack_tau. Fields whose meaning
depends on a learned table (grain source indices and positions) are left to MRSTFT. Because the
distance is on the frame, it works the same for ground truth and for distillation, even when
student and teacher have different K_h or table sizes.

**Read positions carry no gradient.** Gradients through the read position of a buffer (grain
position and rate, sampler start and rate) come from the local slope of the audio in that buffer,
and they were 100 to 1000× larger than every other gradient in tests. They are stopped, so those
controls learn from the control distance (rates) and from the gains and tables around them.

Why distil method by method: each student expert is trained only on batches of its own method,
so every tier plays the same patch with the same character (the tier spec's promise that switching
from Lite to Studio for a bounce does not change the sound). Matching frames rather than audio
alone is also what keeps the character, since two different control paths can land at the same
spectral loss.

Optimiser: AdamW (β 0.9/0.99, weight decay 0.01 except for tables, norms and biases), 2k-step
warmup, cosine decay, gradient clip 1.0, and bf16 autocast on GPUs that support it (A100, L4; not
T4). The DSP render and the losses stay in fp32. Starting learning rates are 3e-4 for Lite and
Standard, 2e-4 for Pro and 1.5e-4 for Studio.

## 6. Clip setup

48 kHz, hop 192 (250 Hz), 1.0 s clips (250 frames) by default, and batch 16 × 4 accumulation.
Notes start at frame 4, so the pre-roll trains silence. Note length is 30 to 85% of the clip, so
every clip holds a release. The recurrent state is reset per clip. The streaming check (section 7)
confirms that running frame by frame gives the same frames as running the whole clip.

## 7. Export (ADR 0001: ONNX Runtime through anira)

`python alfamedomega_train.py export --ckpt <ckpt> --out Models --version 0.1.0` writes
`Models/<tier>-<version>/` and adds that folder to `Models/manifest.json` with each file's SHA-256,
in the inference host's schema (`src/inference/MODEL_CONTRACT.md`). Tensor names follow that
contract:

| File | When the plugin runs it | Inputs → outputs |
|---|---|---|
| `patch_encoder.onnx` | on a patch or knob change | knobs[1,32], method[1,9], hybrid_parts[1,18] → embedding[1,E] |
| `audio_encoder.onnx` | when a sample or reference sound is loaded | logmel[1,64,frames] → e_audio (added to embedding) |
| `instance_step.onnx` | once per frame | embedding, global_mod[1,4], state[1,Hi] → context[1,Hv], state_out[1,Hi] |
| `voice_core.onnx` | once per frame, all voices batched | features[V,8], context[1,Hv], state[V,Hv] → hidden[V,Hv], state_out |
| `expert_<method>.onnx` | once per frame per active method, for its voices | hidden[V,Hv], f0[V] (+ modal_state for physical) → one output per VoiceFrame field it drives, named as in ControlFrame.h |
| `expert_hybrid.onnx` | for hybrid voices | hidden → hidden_a, hidden_b, xf. The host runs both component experts and merges them (section 3) |
| `grain_bank.npy` | loaded once | the learned grain bank [N, 8192] |
| `export_info.json` | reference | layout version, VoiceFrame offsets, graph inputs, per-expert output widths, check results |
| `dsp_golden.npz` | C++ unit tests | per method, plus `unison` (wavetable with every read on): VoiceFrames [T, 2627] from the note-on frame, f0, and the rendered audio (through the safety stage) |

`hybrid_parts` is component A's one-hot followed by component B's, and it is all zeros unless the
method is hybrid. Four points go beyond the host contract as first published, and are with the
inference thread: `f0` and `modal_state` inputs on the experts, the `hybrid_parts` input,
hybrid chaining on the host, and loading `grain_bank.npy` and `audio_encoder.onnx`.

Expert outputs are tier-sized prefixes of the VoiceFrame arrays, except `fm.index`, which is
N×N row-major and spread to the 8-wide rows by the host. All widths except V are fixed in the graphs. An input a graph does not use is pruned by the exporter (for
example, wavetable ignores f0), and `export_info.json` lists each graph's real inputs.

So a frame costs 2 + (number of active methods) `Run()` calls, with all voices of a method
batched. That answers ADR 0001 open question 3 (separate files per method, not one graph per
patch), and it keeps the tier spec's "one batched Run per frame" assumption, per method.

Checks the export runs, and fails on:
1. **Streaming parity:** the step graphs' torch code, frame by frame, matches the whole-clip
   forward (max error about 1e-6 in the smoke test).
2. **ORT parity:** ONNX Runtime matches torch within 1e-3 with dynamic V (about 1e-6 in the smoke
   test).
3. **Parameter report:** total and active params per voice against the tier spec.

## 8. Compute (estimated, not measured)

A Lite pretrain is the loop to prove first, on a Colab T4. Studio pretraining wants an A100:
its optimizer state is about 4 GB, and memory is dominated by audio-rate rendering at
batch 16 × 48k samples, not by weights. The step counts to start from are 200k for stage 1, 50k
for stage 2, 50k for stage 3 and 100k per student for stage 4. Stage 4 needs a trained Studio
model, so Pro, Standard and Lite do not exist in distilled form until Studio does. They can be
pretrained directly from the teachers in the meantime.

## 9. Open questions and known gaps

- **Slots, modulation and the InstanceFrame are designed (section 2b) but not trained.** The
  exported models leave them at their defaults, and instance_step still returns a context vector,
  not an InstanceFrame.
- **Pan is not trained.** Training is mono at centre pan; pan and width wait for the stereo
  instance render.
- **Capacity split.** Expert shares still use the 9-method prevalence formula. Sinin's 14-family
  split (allocation.json) replaces it along with the slots.
- **Large FM gradients.** Gradient norms on FM batches reach the thousands (phase modulation), so
  clipping is doing real work there. Watch FM loss in the first real run.
- **Sampling scope is still open** in the capacity plan. The method assumes user audio only:
  the sampling expert learns to transpose and re-voice the user's sample, trained on
  other-pitch/velocity pairs of the same instrument. A bundled library would add a large sample
  table and change the shares.
- **Teacher programs define the first sound.** Stage 1 teaches the network classic behaviour of
  each method, which is what the knobs mean. What the network adds beyond a classic synth (real
  instrument dynamics, a shared patch space for morphing) comes from stage 3 data, and how good
  that gets depends on the real-audio set, which isn't chosen yet.
- **Physical modelling is modal only**, with no waveguide feedback. FM has no operator feedback.
  Both would need per-sample recurrence in training.
- **Frame-rate modulation:** modulation faster than 125 Hz is smoothed, apart from the attack
  envelope, which is per sample.
