# Alfamedomega training method (draft v1, 2026-09-29)

This is the method `alfamedomega_train.py` implements. It builds on the tier spec
(`docs/tiers/model-tiers.md`, v3), the capacity plan (Claude Doc "Alfamedomega synthesis capacity
plan") and ADR 0001 (JUCE 8 + ONNX Runtime through anira). Nothing here has been trained yet, so
every loss weight, size and schedule is a starting point to be tuned.

## 1. What the network learns

The network never produces audio samples. Every 4 ms (250 Hz) it produces **control frames** for a
small, fixed set of DSP primitives that the plugin renders on the audio thread. Training renders
the same primitives in PyTorch, so the loss is computed on audio while the network stays at
frame rate.

```
knobs (32) + method ──► PatchEncoder ──► patch embedding e          (once per patch change)
e + global mod (4) ──► InstanceNet (GRU) ──► context c_t              (once per instance, per frame)
note features (8) + c_t ──► VoiceCore (GRU) ──► h_t                   (per voice, per frame)
h_t ──► method expert (routed MLP + lookup tables) ──► method params
method params ──► builder (fixed math, inside the graph) ──► primitive controls
primitive controls ──► DSP primitives ──► audio                        (per voice, per sample)
```

This is where the tier constraint from the tier spec is met. Only VoiceCore, one routed sub-expert
per layer and the head run per voice per frame, which is the 1.4 to 3% "active" budget. The bulk of
the trunk is the PatchEncoder, which runs only when a patch changes, and the bulk of each expert is
unused sub-experts and lookup tables that are gathered rather than multiplied.

### Note features (per voice, per frame)

`[pitch/127, velocity, gate, log1p(time since note-on)/log1p(4 s), log1p(time since note-off)/log1p(4 s), mod wheel, aftertouch, pitch bend]`.
f0 comes from pitch and bend (±2 semitones) and goes straight to the DSP; the network does not
have to predict it.

## 2. DSP primitives (the plugin contract)

The PyTorch render functions in the script are the reference implementation. The plugin's C++
DSP must match them, which the export step checks with golden vectors (section 7). Method-specific
logic (a subtractive filter curve, a wavetable morph, FM routing) lives in the network graph as
fixed math, so C++ implements only these six primitives once.

| Primitive | Controls per frame | Render |
|---|---|---|
| **OSC** harmonic oscillator | complex spectrum of harmonics 1..1024 (real and imaginary parts) | each frame's spectrum is turned into a 2048-sample single-cycle wave by inverse FFT, harmonics above Nyquist are zeroed for that frame's f0, and the wave is read with a phase accumulator and crossfaded to the next frame's wave. This is full-band at any pitch, and its cost does not depend on harmonic count. |
| **PARTIALS** inharmonic sinusoids | M frequency ratios, M amplitudes | sinusoid bank, frame values linearly interpolated per sample |
| **FM** operators | N ratios, N output levels, N×N modulation indices (strictly upper-triangular, so there is no feedback) | phase modulation, evaluated from the last operator to the first |
| **NOISE** filtered noise | 64 band gains on a log axis from 20 Hz to Nyquist | windowed white-noise frames (Hann, 50% overlap), multiplied by the interpolated band curve, then overlap-added |
| **GRAIN** grain layers | per layer: source index weights, read position, playback rate, gain | 2-hop Hann grains read from a source table, overlap-added at the hop |
| **SAMPLER** sample player | rate multiplier, start offset, gain, 64 EQ band gains | reads the loaded sample at `f0/root × rate`, then applies a short-time EQ using the NOISE band layout |

Two things apply to every voice:
- **Mix gains** per primitive, per frame.
- **Attack envelope** `1 − exp(−t/τ)`, computed per sample from the note-on sample, with τ taken
  from the note-on frame. This keeps attacks sharper than one 4 ms frame, as the tier spec asks.

Tier-dependent sizes: OSC explicit harmonics K_h = 32 / 64 / 128 / 256 (the builders extrapolate
the rest up to 1024 with a slope and odd/even tail), FM operators 2 / 4 / 6 / 8, partials
16 / 32 / 48 / 64, and grain layers 2 / 4 / 6 / 8.

## 3. Method experts and their builders

Each expert is two residual **routed MLP layers** (top-1 switch routing over K sub-experts),
an optional **lookup table**, a linear head and a builder. Builders are fixed differentiable math,
exported inside the ONNX graph.

| Expert | Head outputs (method params) | Builder → primitives | Lookup table |
|---|---|---|---|
| Subtractive | saw/square/triangle mix, pulse width, osc-2 detune and level, cutoff, resonance, LP/BP/HP mix, noise level, amp | analytic oscillator spectra × 4-pole state-variable filter magnitude at each harmonic → OSC; the same filter applied to noise bands → NOISE | none |
| Wavetable | table selection (top-2 over the table bank), position, brightness tilt, amp | interpolate table frames at the position → OSC | tables of 32 frames × K_h complex harmonics (60% of the expert) |
| Additive | K_h harmonic amps, slope and odd/even tail, amp | → OSC | harmonic-envelope codebook (30%) |
| Spectral | spectral-envelope codebook selection, morph, harmonic/noise balance | envelope sampled at the harmonics → OSC, the same envelope → NOISE | spectral-frame codebook (40%) |
| FM | ratios (snapped toward integers by a learned blend), levels, modulation indices | → FM | none |
| Physical (modal) | mode-set selection, per-mode decay, excitation energy, damping, brightness, breath noise | a leaky integrator per mode `a_t = a_{t−1}·exp(−hop/τ) + excitation_t` → PARTIALS; breath → NOISE | mode-ratio codebook (20%) |
| Granular | per-layer source selection, position, rate, gain, spray | → GRAIN | grain-source bank of 8192-sample buffers (50%) |
| Sampling | rate fine-tune, start offset, EQ curve (codebook plus correction), harmonic residual | → SAMPLER, plus a small OSC residual | EQ-residual codebook (30%) |
| Hybrid | two FiLM transforms of h (one per component expert), crossfade, ring-mod amount | runs two component experts on the transformed h, then mixes them | none |

Expert sizes follow the capacity shares: expert budget = 0.70 × tier total × share, where the
share formula is copied from `tier_budget.py`. Of that budget, the table takes the fraction shown
above, and the rest sets the number of sub-experts K. If K would fall below 2, which happens for
Lite's small experts, K = 1 and the sub-expert width shrinks. That is the tier spec's "small
control map" fallback.

Routing uses a switch-style top-1 router with a straight-through gate (output × router
probability), a load-balance loss `K · Σ_k f_k · P_k`, and a router z-loss. In ONNX the chosen
sub-expert's weights are gathered per voice (`Gather` + `MatMul`), so compute per voice is one
sub-expert whatever K is.

## 4. Data

There is no public dataset of "the sound of each synthesis method", so training starts from
**procedural teacher programs**. For each method, a classic parametric synth maps 32 knobs plus a
performance (note, velocity, length, mod wheel, bend, aftertouch) to method params. The params go
through the same builders and primitives, which gives paired (inputs, primitive controls, audio)
examples on the fly with no download. Factory content for the teachers (wavetables, grain
sources, mode sets) is generated from fixed seeds, so the "factory bank" is reproducible.

Real audio comes in at stage 3 (section 5): any folder of labelled notes, with NSynth's
`examples.json` + `audio/*.wav` layout supported directly.

Method sampling per batch: each batch is one method, drawn with probability ∝ share^0.5. That
keeps the trunk weighted toward common methods while the 4% experts still train often enough.
Gradients are accumulated over 4 batches, so each optimizer step mixes methods.

## 5. Training stages

| Stage | Trains | Data | Loss |
|---|---|---|---|
| 1. Pretrain (Studio, or any tier trained from scratch) | everything | teacher programs | MRSTFT(audio) + λ_ctrl · control distance + 0.01 · balance + 0.001 · z-loss; λ_ctrl decays 1 → 0.1 |
| 2. Hybrid and polish | everything, lower LR | teachers, 30% hybrid patches | same, λ_ctrl = 0.1 |
| 3. Real-audio fine-tune | everything; AudioEncoder added | labelled notes: the reference is another note from the same instrument, and the target is this note | MRSTFT only, masked above the source Nyquist; sampling plays the reference note |
| 4. Distil (Pro, Standard, Lite from Studio) | the student | teacher programs + real audio | control distance to the Studio controls + MRSTFT to the Studio render + 0.25 × the stage-1 ground-truth loss |
| 5. Quantise and export | none | none | ONNX parity checks; int8 dynamic quantisation for Lite, fp16 for the rest |

**MRSTFT** is the multi-resolution spectral loss, over FFT sizes 2048/1024/512/256/128/64 with 75%
overlap: `‖|S|−|Ŝ|‖₁ / ‖|S|‖₁ + ‖log(|S|+1e-5) − log(|Ŝ|+1e-5)‖₁`.

**Control distance** compares primitives in perceptual units: log-magnitude of the OSC spectrum
weighted by 1/k, log NOISE band gains, log FM ratios and levels, log partial frequencies and
amplitudes, and log mix gains. Primitives whose meaning depends on a learned table (GRAIN source,
SAMPLER EQ codebook) are left to MRSTFT. Because the distance is on primitives, it works the same
for ground truth and for distillation, even when student and teacher have different K_h or table
sizes.

Why distil method by method: each student expert is trained only on batches of its own method,
so every tier plays the same patch with the same character (the tier spec's promise that switching
Lite to Studio for a bounce does not change the sound). Matching controls rather than audio alone
is also what keeps the character, since two different control paths can land at the same
spectral loss.

Optimiser: AdamW (β 0.9/0.99, weight decay 0.01 except for tables, norms and biases), 2k-step
warmup, cosine decay, gradient clip 1.0, and bf16 autocast on GPUs that support it. The DSP render
and the losses stay in fp32.

Starting learning rates: 3e-4 for Lite and Standard, 2e-4 for Pro, 1.5e-4 for Studio.

## 6. Clip setup

48 kHz, hop 192 (250 Hz), 1.0 s clips (250 frames) by default, and batch 16. Notes start at frame
4 so the pre-roll trains silence. Note length is 30 to 85% of the clip, so every clip holds a
release. The recurrent state is reset per clip. The streaming export (section 7) checks that
running frame by frame gives the same result as running the whole clip.

## 7. Export (ADR 0001: ONNX Runtime through anira)

The script exports one folder per tier:

| Graph | When the plugin runs it | Inputs → outputs |
|---|---|---|
| `patch_encoder.onnx` | on a patch or knob change | knobs[1,32], method one-hot[1,9] → e[1,E] |
| `instance_step.onnx` | once per frame | e, global mod[1,4], state → c[1,C], state' |
| `voice_core.onnx` | once per frame, all voices batched | note features[V,8], c[1,C], state[V,H] → h[V,H], state' |
| `expert_<method>.onnx` | once per frame per active method, for its voices | h[V,H] → the primitive controls it drives |
| `expert_hybrid.onnx` | for hybrid voices | h → h_a, h_b, mix |

So a frame costs 2 + (number of active methods) `Run()` calls, with all voices of a method
batched. That is a concrete answer to ADR 0001 open question 3 (separate files per method, not
one graph per patch), and it keeps the tier spec's "one batched Run per frame" assumption, per
method. The step graphs are driven from the same weights as training, and the export checks:

1. **Streaming parity:** frame-by-frame torch steps match the full-clip forward.
2. **ORT parity:** ONNX Runtime matches torch within 1e-4, with dynamic V.
3. **Golden DSP vectors:** `dsp_golden.npz` holds controls and rendered audio for each primitive,
   for the C++ engine's unit tests.
4. **Parameter report:** total vs active params per voice against the tier spec, flagged if more
   than 10% off.

## 8. Compute (estimated, not measured)

The first targets are a Lite pretrain to prove the loop on a Colab T4, and a Studio pretrain on
an A100. Studio's optimizer state is about 4 GB. Memory is dominated by audio-rate rendering at
batch 16 × 48k samples, not by weights. The step counts to start from are 200k for stage 1, 50k for
stage 2, 50k for stage 3 and 100k per student for stage 4. Stage 4 needs a trained Studio model,
so Pro, Standard and Lite do not exist until Studio does.

## 9. Open questions and known gaps

- **Sampling scope is still open** in the capacity plan. The method assumes user audio only:
  the sampling expert learns to transpose and re-voice the user's sample, trained on
  other-pitch/velocity pairs of the same instrument. A bundled library would add a large sample
  table and change the shares.
- **Teacher programs define the first sound.** Stage 1 teaches the network classic behaviour of
  each method, which is what the knobs mean. What the network adds beyond a classic synth (real
  instrument dynamics, a shared patch space for morphing) comes from stage 3 data, and how good
  that becomes depends on the real-audio set, which isn't chosen yet.
- **Physical modelling is modal only.** That covers struck and plucked sounds well. Bowed and
  blown sounds get a continuous excitation, but no waveguide feedback. A waveguide primitive can
  be added later, but it is recurrent per sample, so it is slow to train.
- **FM has no operator feedback** in v1, because it would need per-sample recurrence in training.
- **Frame-rate modulation:** modulation faster than 125 Hz is smoothed, apart from the attack
  envelope, which is per sample.
- **The C++ DSP does not exist yet.** The golden vectors define it, and any deviation needs
  either a C++ fix or a matching change here, followed by retraining.
