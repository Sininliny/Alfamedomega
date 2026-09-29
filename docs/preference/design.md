# Adaptive preference learning: design (draft v1, 2026-09-29)

- **Status:** Draft. Nothing here is confirmed by Sinin yet; numbers are starting points.
- **Spec:** [adaptive-preference-learning.md](adaptive-preference-learning.md)
- **Builds on:** `training/METHOD.md` (draft v1), `docs/ui/ui-design.md`, ADR 0001
- **Reference code:** `preference/` (Python + NumPy, runs anywhere, no torch needed)

## 1. What this component does

The spec names three objectives. The first, inverse synthesis (reference WAV to patch), is
already the training method's job: the AudioEncoder in stage 3 of `METHOD.md`. This component
owns the other two and the loop that uses them:

| Objective | Question | Built here as |
|---|---|---|
| Preference model | What sounds does this user tend to like? | a scorer `s(x, c)` over patches `x` in a context `c` (section 4) |
| Personal text–timbre map | What does this user mean by "airy"? | one direction per word in timbre space (section 5) |
| Guided generation | Which new patch should I offer? | sample from the patch prior, tilted by the two above (section 6) |

**Decision proposed here: preferences never change the synthesis network.** The network's job is
the fixed map from knobs to sound, and METHOD.md depends on that map being identical across tiers
(a patch sounds the same on Lite and Studio) and across machines (a preset sent to a collaborator
sounds the same). Personalisation therefore lives entirely in *which patches get proposed*: a
scorer, a word map and a proposal distribution, all small, all local, all removable without
touching the model. This is the "preference weights" option in the spec's privacy paragraph.
A later personal generator (section 6.4) is also a proposal model, not a change to the synth.

## 2. Representations

**Patch** `x = (m, k)`: the method (one of the nine in METHOD.md, with the hybrid pair) and the 32
knobs in [0, 1]. This is the space the training prior samples from (uniform knobs, method drawn
with `method_probs`), and it is the space generation samples from. Behaviour events store patches,
never audio.

**Timbre descriptor** `a ∈ R^134`, computed from audio: the training script's log-mel front end
(48 kHz, n_fft 2048, hop 512, 64 mel bands, 20 Hz to 16 kHz), reduced to

- the time-mean (64) and time-standard-deviation (64) of the log-mel over active frames, after
  subtracting the clip's mean log energy, so loudness does not count as timbre;
- six scalars: mel centroid, spectral flatness, log attack time, decay slope, spectral flux,
  log active duration.

The descriptor is deliberately **independent of the model and the tier**. A profile survives a
tier switch or a new model version, and the user's own library files (which the model never
produced) land in the same space as rendered patches. It is also not invertible: 134 summary
numbers cannot reconstruct a file, which matters for section 7. The descriptor has a version id
(`logmel-stats-v1`); a learned encoder can replace it later (section 8.2). Because behaviour stores
patches, a descriptor change only needs a re-render of those patches; library entries need a
rescan of the folder, which the user still has.

**Basis.** Descriptors are standardised with the statistics of a **background set**, renders of
N = 512 patches drawn from the training prior (one note at C4, 1 s), and projected onto the top
d = 24 principal components of background plus user data. Call the result `z ∈ R^24`. The
background set is also the "unlabelled" class in section 4.

**Context** `c ∈ R^4` describes where a sound is used, inferred from what the user plays:
bass (mean pitch below 48), mid, high (above 72), and sustained (mean note length above 1 s).
Only these four for v1; host tempo and track name would help but are private-ish and noisy.

## 3. Evidence

### 3.1 Implicit behaviour

The plugin logs events (section 8.1), which are grouped into **exposures**: one exposure is the
time a patch stays loaded in one plugin instance. For each exposure:

| Signal | Symbol | From |
|---|---|---|
| seconds of held notes | `p` | play events |
| saved as a preset | `S` ∈ {0,1} | save event |
| still loaded when the project was saved | `K` ∈ {0,1} | project-save event |
| loaded again in another project | `R` (count) | patch fingerprint seen under a different project hash |
| explicit like / dislike, if the UI offers it | `L` ∈ {−1,0,1} | rate event |
| replaced quickly | `Q` ∈ {0,1} | replaced after less than 20 s dwell with `p` < 5 s, not saved or kept |
| edit distance | `δ` | RMS knob difference between the loaded and the final patch |

Weak reward: `r = pos − neg`, with

```
pos = 0.5 · min(1, log1p(p) / log1p(120)) + 1.0 · S + 0.6 · K + 0.5 · min(R, 2) + 1.5 · max(L, 0)
neg = Q · (1.0 if loaded deliberately else 0.3 when stepping through the browser) + 1.5 · max(−L, 0)
```

The exposure becomes a pointwise example with label `sign(r)` and weight `min(1, |r|)`, dropped
when `|r| < 0.15`. Quick skips while browsing count for less because auditioning is cheap and a
skip often means "not for this track" rather than "bad sound". Weights also decay with a
half-life of 180 days, since taste drifts.

Two situations give **pairwise** evidence, which is less biased than pointwise labels because it
compares sounds under the same context and mood:

- **Edits.** If the user changes the patch by more than δ = 0.05 and the result earns `pos > 0.5`,
  then "edited ≻ loaded". Small edits (δ ≤ 0.05) are treated as the spec says: the loaded sound was
  close, so it keeps its positive label and no pair is formed.
- **Replacements.** If patch A is replaced by B in the same instance, A's exposure is negative and
  B's is positive, then "B ≻ A", weighted by the smaller of the two.

### 3.2 Explicit library

Every audio file in a folder the user picks is a positive example (weight 1.0) with no patch, only
a descriptor, plus the words in its path (section 5). Library sounds are almost all positive,
so on their own they are positive–unlabelled data; section 4 handles that with the background set.

## 4. Preference model

A linear scorer on features that include context interactions:

```
f(x, c) = vec( [z(x); onehot9(m)] ⊗ [1; c] )          (33 × 5 = 165 features)
s(x, c) = θᵀ f(x, c)
```

The first block (`⊗ 1`) is the context-free taste; the rest lets the preference for a timbre
depend on role ("bright is good in a lead, bad in a bass"). This is the "contextual preference
learning" in the spec, in its simplest convex form.

Training minimises a weighted logistic loss over three kinds of rows, with a Gaussian prior:

```
L(θ) = Σ_points  w_i · ℓ(y_i · θᵀ f_i)
     + Σ_pairs   w_ij · ℓ(θᵀ (f_i − f_j))
     + Σ_bg      w_bg · ℓ(−θᵀ f_b)
     + ½ θᵀ Λ θ,          ℓ(u) = log(1 + e^(−u))
```

- **Background rows** are the 512 prior renders labelled "not chosen", with total weight equal to
  the positives' total weight. For library-only users this makes `s` a density-ratio estimate,
  `s ≈ log p_user(z) − log p_prior(z) + const`, which is exactly the tilt section 6 needs to turn
  the prior into the user's distribution.
  Background and library rows carry no method and no context (both one-hot blocks are zero), so
  they teach timbre only; method and context effects come from behaviour, where they are known.
  Otherwise the background would teach "methods the prior samples often are disliked".
- **Prior** Λ = λ I with λ = 1 on context-free weights and 4λ on context interactions, so an empty
  profile scores everything 0 (no preference beyond the training prior) and context effects need
  more evidence.
- The loss is convex; the fit is Newton's method with a backtracking line search, a few
  milliseconds at 165 dimensions. The Hessian at the optimum gives a Laplace posterior
  `θ ~ N(θ̂, H⁻¹)`, used for exploration (Thompson sampling) in section 6.

## 5. Personal text–timbre map

**Words from paths.** Each path relative to the library root is split on separators, camelCase and
letter–digit boundaries, lower-cased, stripped of plural "s", and filtered: numbers, BPM and key
tags (`120bpm`, `c#m`), file-format words and a short list of filler (`sample`, `loop`, `final`,
`copy`, `v2`, …) are dropped. File-name words weigh 1.0 and folder words 0.5, since folders are
coarser. Words in scripts without separators (Chinese, Japanese) stay whole; that is a known gap.

**Word model.** Each word `t` gets a Gaussian in z-space with a shared diagonal covariance Σ pooled
over the library, and a mean shrunk toward the library mean μ₀ with strength κ = 3 files:

```
μ_t = (Σ_i w_i z_i + κ μ₀) / (Σ_i w_i + κ)
LLR_t(z) = log N(z; μ_t, Σ) − log N(z; μ₀, Σ) = d_tᵀ (z − (μ_t + μ₀)/2),    d_t = Σ⁻¹ (μ_t − μ₀)
```

So each word is a direction `d_t`, and "warm" means "what separates this user's warm files from the
rest of their library". A text query is a sum of word scores (a naive-Bayes composition):
"warm airy pad" scores `LLR_warm + LLR_airy + LLR_pad`, and `not`/`no`/`-` negate the next word.
Words the user has never used are reported back as unknown rather than guessed; there is no
remote text model. Words whose directions have cosine above 0.8 are shown as related, which gives
the user a view of their own vocabulary ("warm" ≈ "analog" ≈ "soft").

The UI's Timbre space pad (`ui-design.md`, warm to bright, stable to evolving) can use two of the
user's own words as its axes, projecting patches onto `d_t`.

## 6. Guided generation

### 6.1 Target

```
q(x | c, text) ∝ p_prior(x) · exp( β · [ s(x, c) + γ · Σ_t LLR_t(z(x)) ] )
```

`p_prior` is the training prior, so every candidate is a patch the network was trained on
("technically valid", in the spec's words). β is the user-facing **Familiar ↔ Surprising** control
(β = 0 is the plain random patch generator); γ = 1 weights a text query against general taste.

### 6.2 Sampler (implemented)

1. Draw N = 256 candidates: 75% from the prior, 25% by perturbing patches the user liked
   (Gaussian on knobs, σ = 0.08), which is the UI's "randomize nearby" with taste.
2. Render each as a short preview (one note in the context's register, 1 s), compute `z`, score.
3. Select K = 8 without replacement with probability ∝ exp(β · score) (Gumbel top-k).
   One in five of the K slots (at least one) is always a pure prior draw, so the user keeps
   seeing things outside their current taste and the loop does not collapse onto one sound.
4. Optionally refine with 2–3 rounds of the cross-entropy method: fit a per-method Gaussian over
   knobs to the top 10% and resample. This is the "reinforcement-learning-style optimisation" of
   the spec, black-box, so it works with the real C++ engine.

Exploration uses Thompson sampling: one θ is drawn from the Laplace posterior per batch.

### 6.3 Cost

Previews dominate. At the tier spec's frame costs, a 1 s preview is about 0.1 s of CPU on Lite
and 0.3 s on Standard, so 256 candidates cost roughly 25 s of one core on Lite. Previews therefore
run on a background thread, always on Lite whatever tier is playing, and cache by patch
fingerprint. The scorer itself is free by comparison.

### 6.4 Later: amortised personal generator

Once a profile has a few hundred accepted proposals, the screening can be amortised: a small
generator `g(noise, text words, c) → (m, k)`, trained locally by reward-weighted regression on
the sampler's own selections, so most candidates are good before rendering. It is still a proposal
model (section 1), so tier consistency and preset portability are untouched. Not built in v1.

## 7. Privacy and reproducibility

Everything runs on the user's machine. Nothing is uploaded, and there is no telemetry.

| Stored | Not stored |
|---|---|
| descriptors (134 numbers per file), path words | audio, file names, full paths |
| salted SHA-256 of each relative path, plus size and mtime, for incremental rescans | the library root's path (only a salted hash of it) |
| patches the user loaded, as knobs and method; project and instance ids as salted hashes | project names, host names, track names |
| the fitted basis, θ and Laplace factor, word means | anything from another user |

The salt is random per profile and never leaves it, so hashes cannot be matched across users.
The profile is one folder (`profile.json`, `events.jsonl`, `library.jsonl`) in the app's data
directory. Deleting it resets everything; "forget this folder" deletes the entries of one library
root. Behaviour learning is **off until the user turns it on** (open question 1).

Reproducibility: the profile is a deterministic function of the event log, the library index and
the seed, and generation takes a seed, so the same profile and seed give the same proposals.

## 8. Changes needed from other owners (proposals, not made here)

### 8.1 Plugin shell and UI (`src/plugin`, `src/ui`)

- **Event log.** The audio thread counts note-ons and held samples per instance in atomics; the
  message thread snapshots them once a second and appends JSONL through a background writer.
  Events: `load` (with source: browser, project restore, generate, init), `play` (seconds, notes,
  pitch range, mean note length), `edit` (debounced final patch), `save_preset`, `project_save`,
  `rate`, `close`. Schema: `preference/alfapref/events.py`. Nothing on the audio thread touches
  the file.
- **`project_save` needs care.** Hosts call `getStateInformation` for autosave and undo as well as
  for a real save. Proposal: count it as "kept" only once per exposure, and only after the patch
  has been loaded for 5 minutes of wall time.
- **UI.** A "Learn from how I play" switch (off by default), a library folder picker with "forget
  this folder", a Generate button with the Familiar ↔ Surprising control and a words box, an
  optional like/dislike on the preset bar, and a "why this sound" readout listing its top words.

### 8.2 Training (`training/`)

- **Done by training:** `render_patch(patch, pitch=60, velocity=0.8, seconds=1.0, ckpt=None,
  tier="standard")` in `alfamedomega_train.py` renders one patch with the teacher programs, or with
  a checkpoint via `ckpt` or `ALFA_CKPT`. The CLI uses it with
  `--renderer alfamedomega_train:render_patch` (training/ on `PYTHONPATH`, needs torch), about
  0.03 to 0.4 s per preview on CPU. Preview levels are not normalised (peaks about 0.1 to 0.95
  since training d2b34c5), which the gain-invariant descriptor does not mind. The results in
  section 9 predate d2b34c5, which changed subtractive previews.
- The prior (uniform knobs, `method_probs`) is mirrored in `alfapref/generate.py`. If METHOD.md
  changes the prior, this copy needs the same change; a shared definition would be better.
- **Proposed descriptor v2:** a contrastive timbre encoder trained on teacher renders (the same
  patch at different pitches and velocities as positives), exported as a small ONNX graph. It would
  make "similar" mean "same patch" rather than "same spectrum statistics".

### 8.3 Control center

The control center (`docs/control-center/preference-batch.md` on its branch) runs batches with
`<learner> batch --index <home>/preference/library_index.jsonl --events <home>/preference/events
--out <home>/preference/profiles/<timestamp>`. The learner command is
`python <repo>/preference/learner.py`, which prints `progress N/M`, exits 0 on `--help` and writes
`profile.json`, `library.jsonl` and `summary.json` (counts and the learned vocabulary) to `--out`.
Words are recomputed from each record's `rel` with `alfapref/tokens.py`, so the index's own labels
are for display only. The events folder holds `<date>.jsonl` files in the schema of
`alfapref/events.py`; the plugin should only write it when behaviour learning is on, so the
learner treats a present log as consent.

The index keeps absolute paths and file names, which section 7 keeps out of the profile. That is
acceptable because the index is a local cache the control center rebuilds and "Forget everything"
deletes, but it is the one place names live, and the profiles made from it hold none.

### 8.4 Repo and workflow

- Add `preference/` to the CONTRIBUTING ownership table (owner: preference learning) and a CI job
  running `python -m unittest discover preference/tests` (NumPy only).
- A JUCE-free C++ port, `src/preference` (`Alfa::preference`), for the plugin: descriptor, scorer,
  word map and Newton fit are all small linear algebra. The Python code is the reference, checked
  against golden vectors the same way the DSP primitives are.

## 9. Evaluation

A simulated user (`python -m alfapref simulate`) has a hidden taste, a direction in descriptor
space, and generates behaviour from a noisy choice model (keeps a sound with probability
sigmoid(2u − 1), otherwise skips it within seconds). With the NumPy stand-in renderer:

| Exposures | 10 | 30 | 50 | 100 | 300 |
|---|---|---|---|---|---|
| Scorer AUC against the hidden taste | 0.71–0.73 | 0.76–0.77 | 0.79–0.89 | 0.90 | 0.88 |
| Mean hidden utility of proposals, in prior standard deviations | +0.4 to +1.7 | +1.0 to +2.0 | +1.3 to +2.6 | +3.1 | +2.6 |

Ranges are over three runs: seed 0 to 300 exposures with the default settings, and seeds 0 and 1
to 60 exposures with a 128-patch background (the 100 and 300 columns are the first run only). Learning is fast and then plateaus near AUC 0.88, which is the label
noise of the choice model plus the linear scorer's limit.

With the real teacher programs (`render_patch` from `training/`, 256-patch background, seeds 0
and 1):

| Exposures | 25 | 50 | 75 | 100 |
|---|---|---|---|---|
| Scorer AUC | 0.74–0.77 | 0.70–0.79 | 0.72–0.78 | 0.80–0.81 |
| Proposal utility (prior sd) | −0.7 to +0.7 | +0.2 to +1.1 | +0.4 to +2.5 | +0.9 to +2.2 |

Learning is slower and noisier than with the stand-in, because the real methods spread much more
widely in timbre space, but proposals beat random patches by 50 to 100 exposures. These numbers
say the loop works, not how real users behave. With real users the honest metric
is the fraction of generated patches that are kept or saved, compared with β = 0 patches, which the
exploration slot in section 6.2 provides for free.

## 10. Risks

- **Filter bubble.** Mitigated by the exploration slot, Thompson sampling and the β control.
- **Confounded signals.** Long play can mean "debugging the mix", not "love it". Pointwise weights
  are capped at 1 and pairwise evidence is preferred for that reason.
- **Noisy file names** (`Kick_01_FINAL_real`). Filler words are dropped, rare words are shrunk,
  and the user can see and delete any word.
- **Small data.** Everything shrinks to "no preference" when evidence is thin.

## 11. Open questions for Sinin

1. Should behaviour learning be opt-in (recommended) or on by default?
2. Should words map only to the user's own vocabulary (v1, recommended), or should a bundled
   offline text–audio model (~100 MB or more) let unseen words work too?
3. Should "randomize nearby" in the Perform view use the user's taste by default?
