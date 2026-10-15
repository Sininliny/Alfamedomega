# AMO tensor training requirements

This is the new training specification supplied by Sinin on 2026-09-30. It is not a claim that
`alfamedomega_train.py` implements it. That script remains the legacy method-based trainer,
using 32-knob inputs, fixed VoiceFrame layouts and 250 Hz control frames; its exports target
`src/inference/MODEL_CONTRACT.md`, not the new recurrent tensor contract.

The retained [legacy method reference](LEGACY_METHOD.md) documents the existing trainer and its
historical export contract.

## Computational construction supervision

Training must learn audio construction from audio plus DSP representation, routing relationships,
parameter behavior and performance context. Neural predictions describe activation, parameters,
routing and modulation; deterministic DSP executes recurrent state updates. Do not train a raw
waveform replacement, method selector or fixed preset-knob generator as the universal compiler.

The runtime representation includes persistent state allocation and multi-timescale scheduling.
Training must distinguish sample-rate memory, audio-rate modulation, control decisions and slow
context. No single 250 Hz hop or fixed primitive-slot count defines the replacement contract.

## Separate model roles

| Model | Purpose | Resource priority |
|---|---|---|
| Runtime | Real-time DSP configuration/control prediction | Low latency, predictable CPU/runtime cost |
| Preset learning | Offline arbitrary WAV-to-editable-AMO inference and structure estimation | Reconstruction quality, deeper analysis and search |
| Preference | Local taste adaptation and exploration | Personalization and local learning |

The offline learner may be much larger and slower than the runtime model. Its importer must estimate
available hardware capacity and usable RAM/VRAM before inference/search. Laptop, 8-core desktop,
high-memory workstation and GPU system categories illustrate capability differences; do not turn
these examples into mandatory hardware assumptions. Offline latency is acceptable; host crashes,
VRAM exhaustion and unreasonable allocations are not. No numerical budget or concrete memory
estimator has been selected or implemented for this new model yet.

## WAV learning: dual targets

Define:

```text
C       = canonical normalized AMO representation before parameter perturbation
I_delta = valid perturbed instance of C
W_delta = deterministic DSP render of I_delta at a sampled performance/pitch
W_input = W_delta under sampled recording/loudness conditions
```

Each example retains both C and I_delta. The primary supervised target is C: recover the original
canonical representation **before perturbation**. The secondary supervised target is I_delta:
recover the actual rendered instance for reconstruction and parameter understanding. Training only
against I_delta is explicitly ruled out. A reconstruction target does not replace canonical identity
supervision, and perturbed examples must not be mislabeled as the canonical target.

Generate examples by selecting a known AMO preset, perturbing parameters within valid ranges,
rendering multiple variations, and rendering the same sound at different pitches. Preserve the
canonical identity and perturbation provenance across these pairs. Source construction, routing,
state and performance labels must remain distinguishable from inferred or unknown metadata.

Loss design must retain separate canonical, instance and reconstruction objectives. Their detailed
metrics, relative numerical weights, architecture and optimizer are not prescribed here. Where an
arbitrary recording admits multiple plausible constructions, do not claim a uniquely recovered true
preset without supporting labels or evidence.

## Invariance and preserved behavior

Normalize irrelevant input pitch, loudness and recording-condition variation while preserving
sound timbre, articulation and temporal behavior. Pitch diversity includes octave/tuning changes;
recording diversity must not erase attacks, releases, envelopes or time-dependent sound character.
Do not assume pitch shifting, peak normalization or denoising automatically meets these requirements.

Validation must compare canonical recovery across nuisance variations, instance recovery after
perturbation, and reconstruction/performance behavior. Include cases where articulation or temporal
structure changes while superficial spectral resemblance remains similar. A controlled-render dataset producer and independent native reconstruction checks are
implemented in `training/amo`. The pitch-relative reference feature policy is validated
on small controlled fixtures; general perceptual normalization remains producer-owned
and is not certified for arbitrary recordings.

## Gradients: spectral distance, training coordinates and perceptual metric

Chosen by Sinin on 2026-10-15 from the gradient proposal (steps 1 and 2). Notation:
p are declared parameter values, u = c(p) training coordinates (`linear` keeps p, `log`
uses log p), y = R(p) the native render under the example's declared performance.

Feature map phi(y), in `training/amo/perceptual.py`, per channel: log band energies on
48 log-spaced bands at STFT sizes 256, 1024 and 4096 (weight 0.03), log RMS envelope,
and the smoothed cumulative power distribution F over log frequency in octaves
(Cramer term). Reconstruction distance:

```text
d(y, y*) = ||phi(y) - phi(y*)||^2 + 10 * integral |F_y(l) - F_y*(l)| dl     (W1 in octaves)
```

None of the terms depends on phase. F is mass-conserving and moves continuously with
pitch, so the slope with respect to a pitch-like coordinate keeps its sign when spectral
peaks no longer overlap. Measured: sign descent on log frequency from up to 0.85 octave
away converges within 1% from 30 of 30 starts (110 Hz with 8 partials, a 440 Hz sine,
1 kHz with 3 partials), against 4 of 32 starts for sample MSE in the proposal test.
Heavier band weights reintroduce local minima.

Coordinate objective per head, with Delta u the prediction error in training coordinates:

```text
L = lambda * masked_mse(normalised u)  +  w * Delta u^T U diag(S^2) U^T Delta u
U diag(S^2) U^T ~ J^T J,  J = d phi(R(c^-1(u))) / d u   at the example's own target
```

The second term is the Gauss-Newton approximation of ||phi(R(u + Delta u)) - phi(R(u))||^2.
J is measured once per example and head by the dataset producer (central differences,
2 renders per free coordinate) and stored as a rank-r factor; training costs one small
matrix product. Directions the render does not hear get zero weight, so equivalent
constructions are not averaged into a sound that matches neither; the masked MSE stays
as the identity term required above. The factor is local: far from the target the
optional native reconstruction loss (now spectral, in u, with a bound barrier) supplies
the long-range gradient. Discrete topology, presence and slot assignment are unchanged
by this step; SPSA, slot matching and primitive JVPs remain later proposal steps.

## Shared tiers, padding and distillation

Lite, Standard, Pro and Studio use one ontology and common training representation. Pad unused
capacity and carry explicit presence information. An absent dimension is not a present numeric zero:
it must not become an artificial target or be counted as active capacity solely from storage size.
Shared datasets, compatible inference contracts and knowledge distillation should retain preset
identity across capacity limits.

Each tier budgets internal tensor resolution, active primitives, routing, temporal state, parameter
precision, simultaneous processing and model/runtime cost. It must not remove user controls or define
an incompatible synthesis universe. The current learners train padded value/presence heads, validate shared axes and
resources, and refuse truncation. Numerical tier budgets and compatible trained
perceptual models remain producer decisions; no universal tier dataset or distilled
pretrained model is bundled.

## Learned importance and constrained conversion

Learn importance from current sound state, user preference, musical context, performance style,
preset structure and parameter relationships. Do not hard-code rankings such as cutoff above FM
ratio above grain density, or allocate capacity by predefined synthesis-method prevalence.

Human-defined constraints protect pitch correctness, timing stability, fundamental musical behavior
and user-locked controls. Tier conversion optimizes perceptual similarity and musical identity under
the target budget. Candidate strategies include resolution reduction, merged activations, simplified
routing, reduced temporal detail and approximated interactions. Simple module deletion is not an
acceptable conversion algorithm. Capacity rejection must not trigger silent destructive pruning.

## Wave feature knobs

Decided by Sinin on 2026-10-15. Training produces eight continuous, bipolar **wave feature
knobs**. Each names one measured audio descriptor (`training/amo/wave_features.py`,
`amo.wave-features/1`):

| Knob | Descriptor | Measurement | Unit |
|---|---|---|---|
| 0 | tilt | weighted slope of band power density against log2 frequency | dB/oct |
| 1 | attack | -log10 of the 10 %-90 % envelope rise time per note | -log10 s |
| 2 | sustain | late held-note energy relative to the note peak | dB |
| 3 | tail | time for Schroeder-integrated energy after note-off to fall 30 dB (release, echo, reverb) | log10 s |
| 4 | noisiness | power-weighted 1 - cos of the second phase difference across frames | 0-1 |
| 5 | roughness | band-envelope modulation power in 20-150 Hz over DC | depth^2 |
| 6 | motion | std of detrended band level over the held note | dB |
| 7 | width | side/mid energy | dB |

Descriptors are measured on the output relative to the knob-0 render of the same input, so
instrument, FX and hybrid operation share one definition; segments come from voice_indexed MIDI
or, without MIDI, from the signal envelope. A descriptor that cannot be measured on a sound
(width of mono, tail with no release) is invalid there and its knob inactive. Targets live in
percentile space of a background calibration: `target = clip(base + 0.35 * knob, 0, 1)`.

The descriptor is the **main direction** of the move, not the only change: turning a knob moves
the sound itself through the continuous tensor space, so the other descriptors may follow and the
network decides how the sound gets there. Losses (`knob_losses`): the turned descriptor reaches its
target (Huber), no unturned descriptor moves more than the turned ones (dominance), and turning a
knob never moves its descriptor backwards (monotone). Unturned descriptors have no target.

The local teacher is the cheapest move along the descriptor in the training-gradient metric
`M = diag(lambda) + U S^2 U^T` (offline Jacobians in unconstrained coordinates, shared with the
perceptual metric; descriptors are not part of that metric):

```text
dp_k = M^-1 j_k / (j_k^T M^-1 j_k)          j_k = d descriptor_k / d u
```

It moves descriptor k by one unit at least perceptual cost; locked and absent coordinates are
excluded. `follow_main_direction` integrates it through the space for large moves, and the runtime
compiler trains on `sample_knob_curves` automation with the knob smoothness term in the same
metric. Gradients through the renderer use SPSA with common random numbers, primitive JVPs or a
differentiable proxy; descriptors need not be differentiable for SPSA.

Two outputs exist together: the runtime compiler reads the knobs as its seventh input category
`wave_feature`, and `fallback_matrix` exports the knob-0 linearization as the program's
`wave_features` block for sounds without such a model. The eight wave feature knobs and eight
user preference faders are separate host controls. These are measured-descriptor controls, not
hand-authored semantic macros (see `docs/audio-tensor-space/mission.md`).

## Implementation status

The `src/tensor` runtime executes and validates the four shared tensors. The AMO
Centre training system now supplies separate WAV, recurrent runtime, local-preference
and measured-importance workflows, explicit data provenance/group validation,
RAM/VRAM checks, CPU/CUDA training, cancellation and reproducible checkpoints.
WAV training retains original and instance losses, masked presence supervision and
an optional independent native waveform reconstruction objective for explicitly
selected continuous unlocked parameters. Discrete topology is supervised separately;
native reconstruction acceptance tests both complete predicted heads.

The controlled dataset producer, native GRU export/golden vectors and local queue/GUI
integration are verified with small deterministic fixtures. Publication is atomic and
can be gated by held-out numerical/native reconstruction thresholds. No universal
trained compiler, arbitrary-recording normalization guarantee, perceptually validated
cross-tier model or pretrained musical history is supplied. Shared ontology/padding
and constrained measured-candidate tier conversion are implemented; full learned
cross-tier distillation remains outside these reference fixtures. Legacy training
commands keep their historical outputs and are explicitly identified as legacy.

Operational manifests, examples and limits: [AMO Centre training system](../docs/control-center/amo-training-system.md).
