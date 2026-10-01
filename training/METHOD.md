# AOM tensor training requirements

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
| Preset learning | Offline arbitrary WAV-to-editable-AOM inference and structure estimation | Reconstruction quality, deeper analysis and search |
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
C       = canonical normalized AOM representation before parameter perturbation
I_delta = valid perturbed instance of C
W_delta = deterministic DSP render of I_delta at a sampled performance/pitch
W_input = W_delta under sampled recording/loudness conditions
```

Each example retains both C and I_delta. The primary supervised target is C: recover the original
canonical representation **before perturbation**. The secondary supervised target is I_delta:
recover the actual rendered instance for reconstruction and parameter understanding. Training only
against I_delta is explicitly ruled out. A reconstruction target does not replace canonical identity
supervision, and perturbed examples must not be mislabeled as the canonical target.

Generate examples by selecting a known AOM preset, perturbing parameters within valid ranges,
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
implemented in `training/aom`. The pitch-relative reference feature policy is validated
on small controlled fixtures; general perceptual normalization remains producer-owned
and is not certified for arbitrary recordings.

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

## Implementation status

The `src/tensor` runtime executes and validates the four shared tensors. The AOM
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

Operational manifests, examples and limits: [AOM Centre training system](../docs/control-center/aom-training-system.md).
