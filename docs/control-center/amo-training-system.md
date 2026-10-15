# AMO Centre training system

The Audio Tensor Space page runs offline AMO work through the shared Jobs queue.
Each run has durable progress, logs, output paths and a private training checkpoint.
The Python selected in Settings runs the learner; the Centre process does not import
PyTorch or execute DSP training inside an audio callback. Training starts only when
the user starts a job. Opening the Centre or inspecting a manifest starts no optimizer.

## Model roles and artifacts

| Role | Centre action | Published artifact |
|---|---|---|
| WAV recognition | Train WAV model / `train` | `amo-dual-tensor-v1` weights-only PyTorch package |
| Runtime tensor compiler | Train runtime compiler / `train-runtime` | `amo.runtime.gru/1` native weights, clocks, bounds and golden vectors |
| Local preference | Train preference / `train-preference` | `amo.local-learning/1`, category `user_preference` |
| Measured importance | Train importance / `train-importance` | `amo.local-learning/1`, category `context_importance` |

These models have separate data, architecture and memory allowances. Published
inference artifacts retain compact training summaries and data digests; complete
optimizer history remains in the private checkpoint, avoiding growing inference files. Local learned
state does not automatically become the host's eight preference faders:
the caller owns that encoding. AMO jobs do not install legacy method-vector/ONNX
models. The legacy pages retain their historical workflows. A single Centre instance
serializes queued jobs; independent Centre/CLI processes do not share a process lock.

## Prepare and inspect data

`training/amo/examples/controlled-dataset-plan.json` is an executable, small
controlled-render example. Its two sine primitives, widths, losses, capacities and
synthetic observations are test choices, not product tiers or pretrained knowledge.

Choose a Python containing PyTorch and NumPy in Settings, then:

1. Select the repository, the dataset plan in the manifest field and a **new folder**
   in the output field. Use **Build dataset**.
2. Select the generated `manifest.json`, choose **WAV learning** as the preflight
   category and use **Check manifest**.
3. Select an output model path and use **Train WAV model**.
4. Select that model, the generated manifest and an output report path, then use
   **Evaluate reconstruction**. Inspect Jobs for the measured results.

The producer copies original canonical tensors, applies explicitly bounded unlocked
perturbations and renders declared six-stream performances (plus an optional `wave_feature` stream) through native DSP. Each input may
be a constant feature vector or an explicit `{"frames": [[...], ...]}` trajectory
with one fixed-width row per rendered sample. This preserves authored note releases,
gestures and automation rather than replacing them with held controls. It writes PCM32 recordings without silently clipping or rescaling them.
It publishes a dataset directory only after validation. Failed/interrupted generation
can leave an unpublished `*.staging-*` directory for inspection. Existing dataset
directories are refused; create a new version rather than overwriting source data.

Every WAV manifest retains original and actual rendered-instance labels, example
and perturbation IDs, a normalization policy and train/validation membership.
Preflight checks content hashes, duplicate IDs, conflicting originals, same-original
aliases and identical recordings crossing splits, shape/presence compatibility and
memory estimates. Feature extraction reads one WAV at a time; it retains compact
features and labels rather than all decoded audio. Channel feature widths must match.

Split **by original canonical group**, keeping all pitch/performance/perturbation
variants together. `require_validation: true` refuses a missing held-out partition.
Legacy manifests without splits still run, but their reports have
`has_validation: false` and cannot satisfy held-out acceptance gates.

The included pitch-relative analysis policy is a reference for controlled voiced
recordings. It preserves ordered relative envelope, duration and channel energy;
it is not a universal pitch estimator or proof of invariance for arbitrary polyphonic,
unpitched or colored recordings. `known_calibration` requires an approved correction
adapter and is refused by the manifest workflow rather than silently treated as clean
audio. Normalization declarations remain producer-owned claims requiring evaluation.

## Training objectives and acceptance

The WAV model has independent original-canonical and rendered-instance heads plus
presence heads. Regression ignores padding; training-only statistics normalize the
targets. Validation uses those same statistics and never changes model parameters.
The runtime learner preserves ordered events, resets neural context at sequence
boundaries and supports all seven explicitly encoded input categories. Its value targets are
normalised per coordinate over present training entries (scale at least 1), and the
statistics are folded into the values head after training, so exports stay in raw units.

An optional WAV manifest section adds a **real native DSP reconstruction loss**:

```json
"reconstruction": {
  "weight": 0.1,
  "primary_weight": 2.0,
  "secondary_weight": 1.0,
  "finite_difference_step": 0.0001,
  "parameter_indices": [3],
  "metric": "spectral",
  "bound_weight": 1.0
}
```

The numbers above belong to the included sine fixture. Supply appropriate values
for the authored program. Each example additionally needs its producer `render`
performance and the manifest's explicit `reconstruction_bounds`. Native finite
differences backpropagate the reconstruction distance into the selected continuous,
present, unlocked parameter coordinates. `metric` defaults to `spectral`, the
phase-free `training/amo/perceptual.py` distance (multi-resolution log band energies,
log RMS envelope and the cumulative power distribution over log frequency, plus its
Wasserstein-1 term); `waveform` keeps sample MSE relative to reference power, which
has almost no usable slope for pitch-like parameters. Renders clamp to the bounds;
`bound_weight` (default 1) adds a quadratic barrier, in bound widths, so a value
outside its bound still receives a gradient pointing back in. Original and rendered-instance
references remain separate. Masks, activations, routing and other coordinates retain
their coordinate/presence supervision; this objective does not differentiate discrete
compiler/topology decisions. Each selected coordinate costs two extra native renders
per head and example, so choose the set explicitly. Reference audio has its own
memory estimate. Cancelled differentiation checkpoints the last complete optimizer
step, without publishing a partial update.

Two further optional manifest fields shape the coordinate objective itself:

```json
"parameter_coordinates": ["linear", "log", "linear", "log"],
"perceptual_metric": {"rank": 8, "finite_difference_step": 0.001, "weight": 1.0,
                      "descriptors": "amo.wave-features/1"}
```

`parameter_coordinates` names one kind per padded parameter coordinate. `log`
coordinates (frequencies, times, ratios; they need positive bounds) are learned as
log p, normalised in that space and mapped back by `predict`, so equal steps mean
equal ratios. The reconstruction loss and its difference step use the same space.

`perceptual_metric` replaces "every coordinate counts the same" with a measured
metric. While building a dataset the producer measures J = d phi / d u, phi being the
spectral feature map above, by central differences around both the original and the
instance program of every example (two renders per free coordinate, once, offline;
steps become one-sided at a bound). It stores the rank-`rank` factor
G = J^T J ~ U diag(S^2) U^T as `<example>-canonical.metric.npz` and
`<example>-instance.metric.npz` (`U` [K, r], `S` [r], `free_mask`, `u_ref`, `step`
and, with `descriptors`, the eight wave feature rows `Jd`, `Jd_valid`), listed in the
row's `metric` field and included in the dataset identity. Training adds
`weight` × Delta u^T G Delta u per head to the existing coordinate loss, an exact
quadratic with no renders per step. Coordinates that do not change the sound cost
nothing; the existing masked coordinate loss stays as the identity term. Losses report
`metric` and `validation_metric`. Locked, absent and fixed-bound coordinates are not
free and get no metric weight.

`evaluate-wav` independently compiles **both complete predicted heads** and measures
native reconstruction against the original and instance programs under the same
declared performances. It reports maximum held-out relative RMSE over the native raw
output channels. Output-channel changes, compilation failures and nonfinite output
fail evaluation. This is not a perceptual listening test or the plugin's master mix.

Optional publication gates are producer-defined:

```json
"acceptance": {"validation_total_max": 0.05},
"reconstruction_acceptance": {
  "validation_primary_canonical_relative_rmse_max": 0.1,
  "validation_secondary_rendered_instance_relative_rmse_max": 0.1
}
```

These thresholds are examples, not recommended universal quality limits. Coordinate
gates check held-out metrics before publication. Reconstruction gates serialize a
private candidate, perform native validation and publish only on success. Failure
preserves the previous model and the training checkpoint. Without declared gates,
completion means optimization finished, not that musical product quality was proven.

## Manifest contracts

All new training manifests use `schema: "amo.training-manifest/1"` and an explicit
`category`. Historical WAV/runtime manifests remain readable when those two tags are
absent. Shared `schemas` declare ordered named `axes` and `padded_shape` for each
output tensor; order is part of the checkpoint identity. Ontology and every real
label's axes/presence must match. Capacities and feature encodings are producer choices.

WAV manifests contain `ontology`, `schemas`, `analysis`, `learn` and `examples`.
`learn` requires hidden width, steps, learning rate, primary/secondary/mask weights,
RAM allowance and seed. Each example requires `id`, `canonical_id`, `perturbation_id`,
`canonical_json`, `instance_json`, `normalized_wav`, `normalization` and optional
`split` (default `train`). Referenced files resolve relative to the manifest.

Runtime manifests contain `ontology`, `schemas`, `input_widths`, `hidden_width`,
`seed`, `steps`, `learning_rate`, `mask_weight`, `available_ram_bytes`, `clock`,
`mask_threshold`, `bounds` and `sequences`. New sequence rows have this form:

```json
{"canonical_id": "original-group", "split": "train", "events": [
  {"inputs": {"audio": [], "midi": [200.0, 0.4], "gesture": [],
              "user_preference": [], "latent": [], "context": [], "wave_feature": []},
   "target_json": "label.amo.json"}
]}
```

Declare widths for **all seven** input categories, including explicit zero widths.
`wave_feature` carries the eight wave feature knobs (at most 8 wide, `user_preference` at most 8);
knob values, descriptor measurement, calibration, teacher directions and losses are in
`training/amo/wave_features.py` and `training/METHOD.md`, "Wave feature knobs".
A model with a `midi` width above zero reads fixed MIDI coordinates, so the plugin
loads it only into a sound whose `performance.midi_layout` is `voice_indexed`, and
the width must be at most `voice_capacity` × 10 (ten fields per voice slot, in the
order listed in `src/tensor_plugin/README.md`). Compact rows shift when a note is
released and are absent while nothing is held, so they are refused for models.
Training data for such a model must use the same voice-indexed slots, with silent
slots present as zero rows.
The feature values above are fixture choices. Native producer `render.inputs` uses
`preference` for the corresponding native wrapper argument; runtime event/model
schemas use `user_preference`. Each clock requires explicit `period_samples` and
`phase_samples`. Bounds cover every flattened output coordinate. Legacy event-list
sequences remain training-only and carry no group-validation claim. Optional
`golden_event_count` selects a prefix of the first training sequence (zero disables
golden vectors; omission preserves the full sequence for compatibility). Preflight
also checks the requested export footprint before optimization.

Preference and importance manifests contain `feature_widths`, `hidden_width`, `seed`,
`steps`, `learning_rate`, `available_ram_bytes` and `observations`. Every observation
has a `group_id` and optional `split`. Preference rows supply `first`, `second` and
boolean `preferred_first`. Importance rows supply `features` and measured nonnegative
`impacts`; declare `dimension_count`, and include state, preference, context and
relationship in the feature schema. Examples of both are checked in beside the
controlled dataset plan. Synthetic example targets do not stand in for actual user
ratings or measured musical importance.

`rank-controls` evaluates a context importance model whose dimension count equals the
controls of one X layer, for the features in CONTEXT.json, and writes a copy of the sound
with `control_layers[layer].ranking` (`model` SHA-256, `name`, `importance`). The plugin's
Play view orders that layer by it. The input sound is never overwritten.

## Cancel and resume

Centre training jobs default to `OUTPUT.checkpoint.pt`. Checkpoints contain model,
optimizer, completed-step history, CPU/CUDA random state and a fingerprint of data,
source contents, objective, schema/feature ordering and device. They are private
training state, never inference packages. A new run refuses an existing checkpoint;
select **Resume from checkpoint** to continue or choose new output paths.

Total steps, publication gates and memory allowances can change on resume. Changed data, normalization,
architecture, learning rate, objective or device are refused. Completed CPU mini-batch,
CUDA and native-loss fixture runs verify exact resumed weights/loss history on the
same machine. Cross-platform/driver numerical identity is not asserted.

Cancel requests a cooperative stop and saves a complete optimizer boundary. A hung
process receives a forced stop after 15 seconds; sudden power loss or that forced
stop resumes from the most recent periodic checkpoint. Increase checkpoint frequency
for expensive native objectives. An independent Centre session can signal the same
running AMO job through its recorded cancellation file. Cancellation of another job
does not terminate the active process.

## CLI and hardware

From the repository root, the training interpreter can run the same work directly:

```powershell
python -m training.amo build-dataset training/amo/examples/controlled-dataset-plan.json build/amo-training-dataset
python -m training.amo preflight build/amo-training-dataset/manifest.json --category wav_preset_learning
python -m training.amo train build/amo-training-dataset/manifest.json build/amo-training.pt --checkpoint build/amo-training.checkpoint.pt
python -m training.amo train build/amo-training-dataset/manifest.json build/amo-training.pt --checkpoint build/amo-training.checkpoint.pt --resume
python -m training.amo evaluate-wav build/amo-training.pt build/amo-training-dataset/manifest.json build/amo-reconstruction.json
python -m training.amo train-preference training/amo/examples/preference-manifest.json build/amo-preference.json
python -m training.amo train-importance training/amo/examples/importance-manifest.json build/amo-importance.json
python -m training.amo rank-controls SOUND.json build/amo-importance.json CONTEXT.json RANKED.json --layer 0
```

The Centre CLI records the same durable jobs:

```powershell
alfa-cc --home C:/AMO/Data amo --repository C:/source/code/Alfamedomega train MANIFEST MODEL --job --python C:/Python-Torch/python.exe
alfa-cc --home C:/AMO/Data amo --repository C:/source/code/Alfamedomega train MANIFEST MODEL --job --resume --python C:/Python-Torch/python.exe
alfa-cc amo --repository C:/source/code/Alfamedomega preflight MANIFEST --category runtime
alfa-cc amo --repository C:/source/code/Alfamedomega train-runtime EVENTS_MANIFEST MODEL_JSON --job
alfa-cc amo --repository C:/source/code/Alfamedomega train-preference FEEDBACK_MANIFEST MODEL_JSON
alfa-cc amo --repository C:/source/code/Alfamedomega train-importance IMPACT_MANIFEST MODEL_JSON
```

CPU is the explicit default. Set `device: "cuda:N"` and a positive
`available_vram_bytes` in the relevant training settings to select a specific GPU.
WAV settings live under `learn`; other roles put them at the manifest root.
`batch_size` controls WAV mini-batches (zero means the whole training partition).
WAV validation uses bounded batches with mask-weighted aggregation; GPU preflight
estimates the resident model/batch separately from the CPU dataset and history. Runtime training
retains full ordered sequence recurrence rather than silently truncating temporal
credit assignment. RAM/VRAM allowances are checked against measured available
capacity before model/working-buffer allocation. Estimates include data, history,
Adam and conservative workspace; reserve operating-system, interpreter, CUDA context
and audio-host overhead separately. CUDA is optional, and exported inference models
remain CPU/native artifacts. The CLI/package can also run on a remote training
machine with its own paths; the Centre does not deploy or start remote training.

All verification is local. Relevant coverage includes exact resume, changed-data
refusal, group leakage, atomic publication failure, native-loss gradients, original/
instance reconstruction, actual queued-process cancellation/resume and offscreen GUI
operation. No GitHub CI or production training is required by this workflow.
