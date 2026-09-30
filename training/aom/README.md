# AOM tensor learning and execution

This independent package supplies validated training data, executable reference
learners, model export and native DSP rendering. The legacy trainer remains
available. The reference models are small controlled experiments, not models
pretrained on the history of digital audio.

`WavExample` contains two separate supervision targets:

- `canonical`: the original normalized AOM representation **before parameter
  perturbation**, used as the primary target.
- `rendered_instance`: the perturbed representation that produced the training
  rendering, used as the secondary target for reconstruction and parameter
  understanding.

`dual_targets` returns both targets with explicit names. `LearnSpec` requires
explicit positive weights for both supervision losses and presence prediction.
`CanonicalGroup` rejects variants whose original canonical target
changes, even when their canonical identity string stays the same. Split datasets
by canonical identity so pitch and perturbation variants cannot leak between
training and validation sets. Canonical provenance must still be established by
the rendering producer; this validator cannot infer it from a waveform.

`Representation` accepts versioned, arbitrarily named tensors with explicitly
named axes. `TensorSchema` supplies shared padded training extents across tiers.
`pad_tensor` pads each axis without truncating values, and returns a boolean mask
that distinguishes unused dimensions from meaningful zero values. These masks
are consumed by masked regression losses and trained presence heads; padding
alone is not masking. Producer policies determine tier execution capacity and
ontology extents. Padding does not change those policies or automatically lower
a configuration to a smaller tier.

`NormalizationRecord` requires identified removal of pitch, loudness and
recording-condition nuisance variation, and preservation of timbre, articulation
and temporal behavior. It records a producer declaration, not a perceptual
guarantee. Normalized audio keeps its sample and channel axes and actual sample
rate. `analysis.py` offers an explicit reference feature profile: pitch-relative
spectral magnitudes, relative envelope frames, duration and channel energy. It
approximately removes global pitch and gain for controlled voiced renderings.
The strongest-component estimator is not reliable for every polyphonic or
missing-fundamental sound. Unknown recording coloration cannot be recovered
universally from arbitrary WAV alone. This reference profile requires controlled
renders or an explicit known calibration; perceptual evaluation and a broader
recording-invariant encoder remain outstanding.

`ModelCategory` separates runtime, offline WAV preset learning and user preference
models. Each requires an explicit `ModelBudget`. Budgets reject peak RAM or VRAM
estimates above the supplied hardware allowance. Accurate hardware estimation,
host reserve, concurrency accounting, inference cancellation and enforcement in
the model backend remain necessary. `hardware.py` measures available physical
RAM/CPU capacity and optionally queries an explicitly selected CUDA device;
host/device reserves are explicit policy. WAV decode, aggregate dataset loading,
model loading and training use measured/supplied estimates to reject oversized
work before large allocation. These estimates do not guarantee safety if a
backend allocates beyond its estimate or available memory changes concurrently.

`learner.py` trains a CPU encoder with independent canonical and rendered tensor
heads, presence heads, coordinate scaling and producer masks. It checks a
working-memory estimate before allocating its model, and supports weights-only
package loading. `pipeline.py` reads paired WAV/JSON manifests and exports
predictions into matching compiler templates with explicit bounds, zero absent
coordinates and preserved locks. The native compiler validates final execution
and capacity and allocates state and schedules. Templates currently supply
primitive identities and projection bindings; arbitrary new structures are not
claimed by the reference model.

`runtime_model.py` supplies a separate recurrent runtime tensor predictor trained
on ordered six-category input events. Its neural context is distinct from DSP
state, and no inference rate is assumed. `export_runtime_model` writes exact
PyTorch GRU weights, explicit inference clock and golden differential vectors for
C++ execution. `local_learning.py` supplies a separate local pairwise preference
model and context-dependent measured-importance model. Both have independent
CPU working-memory allowances; preferences and dimension importance are learned
from local labels rather than assumed rankings.

`tiers.py` learns local sensitivity from measured candidate experiments and
searches explicit proposed configurations under compiler-validated capacities.
Pitch and timing constraints, quality tolerances and objective weights are
caller-supplied. Locks and shared tensor coordinates are preserved. No-fit is an
error, never blind pruning. Actual perceptual evaluation remains caller-owned.

Run verification from the repository root (learner tests require PyTorch/NumPy;
the contracts themselves use only the standard library):

```powershell
python -m unittest discover -s training/tests -p 'test_aom*.py' -v
```

After building `alfa_tensor_c`, run the explicit native-rendered fixture:

```powershell
python -m training.aom.smoke build/aom-smoke
python -m training.aom.runtime_smoke build/aom-runtime-smoke
python -m training.aom train build/aom-smoke/manifest.json build/aom-smoke/retrained.pt
python -m training.aom infer build/aom-smoke/dual-model.pt build/aom-smoke/group-1-variant-1.wav build/aom-smoke/prediction.json --mask-threshold 0.5
```

`import-wav` additionally requires a structural template, explicit bounds, head,
mask threshold and sample rate; it writes an editable executable preset only
after native compiler validation. `train-runtime` uses a separate event-sequence
manifest and exports the GRU model directly for C++ inference.

The same workflows are exposed from AOM Centre:

```powershell
alfa-cc aom --repository C:/source/code/Alfamedomega train MANIFEST MODEL
alfa-cc aom --repository C:/source/code/Alfamedomega train-runtime EVENTS_MANIFEST RUNTIME_MODEL_JSON
alfa-cc aom --repository C:/source/code/Alfamedomega import-wav MODEL WAV TEMPLATE BOUNDS OUTPUT --head primary_canonical --mask-threshold 0.5 --sample-rate 48000
```

Those paths are explicit command arguments. AOM commands do not initialize the
legacy model store. Existing top-level `train` and `export` retain their legacy
workflow and compatibility and are labeled accordingly in command help.

Fixture dimensions, weights and capacities are test choices, not product tiers.
The smoke output includes learned compiler JSON, DSP-rendered WAV, model package
and measured losses. `testdata/runtime_gru.json` is a deterministic numerical
parity fixture, not a trained musical model.

`runtime_smoke` reproduces the learned runtime integration from a clean checkout
after the native library is built: it creates explicit controlled targets and an
event manifest, trains/exports a GRU compiler, embeds it in executable JSON, and
compares actual native audio against the same program without its model. It also
requires bitwise DSP/model reset equivalence and verifies rejection of an
insufficient model RAM budget. All inputs and evidence are written to the chosen
output directory. Use `--dll PATH` when the native library is outside standard
build directories.
