# AOM training data contract

This package is the independent tensor training boundary. The existing legacy
trainer remains available; this package does not relabel its method vectors as
AOM tensors, train a model, or select a normalization algorithm.

`WavExample` contains two separate supervision targets:

- `canonical`: the original normalized AOM representation **before parameter
  perturbation**, used as the primary target.
- `rendered_instance`: the perturbed representation that produced the training
  rendering, used as the secondary target for reconstruction and parameter
  understanding.

`dual_targets` returns both targets with explicit names. It does not assign loss
weights. `CanonicalGroup` rejects variants whose original canonical target
changes, even when their canonical identity string stays the same. Split datasets
by canonical identity so pitch and perturbation variants cannot leak between
training and validation sets. Canonical provenance must still be established by
the rendering producer; this validator cannot infer it from a waveform.

`Representation` accepts versioned, arbitrarily named tensors with explicitly
named axes. `TensorSchema` supplies shared padded training extents across tiers.
`pad_tensor` pads each axis without truncating values, and returns a boolean mask
that distinguishes unused dimensions from meaningful zero values. These masks
must be used by the eventual losses and attention/inference pipeline; padding
alone is not masking. Producer policies determine tier execution capacity and
ontology extents. Padding does not change those policies or automatically lower
a configuration to a smaller tier.

`NormalizationRecord` requires identified removal of pitch, loudness and
recording-condition nuisance variation, and preservation of timbre, articulation
and temporal behavior. It records a producer declaration, not a perceptual
guarantee. Normalized audio keeps its sample and channel axes and actual sample
rate. No automatic pitch shifting, loudness scaling, denoising or temporal
resampling is imposed here: the suitable normalization procedure remains an
explicit design decision requiring perceptual evaluation.

`ModelCategory` separates runtime, offline WAV preset learning and user preference
models. Each requires an explicit `ModelBudget`. Budgets reject peak RAM or VRAM
estimates above the supplied hardware allowance. Accurate hardware estimation,
host reserve, concurrency accounting, inference cancellation and enforcement in
the eventual model backend remain necessary; this contract does not claim to
measure available hardware or prevent a backend from allocating beyond its
estimate.

Run the dependency-free verification from the repository root:

```powershell
python -m unittest discover -s training/tests -p 'test_aom*.py' -v
```
