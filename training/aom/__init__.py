"""Validated AOM training contracts; no learned model or waveform normalizer."""

from .contracts import (
    CanonicalGroup, ModelBudget, ModelCategory, NormalizationRecord, PaddedTensor,
    Representation, Tensor, TensorSchema, WavExample, dual_targets, pad_tensor,
    validate_model_budgets,
)

__all__ = [
    "CanonicalGroup", "ModelBudget", "ModelCategory", "NormalizationRecord",
    "PaddedTensor", "Representation", "Tensor", "TensorSchema", "WavExample",
    "dual_targets", "pad_tensor", "validate_model_budgets",
]
