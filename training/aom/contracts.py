"""Data boundaries for canonical and rendered-instance WAV supervision.

Ontology extents, tier capacities, normalization algorithms and loss weights are
producer decisions. This module validates those decisions without inventing them.
Flattened tensor values use row-major order, matching the CPU runtime tensors.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import isfinite, prod
from typing import Mapping


def _shape(shape: tuple[int, ...]) -> None:
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Tensor extents must be nonnegative integers")


@dataclass(frozen=True)
class Tensor:
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _shape(self.shape)
        if len(self.axes) != len(self.shape) or len(set(self.axes)) != len(self.axes):
            raise ValueError("Each tensor dimension requires a unique named axis")
        if any(not axis for axis in self.axes):
            raise ValueError("Axis names cannot be empty")
        if len(self.values) != prod(self.shape):
            raise ValueError("Tensor storage does not match its shape")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("Tensor values must be finite")


@dataclass(frozen=True)
class TensorSchema:
    axes: tuple[str, ...]
    padded_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        _shape(self.padded_shape)
        # Reuse axis validation without allocating padded storage.
        if len(self.axes) != len(self.padded_shape) or len(set(self.axes)) != len(self.axes):
            raise ValueError("Schema requires unique named axes")
        if any(not axis for axis in self.axes):
            raise ValueError("Axis names cannot be empty")


@dataclass(frozen=True)
class PaddedTensor:
    tensor: Tensor
    mask: tuple[bool, ...]
    original_shape: tuple[int, ...]


def pad_tensor(source: Tensor, schema: TensorSchema) -> PaddedTensor:
    """Pad every axis, keeping real zeros distinguishable from absent dimensions."""
    if source.axes != schema.axes:
        raise ValueError("Tensor axes differ from the shared ontology")
    if any(size > limit for size, limit in zip(source.shape, schema.padded_shape)):
        raise ValueError("Tensor exceeds shared training shape; truncation is forbidden")
    values = [0.0] * prod(schema.padded_shape)
    mask = [False] * len(values)
    for source_index, coordinates in enumerate(product(*(range(s) for s in source.shape))):
        target_index = 0
        for coordinate, extent in zip(coordinates, schema.padded_shape):
            target_index = target_index * extent + coordinate
        values[target_index] = source.values[source_index]
        mask[target_index] = True
    return PaddedTensor(Tensor(schema.axes, schema.padded_shape, tuple(values)),
                        tuple(mask), source.shape)


@dataclass(frozen=True)
class Representation:
    """Named AOM tensors in an explicitly versioned ontology, without method slots."""
    ontology: str
    tensors: tuple[tuple[str, Tensor], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.tensors]
        if not self.ontology or not names or any(not name for name in names):
            raise ValueError("Representation requires ontology and named tensors")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate representation tensor")

    def pad(self, ontology: str, schemas: Mapping[str, TensorSchema]) -> dict[str, PaddedTensor]:
        if self.ontology != ontology or set(schemas) != {name for name, _ in self.tensors}:
            raise ValueError("Representation differs from shared training ontology")
        return {name: pad_tensor(tensor, schemas[name]) for name, tensor in self.tensors}


@dataclass(frozen=True)
class NormalizationRecord:
    """Auditable producer declaration, not proof of perceptual preservation.

    Do not replace this with automatic pitch shifting, gain normalization or
    denoising: each can alter timbre or temporal behavior. A selected normalizer
    must be evaluated separately before its outputs enter this contract.
    """
    policy_id: str
    removed: frozenset[str]
    preserved: frozenset[str]

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("Normalization policy must be identified")
        if not {"pitch", "loudness", "recording_condition"} <= self.removed:
            raise ValueError("Required nuisance normalization is missing")
        if not {"timbre", "articulation", "temporal_behavior"} <= self.preserved:
            raise ValueError("Required perceptual preservation is missing")
        if self.removed & self.preserved:
            raise ValueError("A property cannot be both removed and preserved")


@dataclass(frozen=True)
class WavExample:
    example_id: str
    canonical_id: str
    canonical: Representation
    rendered_instance: Representation
    normalized_audio: Tensor
    sample_rate: float
    normalization: NormalizationRecord
    perturbation_id: str

    def __post_init__(self) -> None:
        if not all((self.example_id, self.canonical_id, self.perturbation_id)):
            raise ValueError("Example, canonical group and perturbation require identities")
        if self.canonical.ontology != self.rendered_instance.ontology:
            raise ValueError("Both supervision targets must share the ontology")
        if self.normalized_audio.axes != ("sample", "channel"):
            raise ValueError("Normalized audio requires sample and channel axes")
        if any(size == 0 for size in self.normalized_audio.shape):
            raise ValueError("Training audio cannot be empty")
        if not isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive and finite")


@dataclass(frozen=True)
class CanonicalGroup:
    """All pitch/perturbation variants share the ORIGINAL primary target.

    Dataset train/validation splits should be made by canonical_id, keeping
    generated variants together to prevent identity leakage across splits.
    """
    examples: tuple[WavExample, ...]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("Canonical group cannot be empty")
        first = self.examples[0]
        if any(e.canonical_id != first.canonical_id or e.canonical != first.canonical
               for e in self.examples):
            raise ValueError("Variants must share the unchanged original canonical target")
        if len({e.example_id for e in self.examples}) != len(self.examples):
            raise ValueError("Duplicate example identity")


def dual_targets(example: WavExample, ontology: str,
                 schemas: Mapping[str, TensorSchema]) -> dict[str, dict[str, PaddedTensor]]:
    """Return both labeled targets, without assuming loss weights or a model."""
    return {
        "primary_canonical": example.canonical.pad(ontology, schemas),
        "secondary_rendered_instance": example.rendered_instance.pad(ontology, schemas),
    }


class ModelCategory(Enum):
    RUNTIME = "runtime"
    WAV_PRESET_LEARNING = "wav_preset_learning"
    USER_PREFERENCE = "user_preference"


@dataclass(frozen=True)
class ModelBudget:
    """Independent producer estimates and hardware allowances, in bytes.

    Estimates must include peak working memory, not just model weights. This
    validation cannot guarantee host safety if estimates are inaccurate.
    """
    estimated_peak_ram_bytes: int
    available_ram_bytes: int
    estimated_peak_vram_bytes: int
    available_vram_bytes: int

    def __post_init__(self) -> None:
        counts = (self.estimated_peak_ram_bytes, self.available_ram_bytes,
                  self.estimated_peak_vram_bytes, self.available_vram_bytes)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Memory counts must be nonnegative integers")
        if self.estimated_peak_ram_bytes > self.available_ram_bytes:
            raise ValueError("Model estimate exceeds available RAM allowance")
        if self.estimated_peak_vram_bytes > self.available_vram_bytes:
            raise ValueError("Model estimate exceeds available VRAM allowance")


def validate_model_budgets(budgets: Mapping[ModelCategory, ModelBudget]) -> None:
    if set(budgets) != set(ModelCategory):
        raise ValueError("Each model category requires its own explicit budget")
    if any(not isinstance(budget, ModelBudget) for budget in budgets.values()):
        raise ValueError("Model budget must be validated")
