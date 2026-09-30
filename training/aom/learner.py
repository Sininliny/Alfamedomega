"""CPU dual-supervised tensor prediction, separate from the real-time DSP engine."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

import torch
from torch import nn

from .contracts import TensorSchema, WavExample, ModelBudget, dual_targets


@dataclass(frozen=True)
class LearnSpec:
    hidden_width: int
    steps: int
    learning_rate: float
    primary_weight: float
    secondary_weight: float
    mask_weight: float
    available_ram_bytes: int
    seed: int

    def validate(self):
        if self.hidden_width <= 0 or self.steps <= 0:
            raise ValueError("Training width and steps must be positive")
        for value in (self.learning_rate, self.primary_weight, self.secondary_weight, self.mask_weight):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Learning rate and both supervision/mask weights must be explicit and positive")
        if self.available_ram_bytes <= 0:
            raise ValueError("Explicit available RAM allowance required")


class DualTensorModel(nn.Module):
    def __init__(self, input_width, hidden_width, target_width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_width, hidden_width), nn.Tanh())
        self.canonical = nn.Linear(hidden_width, target_width)
        self.rendered = nn.Linear(hidden_width, target_width)
        self.canonical_mask = nn.Linear(hidden_width, target_width)
        self.rendered_mask = nn.Linear(hidden_width, target_width)

    def forward(self, features):
        hidden = self.encoder(features)
        return self.canonical(hidden), self.rendered(hidden), self.canonical_mask(hidden), self.rendered_mask(hidden)


def masked_loss(prediction, target, mask):
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1)


def train(examples: tuple[WavExample, ...], features: list[list[float]], ontology: str,
          schemas: dict[str, TensorSchema], spec: LearnSpec):
    """Train against ORIGINAL canonical and perturbed rendered labels independently.

    Features must come from an explicitly selected analysis policy. No waveform
    generator is learned. Returns a CPU model and measurable loss history.
    """
    spec.validate()
    if not examples or len(examples) != len(features) or not features[0]:
        raise ValueError("Nonempty paired examples and features required")
    width = len(features[0])
    if any(len(row) != width or any(not math.isfinite(v) for v in row) for row in features):
        raise ValueError("Features require fixed width and finite values")
    target_width = sum(math.prod(schema.padded_shape) for schema in schemas.values())
    if not target_width:
        raise ValueError("Training schema cannot be empty")
    # Float32 Adam weights, gradients and moments + target/features + conservative
    # activation workspace. Caller separately reserves Python/torch/host overhead.
    count = width * spec.hidden_width + spec.hidden_width + 4 * (spec.hidden_width + 1) * target_width
    estimate = 16 * count + 4 * len(examples) * (width + 4 * target_width + 8 * spec.hidden_width)
    ModelBudget(estimate, spec.available_ram_bytes, 0, 0)
    # Validate every label and canonical grouping BEFORE allocating the model.
    known = {}
    targets = []
    for example in examples:
        if example.canonical_id in known and known[example.canonical_id] != example.canonical:
            raise ValueError("Canonical identity has conflicting original targets")
        known[example.canonical_id] = example.canonical
        targets.append(dual_targets(example, ontology, schemas))
    arrays = []
    for head in ("primary_canonical", "secondary_rendered_instance"):
        values = [[v for name in schemas for v in target[head][name].tensor.values] for target in targets]
        masks = [[float(m) for name in schemas for m in target[head][name].mask] for target in targets]
        arrays.extend((torch.tensor(values, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)))
    torch.manual_seed(spec.seed)
    model = DualTensorModel(width, spec.hidden_width, target_width)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    x = torch.tensor(features, dtype=torch.float32)
    canonical, canonical_mask, rendered, rendered_mask = arrays
    scales = {}
    normalized = []
    for name, values, mask in (("primary_canonical", canonical, canonical_mask), ("secondary_rendered_instance", rendered, rendered_mask)):
        count_present = mask.sum(dim=0).clamp_min(1)
        mean = (values * mask).sum(dim=0) / count_present
        variance = ((values - mean).square() * mask).sum(dim=0) / count_present
        scale = variance.sqrt().clamp_min(1)
        normalized.append((values - mean) / scale)
        scales[name] = {"mean": mean.tolist(), "scale": scale.tolist()}
    canonical, rendered = normalized
    history = []
    model.train()
    for _ in range(spec.steps):
        optimizer.zero_grad(set_to_none=True)
        p, r, pm, rm = model(x)
        pl = masked_loss(p, canonical, canonical_mask)
        rl = masked_loss(r, rendered, rendered_mask)
        ml = nn.functional.binary_cross_entropy_with_logits(pm, canonical_mask) + nn.functional.binary_cross_entropy_with_logits(rm, rendered_mask)
        loss = spec.primary_weight * pl + spec.secondary_weight * rl + spec.mask_weight * ml
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss; training halted")
        loss.backward()
        optimizer.step()
        history.append({"primary_canonical": float(pl.detach()), "secondary_rendered_instance": float(rl.detach()), "mask": float(ml.detach()), "total": float(loss.detach())})
    model.eval()
    return model, {"ontology": ontology, "schemas": {k: asdict(v) for k, v in schemas.items()},
                   "input_width": width, "hidden_width": spec.hidden_width, "target_width": target_width,
                   "estimated_peak_training_bytes": estimate, "spec": asdict(spec), "losses": history,
                   "target_normalization": scales}


def save(path: Path, model: DualTensorModel, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "aom-dual-tensor-v1", "metadata": metadata, "state_dict": model.state_dict()}, path)


def load(path: Path, *, available_ram_bytes=None):
    from .hardware import detect
    available = detect(gpu_device=None).available_ram_bytes
    if available_ram_bytes is not None:
        available = min(available, available_ram_bytes)
    ModelBudget(Path(path).stat().st_size * 4, available, 0, 0)
    package = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if package.get("format") != "aom-dual-tensor-v1":
        raise ValueError("Unsupported AOM learner package")
    metadata = package["metadata"]
    count = metadata["input_width"] * metadata["hidden_width"] + metadata["hidden_width"] + 4 * (metadata["hidden_width"] + 1) * metadata["target_width"]
    ModelBudget(count * 16, available, 0, 0)
    model = DualTensorModel(metadata["input_width"], metadata["hidden_width"], metadata["target_width"])
    model.load_state_dict(package["state_dict"])
    model.eval()
    return model, metadata


def predict(model, metadata, features, *, mask_threshold: float):
    if not 0 < mask_threshold < 1 or len(features) != metadata["input_width"]:
        raise ValueError("Explicit valid mask threshold and compatible features required")
    if any(not math.isfinite(value) for value in features):
        raise ValueError("Features must be finite")
    with torch.no_grad():
        p, r, pm, rm = model(torch.tensor([features], dtype=torch.float32))
    result = {"ontology": metadata["ontology"]}
    for name, values, mask_logits in (("primary_canonical", p, pm), ("secondary_rendered_instance", r, rm)):
        normalization = metadata["target_normalization"][name]
        values = values * torch.tensor(normalization["scale"]) + torch.tensor(normalization["mean"])
        tensors = {}
        offset = 0
        for key, schema in metadata["schemas"].items():
            count = math.prod(schema["padded_shape"])
            masks = (mask_logits[0, offset:offset + count].sigmoid() >= mask_threshold).tolist()
            tensors[key] = {"axes": schema["axes"], "shape": schema["padded_shape"],
                            "values": values[0, offset:offset + count].tolist(), "mask": masks}
            offset += count
        result[name] = tensors
    return result
