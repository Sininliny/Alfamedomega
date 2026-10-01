"""Offline dual-supervised tensor prediction, separate from real-time DSP."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math

import torch
from torch import nn

from .contracts import TensorSchema, WavExample, ModelBudget, dual_targets
from .session import TrainingSession, atomic_write, device_for_training, inference_metadata
from .dataset import canonical_digest


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
    batch_size: int = 0
    device: str = "cpu"
    available_vram_bytes: int = 0
    gradient_clip: float = 1.0

    def validate(self):
        if type(self.hidden_width) is not int or type(self.steps) is not int or self.hidden_width <= 0 or self.steps <= 0:
            raise ValueError("Training width and steps must be positive")
        for value in (self.learning_rate, self.primary_weight, self.secondary_weight, self.mask_weight):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Learning rate and both supervision/mask weights must be explicit and positive")
        if type(self.batch_size) is not int or self.batch_size < 0:
            raise ValueError("Batch size must be a nonnegative integer; zero selects the whole dataset")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("Gradient clipping must be positive and finite")
        if self.available_ram_bytes <= 0:
            raise ValueError("Explicit available RAM allowance required")
        if type(self.seed) is not int:
            raise ValueError("Explicit integer random seed required")


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
          schemas: dict[str, TensorSchema], spec: LearnSpec, *, session=None,
          validation=None, reconstruction=None):
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
    estimate = 2048 * spec.steps + 64 * len(examples) * (width + target_width) + 16 * count + 4 * len(examples) * (width + 4 * target_width + 8 * spec.hidden_width)
    if validation:
        estimate += 64 * len(validation[0]) * (width + target_width) + 4 * len(validation[0]) * (width + 4 * target_width + 8 * spec.hidden_width)
    batch = min(spec.batch_size or len(examples), len(examples))
    vram_estimate = 20 * count + 4 * batch * (width + 16 * target_width + 16 * spec.hidden_width)
    device = device_for_training(spec.device, estimate, spec.available_ram_bytes, spec.available_vram_bytes, estimated_vram_bytes=vram_estimate)
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
    if not all(torch.isfinite(array).all() for array in (*arrays, x, canonical, rendered)):
        raise ValueError("Training values overflow float32")
    validation_data = None
    if validation:
        held_examples, held_features = validation
        if not held_examples or len(held_examples) != len(held_features):
            raise ValueError("Nonempty paired validation data required")
        if {e.canonical_id for e in held_examples} & set(known):
            raise ValueError("Canonical identities leak between training and validation")
        originals = {canonical_digest(value) for value in known.values()}
        held_groups = {}
        for example in held_examples:
            if canonical_digest(example.canonical) in originals:
                raise ValueError("Renamed canonical targets leak into validation")
            if example.canonical_id in held_groups and held_groups[example.canonical_id] != example.canonical:
                raise ValueError("Validation canonical identity has conflicting original targets")
            held_groups[example.canonical_id] = example.canonical
        if any(len(row) != width or any(not math.isfinite(v) for v in row) for row in held_features):
            raise ValueError("Invalid validation features")
        held_targets = [dual_targets(e, ontology, schemas) for e in held_examples]
        held_arrays = []
        for head in ("primary_canonical", "secondary_rendered_instance"):
            v = torch.tensor([[v for name in schemas for v in t[head][name].tensor.values] for t in held_targets])
            m = torch.tensor([[float(v) for name in schemas for v in t[head][name].mask] for t in held_targets])
            norm = scales[head]
            held_arrays.extend(((v - torch.tensor(norm["mean"])) / torch.tensor(norm["scale"]), m))
        validation_data = (torch.tensor(held_features), *held_arrays)
        if not all(torch.isfinite(array).all() for array in validation_data):
            raise ValueError("Validation values overflow float32")
    model.to(device)
    def objective(features, primary, primary_mask, secondary, secondary_mask):
        p, r, pm, rm = model(features.to(device))
        pl = masked_loss(p, primary.to(device), primary_mask.to(device))
        rl = masked_loss(r, secondary.to(device), secondary_mask.to(device))
        ml = nn.functional.binary_cross_entropy_with_logits(pm, primary_mask.to(device)) + nn.functional.binary_cross_entropy_with_logits(rm, secondary_mask.to(device))
        return pl, rl, ml, p, r
    contract = {"category": "wav_preset_learning", "ontology": ontology,
                "schema_order": list(schemas),
                "schemas": {k: asdict(v) for k, v in schemas.items()},
                "spec": {k: v for k, v in asdict(spec).items() if k not in ("steps", "available_ram_bytes", "available_vram_bytes")},
                "features": features, "reconstruction": reconstruction.identity if reconstruction else None,
                "labels": [{"id": e.example_id, "group": e.canonical_id,
                            "canonical": asdict(e.canonical), "instance": asdict(e.rendered_instance)} for e in examples],
                "validation": None if not validation else {"features": held_features,
                    "labels": [{"id": e.example_id, "group": e.canonical_id,
                                "canonical": asdict(e.canonical), "instance": asdict(e.rendered_instance)} for e in held_examples]}}
    session = session or TrainingSession(contract)
    # A caller session adds file provenance to the numerical objective identity.
    session.bind(contract)
    start = session.restore(model, optimizer, available_ram_bytes=spec.available_ram_bytes)
    if start > spec.steps:
        raise ValueError("Checkpoint is beyond requested total steps")
    model.train()
    for _ in range(start, spec.steps):
        session.check_cancel(model, optimizer)
        indices = torch.randperm(len(examples))[:batch] if batch < len(examples) else torch.arange(len(examples))
        optimizer.zero_grad(set_to_none=True)
        pl, rl, ml, p, r = objective(x[indices], canonical[indices], canonical_mask[indices], rendered[indices], rendered_mask[indices])
        loss = spec.primary_weight * pl + spec.secondary_weight * rl + spec.mask_weight * ml
        reconstruction_loss = None
        if reconstruction:
            from .session import TrainingCancelled
            try:
                primary_scale, secondary_scale = scales["primary_canonical"], scales["secondary_rendered_instance"]
                reconstruction_loss = reconstruction(p * p.new_tensor(primary_scale["scale"]) + p.new_tensor(primary_scale["mean"]),
                    r * r.new_tensor(secondary_scale["scale"]) + r.new_tensor(secondary_scale["mean"]), indices)
                loss = loss + reconstruction.config["weight"] * reconstruction_loss
            except TrainingCancelled:
                session.save(model, optimizer)
                raise
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss; training halted")
        from .session import TrainingCancelled
        try:
            loss.backward()
        except TrainingCancelled:
            session.save(model, optimizer)
            raise
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        metrics = {"primary_canonical": float(pl.detach()), "secondary_rendered_instance": float(rl.detach()), "mask": float(ml.detach()), "total": float(loss.detach())}
        if reconstruction_loss is not None:
            metrics["reconstruction"] = float(reconstruction_loss.detach())
        if validation_data:
            totals = {"primary": 0., "secondary": 0., "mask": 0., "reconstruction": 0.}
            held_x, held_p, held_pm, held_r, held_rm = validation_data
            with torch.no_grad():
                for offset in range(0, len(held_examples), batch):
                    end = min(offset + batch, len(held_examples))
                    vp, vr, vm, vprimary, vsecondary = objective(*(t[offset:end] for t in validation_data))
                    totals["primary"] += float(vp) * float(held_pm[offset:end].sum())
                    totals["secondary"] += float(vr) * float(held_rm[offset:end].sum())
                    fraction = (end - offset) / len(held_examples)
                    totals["mask"] += float(vm) * fraction
                    if reconstruction:
                        vl = reconstruction(vprimary * vprimary.new_tensor(primary_scale["scale"]) + vprimary.new_tensor(primary_scale["mean"]),
                            vsecondary * vsecondary.new_tensor(secondary_scale["scale"]) + vsecondary.new_tensor(secondary_scale["mean"]),
                            torch.arange(offset, end), validation=True)
                        totals["reconstruction"] += float(vl) * fraction
            vp = totals["primary"] / max(1., float(held_pm.sum()))
            vr = totals["secondary"] / max(1., float(held_rm.sum()))
            vm = totals["mask"]
            metrics.update(validation_primary=vp, validation_secondary=vr, validation_mask=vm,
                           validation_total=spec.primary_weight * vp + spec.secondary_weight * vr + spec.mask_weight * vm)
            if reconstruction:
                metrics["validation_reconstruction"] = totals["reconstruction"]
                metrics["validation_total"] += reconstruction.config["weight"] * totals["reconstruction"]
        session.complete(metrics, model, optimizer, spec.steps)
    session.check_cancel(model, optimizer)
    history = session.history
    model.cpu()
    model.eval()
    return model, {"ontology": ontology, "schemas": {k: asdict(v) for k, v in schemas.items()},
                   "input_width": width, "hidden_width": spec.hidden_width, "target_width": target_width,
                   "estimated_peak_training_bytes": estimate, "estimated_peak_vram_bytes": vram_estimate if device.type == "cuda" else 0, "spec": asdict(spec), "losses": history,
                   "target_normalization": scales,
                   "reconstruction_objective": reconstruction.config if reconstruction else None}


def save(path: Path, model: DualTensorModel, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda stream: torch.save({"format": "aom-dual-tensor-v1", "metadata": inference_metadata(metadata), "state_dict": model.state_dict()}, stream))


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
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Learner package contains nonfinite weights")
    model.eval()
    return model, metadata


def predict(model, metadata, features, *, mask_threshold: float):
    if not 0 < mask_threshold < 1 or len(features) != metadata["input_width"]:
        raise ValueError("Explicit valid mask threshold and compatible features required")
    if any(not math.isfinite(value) for value in features):
        raise ValueError("Features must be finite")
    with torch.no_grad():
        encoded = torch.tensor([features], dtype=torch.float32)
        if not torch.isfinite(encoded).all():
            raise ValueError("Features overflow float32")
        p, r, pm, rm = model(encoded)
        if not all(torch.isfinite(t).all() for t in (p, r, pm, rm)):
            raise ValueError("Learner inference produced nonfinite values")
    result = {"ontology": metadata["ontology"]}
    for name, values, mask_logits in (("primary_canonical", p, pm), ("secondary_rendered_instance", r, rm)):
        normalization = metadata["target_normalization"][name]
        values = values * torch.tensor(normalization["scale"]) + torch.tensor(normalization["mean"])
        if not torch.isfinite(values).all():
            raise ValueError("Learner normalization produced nonfinite values")
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
