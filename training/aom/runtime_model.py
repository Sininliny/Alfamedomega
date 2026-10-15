"""Separate recurrent runtime compiler model; it predicts tensors, not audio."""
import math
from pathlib import Path
import torch
from torch import nn
from .contracts import ModelBudget, Representation, TensorSchema
from .session import optimize, atomic_json, inference_metadata
from dataclasses import asdict


# wave_feature: the eight wave feature knobs (aom.wave-features/1, see wave_features.py).
INPUT_CATEGORIES = ("audio", "midi", "gesture", "user_preference", "latent", "context", "wave_feature")


class RuntimeTensorModel(nn.Module):
    """Inference is invoked at producer-scheduled events, with no assumed rate.

    Inputs are explicitly encoded feature tensors; raw MIDI units, audio feature
    extraction, semantic controls and preference encodings remain caller-owned.
    Neural hidden state is distinct from persistent deterministic DSP state.
    """
    def __init__(self, input_widths: dict[str, int], hidden_width: int, schemas: dict[str, TensorSchema]):
        super().__init__()
        if set(input_widths) != set(INPUT_CATEGORIES) or any(type(v) is not int or v < 0 for v in input_widths.values()):
            raise ValueError("Explicit widths for all seven input categories required")
        if hidden_width <= 0 or sum(input_widths.values()) <= 0:
            raise ValueError("Positive hidden and aggregate input width required")
        self.input_widths = dict(input_widths)
        self.schemas = dict(schemas)
        self.hidden_width = hidden_width
        output_width = sum(math.prod(s.padded_shape) for s in schemas.values())
        if not output_width:
            raise ValueError("Output tensor schema cannot be empty")
        self.cell = nn.GRUCell(sum(input_widths.values()), hidden_width)
        self.values = nn.Linear(hidden_width, output_width)
        self.presence = nn.Linear(hidden_width, output_width)

    def encode(self, inputs):
        if set(inputs) != set(INPUT_CATEGORIES):
            raise ValueError("All seven input categories require explicit data or empty vectors")
        flat = []
        for category in INPUT_CATEGORIES:
            values = inputs[category]
            if len(values) != self.input_widths[category] or any(not math.isfinite(x) for x in values):
                raise ValueError(f"Invalid {category} feature tensor")
            flat.extend(values)
        encoded = torch.tensor([flat], dtype=torch.float32)
        if not torch.isfinite(encoded).all():
            raise ValueError("Runtime inputs overflow float32")
        return encoded

    def forward(self, encoded, state):
        state = self.cell(encoded, state)
        return self.values(state), self.presence(state), state

    def infer(self, inputs, state, *, mask_threshold: float):
        if not 0 < mask_threshold < 1:
            raise ValueError("Explicit valid presence threshold required")
        if tuple(state.shape) != (1, self.hidden_width) or not torch.isfinite(state).all():
            raise ValueError("Explicit finite neural hidden state required")
        with torch.no_grad():
            values, masks, next_state = self(self.encode(inputs), state)
        if not all(torch.isfinite(t).all() for t in (values, masks, next_state)):
            raise ValueError("Runtime inference produced nonfinite values")
        result, offset = {}, 0
        for name, schema in self.schemas.items():
            count = math.prod(schema.padded_shape)
            present = (masks[0, offset:offset + count].sigmoid() >= mask_threshold)
            tensor_values = values[0, offset:offset + count] * present
            result[name] = {"axes": list(schema.axes), "shape": list(schema.padded_shape),
                            "values": tensor_values.tolist(), "mask": present.tolist()}
            offset += count
        return result, next_state


def train_runtime_sequences(model: RuntimeTensorModel, sequences, *, ontology: str,
                            steps: int, learning_rate: float, mask_weight: float,
                            available_ram_bytes: int, session=None, device="cpu", available_vram_bytes=0,
                            gradient_clip=1.0, validation=None):
    """Separate runtime supervision with ordered input/target event sequences.

    Each sequence contains (seven-category feature dict, Representation) pairs.
    Epochs begin with zero neural context and backpropagate through all events;
    DSP execution is never part of this learned recurrence.
    """
    if not sequences or any(not seq for seq in sequences) or steps <= 0 or not math.isfinite(learning_rate) or not learning_rate > 0 or not math.isfinite(mask_weight) or not mask_weight > 0:
        raise ValueError("Nonempty sequences and explicit positive training settings required")
    parameter_count = sum(p.numel() for p in model.parameters())
    output_width = model.values.out_features
    event_count = sum(len(seq) for seq in sequences)
    estimate = 1024 * steps + 16 * parameter_count + event_count * (64 * (output_width + model.cell.input_size) + 4 * (16 * model.hidden_width + 4 * output_width + model.cell.input_size))
    ModelBudget(estimate, available_ram_bytes, 0, 0)
    def prepare(source):
        data = []
        for sequence in source:
            encoded_sequence = []
            for inputs, target in sequence:
                padded = target.pad(ontology, model.schemas)
                values = torch.tensor([[v for name in model.schemas for v in padded[name].tensor.values]], dtype=torch.float32)
                masks = torch.tensor([[float(v) for name in model.schemas for v in padded[name].mask]], dtype=torch.float32)
                encoded_sequence.append((model.encode(inputs), values, masks))
            if not encoded_sequence:
                raise ValueError("Empty runtime sequence")
            if any(not torch.isfinite(t).all() for event in encoded_sequence for t in event):
                raise ValueError("Runtime features or targets overflow float32")
            data.append(encoded_sequence)
        return data
    if validation:
        estimate += sum(map(len, validation)) * (64 * (output_width + model.cell.input_size) + 4 * (16 * model.hidden_width + 4 * output_width + model.cell.input_size))
    ModelBudget(estimate, available_ram_bytes, 0, 0)
    data = prepare(sequences)
    held = prepare(validation) if validation else None
    def objective(source, selected):
        loss = torch.tensor(0., device=selected)
        for sequence in source:
            state = torch.zeros(1, model.hidden_width, device=selected)
            for encoded, target, mask in sequence:
                values, presence, state = model(encoded.to(selected), state)
                target, mask = target.to(selected), mask.to(selected)
                loss = loss + ((values - target).square() * mask).sum() / mask.sum().clamp_min(1)
                loss = loss + mask_weight * nn.functional.binary_cross_entropy_with_logits(presence, mask)
        return loss / sum(map(len, source))
    contract = {"category": "runtime", "ontology": ontology, "input_widths": model.input_widths,
                "schema_order": list(model.schemas),
                "hidden_width": model.hidden_width, "schemas": {k: asdict(v) for k, v in model.schemas.items()},
                "sequences": [[(inputs, asdict(target)) for inputs, target in seq] for seq in sequences],
                "validation": [[(inputs, asdict(target)) for inputs, target in seq] for seq in validation] if validation else None,
                "mask_weight": mask_weight}
    report = optimize(model, lambda selected: objective(data, selected), steps=steps,
                      learning_rate=learning_rate, available_ram_bytes=available_ram_bytes,
                      estimate=estimate, identity=contract, session=session, device=device,
                      available_vram_bytes=available_vram_bytes, gradient_clip=gradient_clip,
                      validation=(lambda selected: objective(held, selected)) if held else None)
    report["ontology"] = ontology
    return report


def export_runtime_model(path: Path, model: RuntimeTensorModel, *, ontology: str,
                         period_samples: int, phase_samples: int, mask_threshold: float,
                         available_ram_bytes: int, bounds: dict, golden_inputs=(), training_metadata=None):
    """Export exact PyTorch r,z,n GRU gate weights for allocation-free C++ inference.

    The caller supplies model capacity, inference clock and all output bounds.
    No control rate, parameter range or feature encoding is inferred.
    """
    if not ontology or type(period_samples) is not int or period_samples <= 0 or type(phase_samples) is not int or not 0 <= phase_samples < period_samples:
        raise ValueError("Explicit ontology and valid inference clock required")
    if not 0 < mask_threshold < 1 or set(bounds) != set(model.schemas):
        raise ValueError("Explicit threshold and per-tensor output bounds required")
    ordered_bounds = []
    schemas = []
    for name, schema in model.schemas.items():
        count = math.prod(schema.padded_shape)
        if len(bounds[name]) != count:
            raise ValueError("Every output dimension requires explicit bounds")
        for low, high in bounds[name]:
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("Output bounds must be finite and ordered")
            ordered_bounds.append([low, high])
        schemas.append({"name": name, "axes": list(schema.axes), "shape": list(schema.padded_shape)})
    count = sum(p.numel() for p in model.parameters())
    golden_inputs = tuple(golden_inputs)
    estimate = 4 * (count + model.cell.input_size + 12 * model.hidden_width + 4 * model.values.out_features)
    ModelBudget(estimate, available_ram_bytes, 0, 0)
    from .hardware import detect
    export_estimate = 128 * count + 128 * len(golden_inputs) * (model.cell.input_size + 2 * model.values.out_features + model.hidden_width)
    ModelBudget(export_estimate, min(available_ram_bytes, detect(gpu_device=None).available_ram_bytes), 0, 0)
    weights = {name: parameter.detach().cpu().flatten().tolist() for name, parameter in model.named_parameters()}
    if any(not math.isfinite(value) for values in weights.values() for value in values):
        raise ValueError("Runtime weights must be finite")
    document = {"schema": "aom.runtime.gru/1", "ontology": ontology,
                "input_widths": model.input_widths, "hidden_width": model.hidden_width,
                "schemas": schemas, "weights": weights, "initial_state": [0.] * model.hidden_width,
                "output_bounds": ordered_bounds, "presence_threshold": mask_threshold,
                "clock": {"period_samples": period_samples, "phase_samples": phase_samples},
                "budget": {"available_ram_bytes": available_ram_bytes,
                           "estimated_peak_inference_bytes": estimate,
                           "parameter_elements": count}}
    # Golden raw heads and neural state certify mathematical equivalence before
    # bounds, masks or authoritative DSP parameter locks are applied.
    state = torch.zeros(1, model.hidden_width)
    golden = []
    with torch.no_grad():
        for inputs in golden_inputs:
            encoded = model.encode(inputs)
            values, logits, state = model(encoded, state)
            golden.append({"inputs": inputs, "values": values[0].tolist(),
                           "presence_logits": logits[0].tolist(), "state": state[0].tolist()})
    document["golden"] = golden
    if training_metadata is not None:
        document["training"] = inference_metadata(training_metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, document)
    return document
