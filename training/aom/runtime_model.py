"""Separate recurrent runtime compiler model; it predicts tensors, not audio."""
import math
import json
from pathlib import Path
import torch
from torch import nn
from .contracts import ModelBudget, Representation, TensorSchema


INPUT_CATEGORIES = ("audio", "midi", "gesture", "user_preference", "latent", "context")


class RuntimeTensorModel(nn.Module):
    """Inference is invoked at producer-scheduled events, with no assumed rate.

    Inputs are explicitly encoded feature tensors; raw MIDI units, audio feature
    extraction, semantic controls and preference encodings remain caller-owned.
    Neural hidden state is distinct from persistent deterministic DSP state.
    """
    def __init__(self, input_widths: dict[str, int], hidden_width: int, schemas: dict[str, TensorSchema]):
        super().__init__()
        if set(input_widths) != set(INPUT_CATEGORIES) or any(type(v) is not int or v < 0 for v in input_widths.values()):
            raise ValueError("Explicit widths for all six input categories required")
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
            raise ValueError("All six input categories require explicit data or empty vectors")
        flat = []
        for category in INPUT_CATEGORIES:
            values = inputs[category]
            if len(values) != self.input_widths[category] or any(not math.isfinite(x) for x in values):
                raise ValueError(f"Invalid {category} feature tensor")
            flat.extend(values)
        return torch.tensor([flat], dtype=torch.float32)

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
                            available_ram_bytes: int):
    """Separate runtime supervision with ordered input/target event sequences.

    Each sequence contains (six-category feature dict, Representation) pairs.
    Epochs begin with zero neural context and backpropagate through all events;
    DSP execution is never part of this learned recurrence.
    """
    if not sequences or any(not seq for seq in sequences) or steps <= 0 or not learning_rate > 0 or not mask_weight > 0:
        raise ValueError("Nonempty sequences and explicit positive training settings required")
    parameter_count = sum(p.numel() for p in model.parameters())
    output_width = model.values.out_features
    event_count = sum(len(seq) for seq in sequences)
    estimate = 16 * parameter_count + 4 * event_count * (16 * model.hidden_width + 4 * output_width + model.cell.input_size)
    ModelBudget(estimate, available_ram_bytes, 0, 0)
    data = []
    for sequence in sequences:
        encoded_sequence = []
        for inputs, target in sequence:
            padded = target.pad(ontology, model.schemas)
            values = torch.tensor([[v for name in model.schemas for v in padded[name].tensor.values]])
            masks = torch.tensor([[float(v) for name in model.schemas for v in padded[name].mask]])
            encoded_sequence.append((model.encode(inputs), values, masks))
        data.append(encoded_sequence)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.tensor(0.)
        for sequence in data:
            state = torch.zeros(1, model.hidden_width)
            for encoded, target, mask in sequence:
                values, presence, state = model(encoded, state)
                loss = loss + ((values - target).square() * mask).sum() / mask.sum().clamp_min(1)
                loss = loss + mask_weight * nn.functional.binary_cross_entropy_with_logits(presence, mask)
        loss = loss / event_count
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite runtime training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return {"losses": losses, "estimated_peak_training_bytes": estimate, "ontology": ontology}


def export_runtime_model(path: Path, model: RuntimeTensorModel, *, ontology: str,
                         period_samples: int, phase_samples: int, mask_threshold: float,
                         available_ram_bytes: int, bounds: dict, golden_inputs=()):
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
    estimate = 4 * (count + model.cell.input_size + 12 * model.hidden_width + 4 * model.values.out_features)
    ModelBudget(estimate, available_ram_bytes, 0, 0)
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    return document
