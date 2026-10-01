"""Independent local preference and context-dependent importance regressors."""
import math
import json
from pathlib import Path
import torch
from torch import nn
from .contracts import ModelBudget
from .session import optimize, atomic_json, inference_metadata


class FeatureModel(nn.Module):
    def __init__(self, feature_widths: dict[str, int], hidden_width: int, output_width: int):
        super().__init__()
        if not feature_widths or any(type(v) is not int or v < 0 for v in feature_widths.values()) or not sum(feature_widths.values()):
            raise ValueError("Explicit named feature widths required")
        if hidden_width <= 0 or output_width <= 0:
            raise ValueError("Positive explicitly chosen model dimensions required")
        self.feature_widths = dict(feature_widths)
        self.output_width = output_width
        self.network = nn.Sequential(nn.Linear(sum(feature_widths.values()), hidden_width), nn.Tanh(), nn.Linear(hidden_width, output_width))

    def encode(self, rows):
        result = []
        for row in rows:
            if set(row) != set(self.feature_widths):
                raise ValueError("Features differ from model schema")
            values = []
            for name, width in self.feature_widths.items():
                if len(row[name]) != width or any(not math.isfinite(v) for v in row[name]):
                    raise ValueError("Feature values require declared width and finite values")
                values.extend(row[name])
            result.append(values)
        encoded = torch.tensor(result, dtype=torch.float32)
        if not torch.isfinite(encoded).all():
            raise ValueError("Local features overflow float32")
        return encoded

    def forward(self, features):
        return self.network(features)

    def check_training_budget(self, observations: int, available_ram_bytes: int):
        count = sum(parameter.numel() for parameter in self.parameters())
        estimate = 16 * count + observations * (64 * (sum(self.feature_widths.values()) + self.output_width) + 4 * (sum(self.feature_widths.values()) + self.output_width + 8 * self.network[0].out_features))
        ModelBudget(estimate, available_ram_bytes, 0, 0)
        return estimate


class ContextImportanceModel(FeatureModel):
    def __init__(self, feature_widths, hidden_width, dimension_count):
        if not {"state", "preference", "context", "relationship"} <= set(feature_widths):
            raise ValueError("Importance must support state, preference, context and relationships")
        super().__init__(feature_widths, hidden_width, dimension_count)

    def predict_importance(self, features):
        with torch.no_grad():
            return self(self.encode([features])).clamp_min(0)[0].tolist()

    def fit(self, observations, *, steps, learning_rate, available_ram_bytes, session=None,
            device="cpu", available_vram_bytes=0, gradient_clip=1.0, validation=None):
        if not observations or steps <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Measured observations and explicit positive training settings required")
        estimate = self.check_training_budget(len(observations), available_ram_bytes)
        x = self.encode([features for features, impacts in observations])
        labels = [impacts for features, impacts in observations]
        if any(len(row) != self.output_width or any(not math.isfinite(v) or v < 0 for v in row) for row in labels):
            raise ValueError("Measured dimension impacts must be finite, nonnegative and match schema")
        y = torch.tensor(labels, dtype=torch.float32)
        held_x, held_y = None, None
        if validation:
            held_x = self.encode([f for f, _ in validation])
            labels = [y for _, y in validation]
            if any(len(row) != self.output_width or any(not math.isfinite(v) or v < 0 for v in row) for row in labels):
                raise ValueError("Invalid validation impacts")
            held_y = torch.tensor(labels, dtype=torch.float32)
            estimate = self.check_training_budget(len(observations) + len(validation), available_ram_bytes)
        estimate += 1024 * steps
        return optimize(self, lambda d: nn.functional.mse_loss(self(x.to(d)), y.to(d)),
                        steps=steps, learning_rate=learning_rate, available_ram_bytes=available_ram_bytes,
                        estimate=estimate, identity={"category": "context_importance", "features": self.feature_widths,
                        "feature_order": list(self.feature_widths),
                        "hidden": self.network[0].out_features, "outputs": self.output_width,
                        "observations": observations, "validation": validation}, session=session,
                        device=device, available_vram_bytes=available_vram_bytes, gradient_clip=gradient_clip,
                        validation=(lambda d: nn.functional.mse_loss(self(held_x.to(d)), held_y.to(d))) if validation else None)



class LocalPreferenceModel(FeatureModel):
    """Learn local pairwise user feedback over arbitrary representation features.

    Features may contain masked AOM coordinates and semantic/context features;
    their encoding is explicit. Preferences are learned from actual local labels,
    not population-derived rankings or a fixed synthesis-method list.
    """
    def __init__(self, feature_widths, hidden_width):
        super().__init__(feature_widths, hidden_width, 1)

    def score(self, features):
        with torch.no_grad():
            return float(self(self.encode([features]))[0, 0])

    def fit_pairs(self, observations, *, steps, learning_rate, available_ram_bytes, session=None,
                  device="cpu", available_vram_bytes=0, gradient_clip=1.0, validation=None):
        """Each row is (first_features, second_features, preferred_first: bool)."""
        if not observations or steps <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Local feedback and explicit positive training settings required")
        estimate = self.check_training_budget(len(observations) * 2, available_ram_bytes)
        if any(type(preferred) is not bool for _, _, preferred in observations):
            raise ValueError("Pairwise labels must be explicit booleans")
        first = self.encode([a for a, _, _ in observations])
        second = self.encode([b for _, b, _ in observations])
        target = torch.tensor([[float(preferred)] for _, _, preferred in observations])
        held = None
        if validation:
            if any(type(preferred) is not bool for _, _, preferred in validation):
                raise ValueError("Pairwise validation labels must be booleans")
            held = (self.encode([a for a, _, _ in validation]), self.encode([b for _, b, _ in validation]),
                    torch.tensor([[float(p)] for _, _, p in validation]))
            estimate = self.check_training_budget(2 * (len(observations) + len(validation)), available_ram_bytes)
        def objective(a, b, y, d):
            return nn.functional.binary_cross_entropy_with_logits(self(a.to(d)) - self(b.to(d)), y.to(d))
        estimate += 1024 * steps
        return optimize(self, lambda d: objective(first, second, target, d), steps=steps,
                        learning_rate=learning_rate, available_ram_bytes=available_ram_bytes,
                        estimate=estimate, identity={"category": "user_preference", "features": self.feature_widths,
                        "feature_order": list(self.feature_widths),
                        "hidden": self.network[0].out_features, "observations": observations, "validation": validation},
                        session=session, device=device, available_vram_bytes=available_vram_bytes,
                        gradient_clip=gradient_clip, validation=(lambda d: objective(*held, d)) if held else None)

    def preference_tensor(self):
        """Versioned local learned-state vector for caller-defined preference input."""
        values = [value for parameter in self.parameters() for value in parameter.detach().flatten().tolist()]
        return {"axes": ["local_preference_state"], "shape": [len(values)], "values": values}


def save_local_model(path: Path, model: FeatureModel, *, training_metadata=None, available_ram_bytes=None):
    """Persist learned local state without embedding individual user ratings."""
    if isinstance(model, LocalPreferenceModel):
        category = "user_preference"
    elif isinstance(model, ContextImportanceModel):
        category = "context_importance"
    else:
        raise ValueError("Unsupported local model category")
    from .hardware import detect
    allowance = detect(gpu_device=None).available_ram_bytes
    if available_ram_bytes is not None:
        allowance = min(allowance, available_ram_bytes)
    ModelBudget(128 * sum(p.numel() for p in model.parameters()) + (2048 * len(training_metadata.get("losses", [])) if training_metadata else 0), allowance, 0, 0)
    package = {"schema": "aom.local-learning/1", "category": category,
               "feature_widths": model.feature_widths, "hidden_width": model.network[0].out_features,
               "output_width": model.output_width,
               "weights": {name: {"shape": list(tensor.shape), "values": tensor.detach().flatten().tolist()}
                           for name, tensor in model.state_dict().items()}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if training_metadata is not None:
        package["training"] = inference_metadata(training_metadata)
    atomic_json(path, package)
    return package


def load_local_model(path: Path, *, available_ram_bytes: int):
    ModelBudget(Path(path).stat().st_size * 4, available_ram_bytes, 0, 0)
    package = json.loads(Path(path).read_text(encoding="utf-8"))
    if package.get("schema") != "aom.local-learning/1":
        raise ValueError("Unsupported local model format")
    input_width = sum(package["feature_widths"].values())
    hidden, output = package["hidden_width"], package["output_width"]
    ModelBudget(16 * ((input_width + 1) * hidden + (hidden + 1) * output), available_ram_bytes, 0, 0)
    if package["category"] == "user_preference" and output == 1:
        model = LocalPreferenceModel(package["feature_widths"], hidden)
    elif package["category"] == "context_importance":
        model = ContextImportanceModel(package["feature_widths"], hidden, output)
    else:
        raise ValueError("Unsupported local model category")
    weights = {}
    for name, expected in model.state_dict().items():
        source = package["weights"][name]
        if source["shape"] != list(expected.shape) or len(source["values"]) != expected.numel() or any(not math.isfinite(v) for v in source["values"]):
            raise ValueError("Local model weight shape or values invalid")
        weights[name] = torch.tensor(source["values"], dtype=torch.float32).reshape(expected.shape)
    model.load_state_dict(weights)
    model.eval()
    return model
