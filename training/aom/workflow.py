"""Independent manifest workflows and publication gates for AOM Centre."""
import json
import math
from pathlib import Path

from .contracts import ModelBudget, TensorSchema
from .dataset import read_manifest, representation, file_digest
from .session import TrainingSession, fingerprint


def validate_settings(manifest):
    if type(manifest["steps"]) is not int or manifest["steps"] <= 0 or type(manifest["seed"]) is not int:
        raise ValueError("Positive integer steps and explicit integer seed required")
    for key in ("learning_rate", "gradient_clip"):
        value = manifest[key] if key == "learning_rate" else manifest.get(key, 1.0)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Positive finite learning rate and gradient clipping required")


def acceptance(manifest, metrics):
    """Only producer-declared held-out thresholds authorize a quality claim."""
    gates = manifest.get("acceptance", {})
    for name, limit in gates.items():
        if not name.endswith("_max") or type(limit) not in (int, float) or not math.isfinite(limit) or limit < 0:
            raise ValueError("Acceptance gates require finite nonnegative *_max thresholds")
        metric = name[:-4]
        if not metric.startswith("validation_") or metric not in metrics:
            raise ValueError("Acceptance gate requires the named held-out validation metric")
        if metrics[metric] > limit:
            raise ValueError(f"Publication refused: {metric}={metrics[metric]:.8g} exceeds {limit:.8g}")
    return {"quality_gates": gates, "quality_gates_passed": bool(gates)}


def bind_provenance(session, category, manifest, sources=()):
    session = session or TrainingSession(category)
    objective = {k: v for k, v in manifest.items() if k not in ("steps", "available_ram_bytes", "available_vram_bytes", "acceptance")}
    session.producer_identity = fingerprint({"session": session.producer_identity,
                                            "manifest": objective, "sources": sources})
    return session


def prepare_runtime(path):
    from .runtime_model import INPUT_CATEGORIES
    from .hardware import detect
    path = Path(path)
    manifest = read_manifest(path, "runtime")
    validate_settings(manifest)
    if not math.isfinite(manifest["mask_weight"]) or manifest["mask_weight"] <= 0:
        raise ValueError("Positive finite runtime presence loss weight required")
    widths, hidden = manifest["input_widths"], manifest["hidden_width"]
    if set(widths) != set(INPUT_CATEGORIES) or any(type(v) is not int or v < 0 for v in widths.values()) or not sum(widths.values()):
        raise ValueError("Explicit valid widths for all six runtime inputs required")
    if type(hidden) is not int or hidden <= 0:
        raise ValueError("Positive runtime hidden width required")
    schemas = {k: TensorSchema(tuple(v["axes"]), tuple(v["padded_shape"])) for k, v in manifest["schemas"].items()}
    output = sum(math.prod(s.padded_shape) for s in schemas.values())
    count = 3 * hidden * sum(widths.values()) + 3 * hidden * hidden + 6 * hidden + 2 * output * (hidden + 1)
    clock = manifest["clock"]
    if not manifest["ontology"] or not output or type(clock["period_samples"]) is not int or clock["period_samples"] <= 0 or type(clock["phase_samples"]) is not int or not 0 <= clock["phase_samples"] < clock["period_samples"]:
        raise ValueError("Explicit ontology, output dimensions and valid runtime clock required")
    if not 0 < manifest["mask_threshold"] < 1 or set(manifest["bounds"]) != set(schemas):
        raise ValueError("Runtime threshold and output bounds required")
    for name, schema in schemas.items():
        if len(manifest["bounds"][name]) != math.prod(schema.padded_shape) or any(len(b) != 2 or not all(math.isfinite(v) for v in b) or b[0] > b[1] for b in manifest["bounds"][name]):
            raise ValueError("Finite runtime bounds matching each output dimension required")
    allowance = min(manifest["available_ram_bytes"], detect(gpu_device=None).available_ram_bytes)
    ModelBudget(16 * count, allowance, 0, 0)
    sequences, held, groups, sources = [], [], {}, []
    rows = manifest["sequences"]
    if not rows:
        raise ValueError("Runtime sequences cannot be empty")
    event_count = sum(len(row["events"] if isinstance(row, dict) else row) for row in rows)
    estimate = 1024 * manifest["steps"] + 16 * count + event_count * (64 * (output + sum(widths.values())) + 4 * (16 * hidden + 4 * output + sum(widths.values())))
    ModelBudget(estimate, allowance, 0, 0)
    for index, row in enumerate(rows):
        # Legacy ordered event lists remain readable, but carry no validation claim.
        split = row.get("split", "train") if isinstance(row, dict) else "train"
        events = row["events"] if isinstance(row, dict) else row
        group = row.get("canonical_id") if isinstance(row, dict) else str(index)
        if split not in ("train", "validation") or not isinstance(group, str) or not group or not events:
            raise ValueError("Runtime sequence requires events, group identity and a valid split")
        if group in groups and groups[group] != split:
            raise ValueError("Runtime canonical group leaks between training and validation")
        groups[group] = split
        sequence = []
        for event in events:
            inputs = event["inputs"]
            if set(inputs) != set(widths) or any(len(inputs[k]) != widths[k] or any(not math.isfinite(v) for v in inputs[k]) for k in widths):
                raise ValueError("Runtime event inputs differ from their explicit feature schema")
            source = path.parent / event["target_json"]
            ModelBudget(source.stat().st_size * 8 + 16 * count + 32 * output, allowance, 0, 0)
            target = representation(json.loads(source.read_text(encoding="utf-8")))
            target.pad(manifest["ontology"], schemas)
            sequence.append((inputs, target))
            sources.append(file_digest(source))
        (sequences if split == "train" else held).append(sequence)
    if not sequences or ((manifest.get("require_validation") or manifest.get("acceptance")) and not held):
        raise ValueError("Runtime training and declared validation groups required")
    acceptance(manifest, {"validation_total": 0.0})
    golden_count = manifest.get("golden_event_count", len(sequences[0]))
    if type(golden_count) is not int or not 0 <= golden_count <= len(sequences[0]):
        raise ValueError("Golden event count must fit the first ordered training sequence")
    export_estimate = 128 * count + 128 * golden_count * (sum(widths.values()) + 2 * output + hidden)
    ModelBudget(export_estimate, allowance, 0, 0)
    event_count = sum(map(len, sequences + held))
    ModelBudget(estimate, allowance, 0, 0)
    from .session import device_for_training
    device_for_training(manifest.get("device", "cpu"), estimate, allowance, manifest.get("available_vram_bytes", 0))
    audit = {"category": "runtime", "training_sequences": len(sequences), "validation_sequences": len(held),
             "canonical_groups": len(groups), "events": event_count, "has_validation": bool(held),
             "estimated_peak_training_bytes": estimate, "sources": sources}
    return manifest, schemas, sequences, held, audit, allowance


def train_runtime(path, destination, *, session=None):
    import torch
    from .runtime_model import RuntimeTensorModel, train_runtime_sequences, export_runtime_model
    manifest, schemas, sequences, held, audit, allowance = prepare_runtime(path)
    from .session import protect_outputs
    protect_outputs(destination, [path, *(Path(path).parent / event["target_json"] for row in manifest["sequences"] for event in (row["events"] if isinstance(row, dict) else row))], session)
    session = bind_provenance(session, "runtime", manifest, audit["sources"])
    torch.manual_seed(manifest["seed"])
    model = RuntimeTensorModel(manifest["input_widths"], manifest["hidden_width"], schemas)
    report = train_runtime_sequences(model, sequences, ontology=manifest["ontology"],
        steps=manifest["steps"], learning_rate=manifest["learning_rate"], mask_weight=manifest["mask_weight"],
        available_ram_bytes=allowance, validation=held, session=session, device=manifest.get("device", "cpu"),
        available_vram_bytes=manifest.get("available_vram_bytes", 0), gradient_clip=manifest.get("gradient_clip", 1.0))
    report.update(acceptance(manifest, report["metrics"][-1]), dataset=audit)
    report.update(model=str(destination), initial_loss=report["losses"][0], final_loss=report["losses"][-1])
    if session.cancelled():
        from .session import TrainingCancelled
        raise TrainingCancelled("Cancelled before runtime model publication; checkpoint retained")
    export_runtime_model(destination, model, ontology=manifest["ontology"],
        period_samples=manifest["clock"]["period_samples"], phase_samples=manifest["clock"]["phase_samples"],
        mask_threshold=manifest["mask_threshold"], available_ram_bytes=allowance, bounds=manifest["bounds"],
        golden_inputs=[inputs for inputs, _ in sequences[0][:manifest.get("golden_event_count", len(sequences[0]))]], training_metadata=report)
    return report


def prepare_local(path, category):
    from .hardware import detect
    manifest = read_manifest(path, category)
    validate_settings(manifest)
    widths, hidden = manifest["feature_widths"], manifest["hidden_width"]
    output = 1 if category == "user_preference" else manifest["dimension_count"]
    if not widths or any(type(v) is not int or v < 0 for v in widths.values()) or not sum(widths.values()):
        raise ValueError("Explicit feature widths required")
    if type(hidden) is not int or hidden <= 0 or type(output) is not int or output <= 0:
        raise ValueError("Positive local model dimensions required")
    if category == "context_importance" and not {"state", "preference", "context", "relationship"} <= set(widths):
        raise ValueError("Importance features require state, preference, context and relationship")
    groups, train, held = {}, [], []
    def features(row):
        if set(row) != set(widths) or any(len(row[k]) != widths[k] or any(not math.isfinite(v) for v in row[k]) for k in widths):
            raise ValueError("Local observation features differ from schema")
        return row
    for row in manifest["observations"]:
        group, split = row["group_id"], row.get("split", "train")
        if not isinstance(group, str) or not group or split not in ("train", "validation"):
            raise ValueError("Local feedback requires a group identity and valid split")
        if group in groups and groups[group] != split:
            raise ValueError("Local observation group leaks between training and validation")
        groups[group] = split
        if category == "user_preference":
            if type(row["preferred_first"]) is not bool:
                raise ValueError("Local pair preference must be an explicit boolean")
            value = (features(row["first"]), features(row["second"]), row["preferred_first"])
        else:
            impacts = row["impacts"]
            if len(impacts) != output or any(not math.isfinite(v) or v < 0 for v in impacts):
                raise ValueError("Importance targets must be measured nonnegative impacts matching dimensions")
            value = (features(row["features"]), impacts)
        (train if split == "train" else held).append(value)
    if not train or ((manifest.get("require_validation") or manifest.get("acceptance")) and not held):
        raise ValueError("Local training and declared validation groups required")
    acceptance(manifest, {"validation_total": 0.0})
    count = (sum(widths.values()) + 1) * hidden + (hidden + 1) * output
    estimate = 1024 * manifest["steps"] + 16 * count + (len(train) + len(held)) * (2 if category == "user_preference" else 1) * (64 * (sum(widths.values()) + output) + 4 * (sum(widths.values()) + output + 8 * hidden))
    allowance = min(manifest["available_ram_bytes"], detect(gpu_device=None).available_ram_bytes)
    ModelBudget(estimate, allowance, 0, 0)
    from .session import device_for_training
    device_for_training(manifest.get("device", "cpu"), estimate, allowance, manifest.get("available_vram_bytes", 0))
    audit = {"category": category, "training_observations": len(train), "validation_observations": len(held),
             "has_validation": bool(held), "estimated_peak_training_bytes": estimate}
    return manifest, train, held, audit, allowance


def train_local(path, destination, category, *, session=None):
    import torch
    from .local_learning import LocalPreferenceModel, ContextImportanceModel, save_local_model
    manifest, data, held, audit, allowance = prepare_local(path, category)
    from .session import protect_outputs
    protect_outputs(destination, [path], session)
    session = bind_provenance(session, category, manifest)
    torch.manual_seed(manifest["seed"])
    if category == "user_preference":
        model = LocalPreferenceModel(manifest["feature_widths"], manifest["hidden_width"])
        fit = model.fit_pairs
    else:
        model = ContextImportanceModel(manifest["feature_widths"], manifest["hidden_width"], manifest["dimension_count"])
        fit = model.fit
    report = fit(data, steps=manifest["steps"], learning_rate=manifest["learning_rate"],
                 available_ram_bytes=allowance, session=session, validation=held,
                 device=manifest.get("device", "cpu"), available_vram_bytes=manifest.get("available_vram_bytes", 0),
                 gradient_clip=manifest.get("gradient_clip", 1.0))
    report.update(acceptance(manifest, report["metrics"][-1]), dataset=audit)
    if session.cancelled():
        from .session import TrainingCancelled
        raise TrainingCancelled("Cancelled before local model publication; checkpoint retained")
    save_local_model(destination, model, training_metadata=report, available_ram_bytes=allowance)
    return report


def preflight(path, category):
    if category == "wav_preset_learning":
        from .dataset import prepare_wav
        manifest, schemas, _, audit, allowance = prepare_wav(path, extract_features=False)
        if manifest.get("reconstruction"):
            from .reconstruction import NativeReconstruction
            objective = NativeReconstruction(path, manifest, schemas, available_ram_bytes=allowance)
            audit["native_objective_estimated_peak_bytes"] = objective.estimated_peak_bytes
            audit["estimated_peak_training_bytes"] += objective.estimated_peak_bytes
            ModelBudget(audit["estimated_peak_training_bytes"], allowance + audit["resident_dataset_estimate_bytes"], 0, 0)
    elif category == "runtime":
        manifest, _, _, _, audit, _ = prepare_runtime(path)
    elif category in ("user_preference", "context_importance"):
        manifest, _, _, audit, _ = prepare_local(path, category)
    else:
        raise ValueError("Unsupported model category")
    return audit
