"""Serialize learned tensor predictions into the native compiler document."""
import copy
import json
import math
import wave
from pathlib import Path

from .contracts import Representation, Tensor, TensorSchema, NormalizationRecord, WavExample
from .analysis import AnalysisProfile, read_wav, extract
from .learner import LearnSpec, train, save, load, predict


def representation(document):
    return Representation(document["ontology"], tuple(
        (name, Tensor(tuple(tensor["axes"]), tuple(tensor["shape"]), tuple(tensor["values"]),
                     tuple(tensor["presence"]) if "presence" in tensor else None))
        for name, tensor in document["tensors"].items()))


def export_prediction(template: dict, prediction: dict, *, head: str, bounds: dict):
    """Explicit structural template plus predicted tensors, never raw waveform.

    Structural lowering remains the native compiler's job. Parameter locks are
    copied unchanged. Caller bounds apply per tensor element, with no assumed DSP
    range. Predictions cannot add primitive implementations absent in template.
    """
    if head not in ("primary_canonical", "secondary_rendered_instance"):
        raise ValueError("Select a supervision head explicitly")
    if template["ontology"] != prediction["ontology"]:
        raise ValueError("Prediction and template ontology differ")
    result = copy.deepcopy(template)
    tensors = prediction[head]
    if set(tensors) != set(result["tensors"]) or set(bounds) != set(tensors):
        raise ValueError("Template, predictions and explicit bounds must cover identical tensors")
    for name, tensor in tensors.items():
        old = result["tensors"][name]
        if old["axes"] != list(tensor["axes"]) or old["shape"] != list(tensor["shape"]):
            raise ValueError("Prediction needs a matching shared structural lowering template")
        limits = bounds[name]
        if len(limits) != len(tensor["values"]):
            raise ValueError("Every prediction dimension requires explicit bounds")
        values = []
        locks = result.get("parameter_locks", []) if name == "parameters" else []
        presence = list(tensor["mask"])
        for index, (value, (low, high)) in enumerate(zip(tensor["values"], limits)):
            if not all(math.isfinite(v) for v in (value, low, high)) or low > high:
                raise ValueError("Invalid prediction bounds")
            if locks and locks[index]:
                values.append(old["values"][index])
                presence[index] = bool(old.get("presence", [True] * len(limits))[index])
            else:
                values.append(min(high, max(low, value)) if presence[index] else 0.0)
        old["values"] = values
        old["presence"] = presence
    return result


def train_manifest(path: Path, destination: Path):
    """Manifest points to producer-labeled canonical/instance JSON and WAV files."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    profile = AnalysisProfile(**{**manifest["analysis"], "harmonic_ratios": tuple(manifest["analysis"]["harmonic_ratios"])})
    schemas = {name: TensorSchema(tuple(value["axes"]), tuple(value["padded_shape"]))
               for name, value in manifest["schemas"].items()}
    examples, features = [], []
    # Check aggregate resident dataset estimates before loading any WAV labels.
    from .contracts import ModelBudget
    from .hardware import detect
    allowance = min(manifest["learn"]["available_ram_bytes"], detect(gpu_device=None).available_ram_bytes)
    dataset_estimate = 0
    for row in manifest["examples"]:
        with wave.open(str(path.parent / row["normalized_wav"]), "rb") as wav:
            dataset_estimate += wav.getnframes() * wav.getnchannels() * (wav.getsampwidth() + 96)
    ModelBudget(dataset_estimate, allowance, 0, 0)
    for row in manifest["examples"]:
        audio, rate = read_wav(path.parent / row["normalized_wav"], available_ram_bytes=allowance)
        canonical = representation(json.loads((path.parent / row["canonical_json"]).read_text(encoding="utf-8")))
        instance = representation(json.loads((path.parent / row["instance_json"]).read_text(encoding="utf-8")))
        record = row["normalization"]
        normalization = NormalizationRecord(record["policy_id"], frozenset(record["removed"]), frozenset(record["preserved"]))
        examples.append(WavExample(row["id"], row["canonical_id"], canonical, instance,
                        Tensor(("sample", "channel"), tuple(audio.shape), tuple(audio.flat)),
                        rate, normalization, row["perturbation_id"]))
        features.append(extract(audio, rate, profile))
    spec = {**manifest["learn"], "available_ram_bytes": allowance - dataset_estimate}
    model, metadata = train(tuple(examples), features, manifest["ontology"], schemas, LearnSpec(**spec))
    metadata["analysis"] = manifest["analysis"]
    save(destination, model, metadata)
    return metadata


def infer_wav(model_path: Path, wav_path: Path, *, mask_threshold: float):
    model, metadata = load(model_path)
    profile = AnalysisProfile(**{**metadata["analysis"], "harmonic_ratios": tuple(metadata["analysis"]["harmonic_ratios"])})
    audio, rate = read_wav(wav_path, available_ram_bytes=metadata["spec"]["available_ram_bytes"])
    return predict(model, metadata, extract(audio, rate, profile), mask_threshold=mask_threshold)


def import_wav(model_path: Path, wav_path: Path, template: dict, bounds: dict, *,
               head: str, mask_threshold: float, sample_rate: float, dll_path=None):
    """Infer editable tensors and require actual native compiler acceptance."""
    from .native import NativeRuntime
    predicted = infer_wav(model_path, wav_path, mask_threshold=mask_threshold)
    document = export_prediction(template, predicted, head=head, bounds=bounds)
    with NativeRuntime(document, sample_rate, dll_path):
        pass
    return document


def train_runtime_manifest(path: Path, destination: Path):
    """Independent runtime event learner and exact C++ model export."""
    import torch
    from .runtime_model import RuntimeTensorModel, train_runtime_sequences, export_runtime_model
    from .contracts import ModelBudget
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schemas = {name: TensorSchema(tuple(value["axes"]), tuple(value["padded_shape"]))
               for name, value in manifest["schemas"].items()}
    input_width = sum(manifest["input_widths"].values())
    hidden = manifest["hidden_width"]
    outputs = sum(math.prod(schema.padded_shape) for schema in schemas.values())
    parameter_count = 3 * hidden * input_width + 3 * hidden * hidden + 6 * hidden + 2 * outputs * (hidden + 1)
    ModelBudget(16 * parameter_count, manifest["available_ram_bytes"], 0, 0)
    torch.manual_seed(manifest["seed"])
    model = RuntimeTensorModel(manifest["input_widths"], hidden, schemas)
    sequences = []
    for sequence in manifest["sequences"]:
        sequences.append([(event["inputs"], representation(json.loads((path.parent / event["target_json"]).read_text(encoding="utf-8"))))
                          for event in sequence])
    report = train_runtime_sequences(model, sequences, ontology=manifest["ontology"],
                                    steps=manifest["steps"], learning_rate=manifest["learning_rate"],
                                    mask_weight=manifest["mask_weight"], available_ram_bytes=manifest["available_ram_bytes"])
    export_runtime_model(destination, model, ontology=manifest["ontology"],
                         period_samples=manifest["clock"]["period_samples"], phase_samples=manifest["clock"]["phase_samples"],
                         mask_threshold=manifest["mask_threshold"], available_ram_bytes=manifest["available_ram_bytes"],
                         bounds=manifest["bounds"], golden_inputs=[event["inputs"] for event in manifest["sequences"][0]])
    return {"model": str(destination), "initial_loss": report["losses"][0], "final_loss": report["losses"][-1],
            "estimated_peak_training_bytes": report["estimated_peak_training_bytes"]}
