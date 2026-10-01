"""Serialize learned tensor predictions into the native compiler document."""
import copy
import math
from pathlib import Path

from .dataset import representation
from .analysis import AnalysisProfile, read_wav, extract
from .learner import LearnSpec, train, save, load, predict


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
        if len(presence) != len(limits) or locks and len(locks) != len(limits):
            raise ValueError("Presence and parameter locks must cover their tensor dimensions")
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


def train_manifest(path: Path, destination: Path, *, session=None):
    """Stream WAV analysis, retain labels/features, and validate held-out groups."""
    from .dataset import prepare_wav
    manifest, schemas, splits, audit, allowance = prepare_wav(
        path, cancelled=session.cancelled if session else None)
    from .session import protect_outputs
    protect_outputs(destination, [path, *(Path(path).parent / row[key] for row in manifest["examples"] for key in ("normalized_wav", "canonical_json", "instance_json"))], session)
    examples, features = splits["train"]
    held = splits["validation"] if splits["validation"][0] else None
    from .session import TrainingSession
    session = session or TrainingSession("wav_preset_learning")
    from .session import fingerprint
    session.producer_identity = fingerprint({"session": session.producer_identity, "dataset": audit["identity"]})
    reconstruction = None
    if manifest.get("reconstruction"):
        from .reconstruction import NativeReconstruction
        reconstruction = NativeReconstruction(path, manifest, schemas, available_ram_bytes=allowance, cancelled=session.cancelled)
        allowance -= reconstruction.estimated_peak_bytes
    model, metadata = train(tuple(examples), features, manifest["ontology"], schemas,
                            LearnSpec(**{**manifest["learn"], "available_ram_bytes": allowance}),
                            session=session, validation=held, reconstruction=reconstruction)
    metadata["analysis"] = manifest["analysis"]
    metadata["dataset"] = audit
    metadata["estimated_peak_training_bytes"] += audit["resident_dataset_estimate_bytes"] + (reconstruction.estimated_peak_bytes if reconstruction else 0)
    from .workflow import acceptance
    metadata.update(acceptance(manifest, metadata["losses"][-1]))
    if manifest.get("reconstruction_acceptance"):
        import os
        import tempfile
        from .producer import evaluate_wav
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(suffix=".pt", dir=Path(destination).parent)
        os.close(fd)
        try:
            save(temporary, model, metadata)
            reconstruction = evaluate_wav(temporary, path)
            acceptance({"acceptance": manifest["reconstruction_acceptance"]}, reconstruction["metrics"])
            reconstruction["candidate_package_sha256"] = reconstruction.pop("model_sha256")
            metadata["reconstruction"] = reconstruction
            metadata["reconstruction_gates_passed"] = True
            metadata["quality_gates_passed"] = True
        finally:
            os.unlink(temporary)
    if session.cancelled():
        from .session import TrainingCancelled
        raise TrainingCancelled("Cancelled before model publication; checkpoint retained")
    save(destination, model, metadata)
    return metadata


def infer_wav(model_path: Path, wav_path: Path, *, mask_threshold: float):
    model, metadata = load(model_path)
    profile = AnalysisProfile(**{**metadata["analysis"], "harmonic_ratios": tuple(metadata["analysis"]["harmonic_ratios"])})
    audio, rate = read_wav(wav_path)
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


def train_runtime_manifest(path: Path, destination: Path, *, session=None):
    from .workflow import train_runtime
    return train_runtime(path, destination, session=session)
