"""Producer provenance, group splits and bounded, one-record-at-a-time WAV analysis."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import wave

from .contracts import NormalizationRecord, Representation, Tensor, TensorSchema, ModelBudget
from .analysis import AnalysisProfile, read_wav, extract
from .session import fingerprint


def representation(document):
    return Representation(document["ontology"], tuple(
        (name, Tensor(tuple(t["axes"]), tuple(t["shape"]), tuple(t["values"]),
                     tuple(t["presence"]) if "presence" in t else None))
        for name, t in document["tensors"].items()))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    # JSON key ordering, int/float spelling and omitted all-present masks must
    # not allow a renamed copy of the same original into the held-out partition.
    return fingerprint({"ontology": value.ontology, "tensors": {
        name: {"axes": tensor.axes, "shape": tensor.shape,
               "values": [float(v) if v else 0.0 for v in tensor.values],
               "presence": [bool(v) for v in tensor.presence] if tensor.presence is not None else [True] * len(tensor.values)}
        for name, tensor in value.tensors}})


def read_manifest(path, category):
    path = Path(path)
    from .hardware import detect
    ModelBudget(path.stat().st_size * 16, detect(gpu_device=None).available_ram_bytes, 0, 0)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema", "aom.training-manifest/1") != "aom.training-manifest/1":
        raise ValueError("Unsupported training manifest schema")
    if manifest.get("category", category) != category:
        raise ValueError("Manifest belongs to a different model category")
    return manifest


@dataclass(frozen=True)
class PreparedLabel:
    """Only labels survive feature extraction; decoded audio is released per row."""
    example_id: str
    canonical_id: str
    canonical: Representation
    rendered_instance: Representation


def prepare_wav(path, *, extract_features=True, cancelled=None):
    from .hardware import detect
    path = Path(path)
    manifest = read_manifest(path, "wav_preset_learning")
    from .learner import LearnSpec
    spec = LearnSpec(**manifest["learn"])
    spec.validate()
    profile = AnalysisProfile(**{**manifest["analysis"], "harmonic_ratios": tuple(manifest["analysis"]["harmonic_ratios"])})
    profile.validate()
    if profile.recording_policy == "known_calibration":
        raise ValueError("Manifest WAV analysis requires a supplied calibration adapter; use controlled_render for this producer")
    schemas = {k: TensorSchema(tuple(v["axes"]), tuple(v["padded_shape"])) for k, v in manifest["schemas"].items()}
    allowance = min(manifest["learn"]["available_ram_bytes"], detect(gpu_device=None).available_ram_bytes)
    rows = manifest["examples"]
    if not rows:
        raise ValueError("Training examples cannot be empty")
    splits = {"train": ([], []), "validation": ([], [])}
    groups, originals, identities, recordings, sources = {}, {}, set(), {}, []
    peak_decode, resident, max_channels = 0, 32 * len(rows) * sum(math.prod(s.padded_shape) for s in schemas.values()), 0
    ModelBudget(resident, allowance, 0, 0)
    prepared = []
    for row in rows:
        if cancelled and cancelled():
            from .session import TrainingCancelled
            raise TrainingCancelled("Cancelled during dataset preparation")
        split = row.get("split", "train")
        if split not in splits:
            raise ValueError("Each example split must be train or validation")
        if not all(isinstance(row.get(k), str) and row[k] for k in ("id", "canonical_id", "perturbation_id")):
            raise ValueError("Explicit example, canonical and perturbation identities required")
        if row["id"] in identities:
            raise ValueError("Duplicate example identity")
        identities.add(row["id"])
        files = {k: path.parent / row[k] for k in ("normalized_wav", "canonical_json", "instance_json")}
        # Offline perceptual metric factors are part of the objective's identity.
        files.update({"metric_" + head: path.parent / name for head, name in row.get("metric", {}).items()})
        source = {k: file_digest(p) for k, p in files.items()}
        for key in ("canonical_json", "instance_json"):
            resident += files[key].stat().st_size * 8
        ModelBudget(resident, allowance, 0, 0)
        canonical = representation(json.loads(files["canonical_json"].read_text(encoding="utf-8")))
        instance = representation(json.loads(files["instance_json"].read_text(encoding="utf-8")))
        canonical.pad(manifest["ontology"], schemas)
        instance.pad(manifest["ontology"], schemas)
        identity = canonical_digest(canonical)
        group = row["canonical_id"]
        if group in originals and originals[group] != identity:
            raise ValueError("Canonical identity has conflicting original targets")
        originals[group] = identity
        # Detect renamed copies of originals, as well as duplicate recordings.
        for key in ("group:" + group, "original:" + identity):
            if key in groups and groups[key] != split:
                raise ValueError("Canonical identities leak between training and validation")
            groups[key] = split
        if source["normalized_wav"] in recordings and recordings[source["normalized_wav"]] != split:
            raise ValueError("Identical recordings leak between training and validation")
        recordings[source["normalized_wav"]] = split
        record = row["normalization"]
        NormalizationRecord(record["policy_id"], frozenset(record["removed"]), frozenset(record["preserved"]))
        with wave.open(str(files["normalized_wav"]), "rb") as wav:
            if wav.getnframes() == 0 or wav.getnchannels() == 0 or wav.getcomptype() != "NONE" or wav.getsampwidth() not in (1, 2, 3, 4):
                raise ValueError("Nonempty uncompressed PCM recordings required")
            if max_channels and max_channels != wav.getnchannels():
                raise ValueError("Training recordings require a consistent channel feature width")
            max_channels = wav.getnchannels()
            if wav.getnframes() < profile.frames:
                raise ValueError("Analysis frame count exceeds recording length")
            peak_decode = max(peak_decode, wav.getnframes() * wav.getnchannels() * (wav.getsampwidth() + 96))
        sources.append({"id": row["id"], "split": split, "sha256": source})
        prepared.append((split, PreparedLabel(row["id"], group, canonical, instance), files["normalized_wav"]))
    feature_width = profile.frames * (len(profile.harmonic_ratios) + 1) + 1 + max_channels
    resident += len(rows) * feature_width * 64
    ModelBudget(resident + peak_decode, allowance, 0, 0)
    target_width = sum(math.prod(s.padded_shape) for s in schemas.values())
    if not target_width:
        raise ValueError("Training output schema cannot be empty")
    count = feature_width * spec.hidden_width + spec.hidden_width + 4 * (spec.hidden_width + 1) * target_width
    training_estimate = 2048 * spec.steps + 16 * count + len(rows) * (64 * (feature_width + target_width) + 4 * (feature_width + 4 * target_width + 8 * spec.hidden_width))
    from .session import device_for_training
    train_count = sum(split == "train" for split, *_ in prepared)
    batch = min(spec.batch_size or train_count, train_count)
    vram_estimate = 20 * count + 4 * batch * (feature_width + 16 * target_width + 16 * spec.hidden_width)
    device_for_training(spec.device, training_estimate, allowance - resident, spec.available_vram_bytes, estimated_vram_bytes=vram_estimate)
    if not any(split == "train" for split, *_ in prepared):
        raise ValueError("At least one training group required")
    if (manifest.get("require_validation", False) or manifest.get("acceptance") or manifest.get("reconstruction_acceptance")) and not any(split == "validation" for split, *_ in prepared):
        raise ValueError("This manifest requires held-out canonical validation groups")
    from .workflow import acceptance
    acceptance(manifest, {"validation_" + key: 0.0 for key in ("primary", "secondary", "mask", "total", *( ("reconstruction",) if manifest.get("reconstruction") else () ))})
    acceptance({"acceptance": manifest.get("reconstruction_acceptance", {})}, {
        "validation_primary_canonical_relative_rmse": 0.0, "validation_secondary_rendered_instance_relative_rmse": 0.0})
    for split, label, wav_path in prepared:
        splits[split][0].append(label)
        if extract_features:
            if cancelled and cancelled():
                from .session import TrainingCancelled
                raise TrainingCancelled("Cancelled during feature extraction")
            audio, rate = read_wav(wav_path, available_ram_bytes=allowance - resident)
            splits[split][1].append(extract(audio, rate, profile))
            del audio
    audit = {"category": "wav_preset_learning", "examples": len(rows),
             "training_examples": len(splits["train"][0]), "validation_examples": len(splits["validation"][0]),
             "canonical_groups": len(originals), "has_validation": bool(splits["validation"][0]),
             "resident_dataset_estimate_bytes": resident, "peak_decode_estimate_bytes": peak_decode,
             "estimated_peak_training_bytes": resident + training_estimate,
             "estimated_peak_vram_bytes": vram_estimate if spec.device.startswith("cuda:") else 0,
             "sources": sources, "manifest_sha256": file_digest(path)}
    objective = {k: v for k, v in manifest.items() if k not in ("acceptance", "reconstruction_acceptance")}
    objective["learn"] = {k: v for k, v in manifest["learn"].items() if k not in ("steps", "available_ram_bytes", "available_vram_bytes")}
    audit["identity"] = fingerprint({"manifest": objective, "sources": sources})
    return manifest, schemas, splits, audit, allowance - resident
