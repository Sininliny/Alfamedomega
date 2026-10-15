"""Build auditable controlled-render datasets from explicit native tensor programs."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import wave
import numpy as np

from .native import NativeRuntime
from .dataset import prepare_wav, file_digest
from .session import atomic_json, TrainingCancelled
from .schema import schema_is


INPUTS = ("audio", "midi", "gesture", "preference", "latent", "context")
OPTIONAL_INPUTS = ("wave_feature",)   # wave feature knobs; absent means none


def performance(spec, allowance):
    from .contracts import ModelBudget
    frames, rate = spec["frames"], spec["sample_rate"]
    if type(frames) is not int or frames <= 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Explicit positive frame count and sample rate required")
    if not set(INPUTS) <= set(spec["inputs"]) <= set(INPUTS + OPTIONAL_INPUTS):
        raise ValueError("All six native input streams require explicit vectors; wave_feature is optional")
    streams, elements = {}, 0
    for name, value in spec["inputs"].items():
        if isinstance(value, dict):
            if set(value) != {"frames"} or len(value["frames"]) != frames or not isinstance(value["frames"][0], list):
                raise ValueError("Framewise performance requires exactly one feature row per frame")
            rows = value["frames"]
            width = len(rows[0])
            if any(not isinstance(row, list) or len(row) != width or any(not math.isfinite(v) for v in row) for row in rows):
                raise ValueError("Performance trajectories require fixed-width finite feature rows")
            streams[name] = (rows, width, True)
        else:
            if not isinstance(value, list) or any(not math.isfinite(v) for v in value):
                raise ValueError("Constant performance requires a finite feature vector")
            width = len(value)
            streams[name] = (value, width, False)
        elements += frames * width
    ModelBudget(8 * elements + 96 * frames, allowance, 0, 0)
    arrays = {}
    for name, (values, width, framewise) in streams.items():
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Performance values overflow float32")
        arrays[name] = array if framewise else np.broadcast_to(array, (frames, width))
    return arrays


def build_dataset(plan_path, destination, *, dll_path=None, cancelled=None, progress=None):
    plan_path, destination = Path(plan_path), Path(destination).resolve()
    from .hardware import detect
    from .contracts import ModelBudget
    measured = detect(gpu_device=None).available_ram_bytes
    ModelBudget(plan_path.stat().st_size * 16, measured, 0, 0)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not schema_is(plan.get("schema"), "amo.dataset-plan/1") or plan["analysis"]["recording_policy"] != "controlled_render":
        raise ValueError("An explicit controlled-render AMO dataset plan is required")
    allowance = min(plan["learn"]["available_ram_bytes"], measured)
    if destination.exists():
        raise ValueError("Dataset destination already exists; choose a new version directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".staging-", dir=destination.parent))
    examples = []
    metric_plan = plan.get("perceptual_metric")
    coordinates = None
    if metric_plan is not None or "parameter_coordinates" in plan:
        from .perceptual import Coordinates
        width = len(plan["bounds"]["parameters"])
        coordinates = Coordinates(plan.get("parameter_coordinates") or ["linear"] * width, plan["bounds"]["parameters"])
    if metric_plan is not None:
        validate_metric_plan(metric_plan)
    variants = sum(len(g["variants"]) for g in plan["groups"])
    if not variants:
        raise ValueError("At least one explicitly authored group/variant required")
    for group_index, group in enumerate(plan["groups"]):
        source = plan_path.parent / group["canonical_json"]
        document_bytes = source.stat().st_size * 32
        ModelBudget(document_bytes, allowance, 0, 0)
        canonical = json.loads(source.read_text(encoding="utf-8"))
        original_path = staging / f"canonical-{group_index}.json"
        atomic_json(original_path, canonical)
        for variant_index, variant in enumerate(group["variants"]):
            if cancelled and cancelled():
                raise TrainingCancelled(f"Dataset generation cancelled; unpublished staging directory: {staging}")
            instance = copy.deepcopy(canonical)
            for name, updates in variant["updates"].items():
                tensor, bounds = instance["tensors"][name], plan["bounds"][name]
                if len(bounds) != len(tensor["values"]):
                    raise ValueError("Every perturbable dimension requires an explicit bound")
                used = set()
                for update in updates:
                    index, value = update["index"], update["value"]
                    if type(index) is not int or not 0 <= index < len(bounds) or index in used:
                        raise ValueError("Perturbation indices must be valid and unique")
                    used.add(index)
                    low, high = bounds[index]
                    if not all(math.isfinite(v) for v in (low, high, value)) or not low <= value <= high:
                        raise ValueError("Perturbation exceeds declared bounds")
                    if name == "parameters" and instance.get("parameter_locks", [False] * len(bounds))[index]:
                        raise ValueError("Perturbation cannot change a locked parameter")
                    tensor["values"][index] = value
                    if "present" in update:
                        if type(update["present"]) is not bool:
                            raise ValueError("Explicit boolean presence required")
                        tensor.setdefault("presence", [True] * len(bounds))[index] = update["present"]
                        if not update["present"] and value != 0:
                            raise ValueError("Absent perturbation elements must be zero")
            render = variant["render"]
            if type(render["sample_rate"]) is not int:
                raise ValueError("PCM dataset generation requires an integer sample rate")
            arrays = performance(render, allowance - document_bytes)
            with NativeRuntime(canonical, render["sample_rate"], dll_path):
                pass
            with NativeRuntime(instance, render["sample_rate"], dll_path) as runtime:
                from .contracts import ModelBudget
                ModelBudget(document_bytes + render["frames"] * (96 + 8 * runtime.output_count + 4 * sum(v.shape[1] for v in arrays.values())), allowance, 0, 0)
                audio = runtime.render(frames=render["frames"], **arrays)
            if not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) >= 1:
                raise ValueError("Controlled render is empty, nonfinite or clips PCM; adjust authored controls")
            prefix = f"example-{group_index}-{variant_index}"
            metric_files = {}
            if metric_plan is not None:
                for head, document in (("canonical", canonical), ("instance", instance)):
                    metric = measure_metric(document, plan["bounds"]["parameters"], coordinates, metric_plan,
                                            render, arrays, dll_path=dll_path, cancelled=cancelled)
                    name = f"{prefix}-{head}.metric.npz"
                    from .perceptual import save_metric
                    save_metric(staging / name, metric)
                    metric_files[head] = name
            instance_path, wav_path = staging / (prefix + ".json"), staging / (prefix + ".wav")
            atomic_json(instance_path, instance)
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(audio.shape[1])
                wav.setsampwidth(4)
                wav.setframerate(render["sample_rate"])
                wav.writeframes((audio.astype(np.float64) * 2147483648).astype("<i4").tobytes())
            examples.append({"id": variant["id"], "canonical_id": group["id"], "split": group["split"],
                             "perturbation_id": variant["perturbation_id"], "canonical_json": original_path.name,
                             "instance_json": instance_path.name, "normalized_wav": wav_path.name,
                             "normalization": plan["normalization"], "render": render,
                             "canonical_source_sha256": file_digest(source),
                             **({"metric": metric_files} if metric_files else {})})
            if progress:
                progress(len(examples), variants, {})
    manifest = {"schema": "amo.training-manifest/1", "category": "wav_preset_learning",
                "ontology": plan["ontology"], "analysis": plan["analysis"], "schemas": plan["schemas"],
                "learn": plan["learn"], "examples": examples, "require_validation": plan.get("require_validation", True),
                "acceptance": plan.get("acceptance", {}), "producer_plan_sha256": file_digest(plan_path)}
    for key in ("reconstruction_mask_threshold", "reconstruction_acceptance", "parameter_coordinates", "perceptual_metric"):
        if key in plan:
            manifest[key] = plan[key]
    manifest["reconstruction_bounds"] = plan["bounds"]
    atomic_json(staging / "manifest.json", manifest)
    prepare_wav(staging / "manifest.json", extract_features=False)
    if cancelled and cancelled():
        raise TrainingCancelled(f"Dataset publication cancelled; staging directory: {staging}")
    os.rename(staging, destination)
    return {"manifest": str(destination / "manifest.json"), "examples": len(examples)}


def validate_metric_plan(config):
    rank, step, weight = config.get("rank"), config.get("finite_difference_step"), config.get("weight")
    if type(rank) is not int or rank <= 0 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (step, weight)):
        raise ValueError("perceptual_metric requires a positive integer rank, finite_difference_step and weight")
    if config.get("descriptors") is not None and not schema_is(config["descriptors"], "amo.wave-features/1"):
        raise ValueError("perceptual_metric descriptors must be amo.wave-features/1 when given")


def measure_metric(document, bounds, coordinates, config, render, arrays, *, dll_path=None, cancelled=None):
    """Offline J = d phi / d u of one tensor program under one declared performance."""
    from .perceptual import free_coordinates, parameter_metric
    parameters = document["tensors"]["parameters"]
    p_ref = np.asarray(parameters["values"], dtype=np.float64)
    def render_parameters(p):
        candidate = copy.deepcopy(document)
        candidate["tensors"]["parameters"]["values"] = [float(v) for v in p]
        with NativeRuntime(candidate, render["sample_rate"], dll_path) as runtime:
            audio = runtime.render(frames=render["frames"], **arrays)
        if not np.isfinite(audio).all():
            raise ValueError("Perceptual metric candidate render is nonfinite")
        return audio
    descriptors = None
    if config.get("descriptors"):
        from .wave_features import measure
        descriptors = lambda audio: measure(np.asarray(audio, dtype=np.float64).T, render["sample_rate"])
    return parameter_metric(render_parameters, coordinates.to_u(p_ref), free_coordinates(document, bounds),
                            coordinates, step=config["finite_difference_step"], rank=config["rank"],
                            sample_rate=render["sample_rate"], descriptors=descriptors, cancelled=cancelled)


def evaluate_wav(model_path, manifest_path, *, dll_path=None):
    """Evaluate both heads through the native compiler under declared performances.

    Coordinate loss cannot certify waveform/performance reconstruction. This is
    an independent measured acceptance report, not a differentiable DSP loss.
    """
    from .pipeline import infer_wav, export_prediction
    from .dataset import prepare_wav
    path = Path(manifest_path)
    manifest, _, _, audit, _ = prepare_wav(path, extract_features=False)
    rows = [row for row in manifest["examples"] if row.get("split") == "validation"]
    if not rows:
        raise ValueError("Native reconstruction evaluation requires held-out canonical groups")
    if "reconstruction_bounds" not in manifest:
        raise ValueError("Explicit reconstruction_bounds required for all output dimensions")
    results = []
    for row in rows:
        predicted = infer_wav(model_path, path.parent / row["normalized_wav"], mask_threshold=manifest["reconstruction_mask_threshold"])
        render = row["render"]
        arrays = performance(render, manifest["learn"]["available_ram_bytes"])
        metrics = {}
        for head, key in (("primary_canonical", "canonical_json"), ("secondary_rendered_instance", "instance_json")):
            template = json.loads((path.parent / row[key]).read_text(encoding="utf-8"))
            inferred = export_prediction(template, predicted, head=head, bounds=manifest["reconstruction_bounds"])
            with NativeRuntime(template, render["sample_rate"], dll_path) as reference:
                expected = reference.render(frames=render["frames"], **arrays)
            with NativeRuntime(inferred, render["sample_rate"], dll_path) as candidate:
                actual = candidate.render(frames=render["frames"], **arrays)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("Reconstructed native waveform has invalid dimensions or values")
            rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
            if rms <= 1e-12:
                raise ValueError("Silent reference cannot establish relative reconstruction quality")
            metrics[head + "_relative_rmse"] = float(np.sqrt(np.mean((actual.astype(np.float64) - expected) ** 2)) / rms)
        results.append({"id": row["id"], "metrics": metrics})
    return {"schema": "amo.reconstruction-report/1", "model_sha256": file_digest(model_path),
            "dataset_identity": audit["identity"], "examples": results,
            "metrics": {"validation_" + key: max(r["metrics"][key] for r in results) for key in results[0]["metrics"]}}
