"""Reproducible, cancellable optimizer sessions; no training starts on import."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from .schema import schema_is


class TrainingCancelled(RuntimeError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    data = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda stream: stream.write(data))


def protect_outputs(destination, sources, session=None):
    source_paths = {Path(path).resolve() for path in sources}
    destination = Path(destination).resolve()
    if destination.is_dir():
        raise ValueError("Select a model output file, not a directory")
    if destination in source_paths:
        raise ValueError("Model output cannot overwrite a training source")
    if session and session.path and (session.path.resolve() == destination or session.path.resolve() in source_paths):
        raise ValueError("Checkpoint must be separate from published model and training sources")


def inference_metadata(metadata):
    """Inference artifacts keep a summary, not optimizer history or source lists."""
    result = dict(metadata)
    losses = result.pop("losses", None)
    metrics = result.pop("metrics", None)
    if losses:
        result["training_summary"] = {"completed_steps": len(losses), "initial": losses[0],
                                       "final": metrics[-1] if metrics else losses[-1]}
    if "dataset" in result:
        dataset = dict(result["dataset"])
        sources = dataset.pop("sources", None)
        if sources is not None:
            dataset["sources_sha256"] = fingerprint(sources)
        result["dataset"] = dataset
    return result


class TrainingSession:
    """Checkpoint completed optimizer steps with identity, Adam and RNG state.

    Checkpoints are private training state, never inference packages. A changed
    dataset/objective/device contract is refused instead of silently fine-tuning.
    Total steps may increase. Cancellation preserves the last completed step.
    """
    def __init__(self, identity, *, checkpoint=None, resume=False, every=100,
                 progress=None, cancelled=None):
        if type(every) is not int or every < 1:
            raise ValueError("Checkpoint interval must be a positive integer")
        if resume and checkpoint is None:
            raise ValueError("Resume requires an explicit checkpoint")
        self.identity = fingerprint(identity)
        self.producer_identity = self.identity
        self.path = Path(checkpoint) if checkpoint else None
        if self.path and not resume and self.path.exists():
            raise ValueError("Checkpoint exists; choose Resume or a new output path")
        if resume and not self.path.is_file():
            raise ValueError("Resume checkpoint does not exist or is not a file")
        self.resume = resume
        self.every = every
        self.progress = progress
        self.cancelled = cancelled or (lambda: False)
        self.step = 0
        self.history = []

    def bind(self, objective):
        self.identity = fingerprint({"producer": self.producer_identity, "objective": objective})

    def restore(self, model, optimizer, *, available_ram_bytes):
        import torch
        from .contracts import ModelBudget
        if not self.resume:
            if self.path and self.path.exists():
                raise ValueError("Checkpoint exists; choose Resume or a new output path")
            return 0
        ModelBudget(self.path.stat().st_size * 4, available_ram_bytes, 0, 0)
        saved = torch.load(self.path, map_location="cpu", weights_only=True)
        if not schema_is(saved.get("schema"), "amo.training-checkpoint/1") or saved.get("identity") != self.identity:
            raise ValueError("Checkpoint dataset/model/objective identity differs")
        if type(saved.get("step")) is not int or saved["step"] < 0 or len(saved["history"]) != saved["step"]:
            raise ValueError("Invalid checkpoint progress")
        if any(not isinstance(row, dict) or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row.values()) for row in saved["history"]):
            raise ValueError("Invalid checkpoint metrics")
        model.load_state_dict(saved["model"])
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Checkpoint contains nonfinite model weights")
        optimizer.load_state_dict(saved["optimizer"])
        if any(not torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)):
            raise ValueError("Checkpoint contains nonfinite optimizer state")
        torch.set_rng_state(saved["rng"])
        if saved.get("cuda_rng"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA checkpoint requires its declared CUDA device")
            if isinstance(saved["cuda_rng"], dict):
                devices = {str(p.device) for p in model.parameters() if p.is_cuda}
                if set(saved["cuda_rng"]) != devices:
                    raise ValueError("Checkpoint CUDA random state differs from the selected device")
                for device, state in saved["cuda_rng"].items():
                    torch.cuda.set_rng_state(state, device=device)
            else:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        self.step, self.history = saved["step"], saved["history"]
        return self.step

    def save(self, model, optimizer):
        import torch
        if self.path:
            devices = {str(p.device) for p in model.parameters() if p.is_cuda}
            package = {"schema": "amo.training-checkpoint/1", "identity": self.identity,
                       "step": self.step, "history": self.history,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": torch.get_rng_state(),
                       "cuda_rng": {device: torch.cuda.get_rng_state(device) for device in devices}}
            atomic_write(self.path, lambda stream: torch.save(package, stream))

    def check_cancel(self, model, optimizer):
        if self.cancelled():
            self.save(model, optimizer)
            raise TrainingCancelled("Cancelled; completed optimizer steps are checkpointed")

    def complete(self, metrics, model, optimizer, total):
        import torch
        if any(not math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Nonfinite training metrics; artifact was not published")
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite model weights; artifact was not published")
        self.step += 1
        self.history.append(metrics)
        if self.path and (self.step % self.every == 0 or self.step == total):
            self.save(model, optimizer)
        if self.progress:
            self.progress(self.step, total, metrics)


def device_for_training(device, estimate, available_ram_bytes, available_vram_bytes=0, *, estimated_vram_bytes=None):
    import torch
    from .hardware import detect
    from .contracts import ModelBudget
    if not isinstance(device, str) or device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("Device must be cpu or an explicit cuda:N")
    index = None if device == "cpu" else int(device[5:])
    capacity = detect(gpu_device=index)
    ModelBudget(estimate, min(available_ram_bytes, capacity.available_ram_bytes),
                (estimate if estimated_vram_bytes is None else estimated_vram_bytes) if index is not None else 0,
                min(available_vram_bytes, capacity.available_vram_bytes))
    return torch.device(device)


def optimize(model, objective, *, steps, learning_rate, available_ram_bytes,
             estimate, identity, session=None, device="cpu", available_vram_bytes=0,
             gradient_clip=1.0, validation=None):
    """Shared lifecycle for runtime, preference and importance objectives."""
    import torch
    if type(steps) is not int or steps <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Positive finite training settings required")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("Positive finite gradient clipping required")
    selected = device_for_training(device, estimate, available_ram_bytes, available_vram_bytes)
    model.to(selected).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    session = session or TrainingSession(identity)
    session.bind({"data": identity, "device": device, "learning_rate": learning_rate,
                  "gradient_clip": gradient_clip})
    start = session.restore(model, optimizer, available_ram_bytes=available_ram_bytes)
    if start > steps:
        raise ValueError("Checkpoint is beyond requested total steps")
    for _ in range(start, steps):
        session.check_cancel(model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(selected)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        metrics = {"total": float(loss.detach())}
        if validation:
            with torch.no_grad():
                metrics["validation_total"] = float(validation(selected))
        session.complete(metrics, model, optimizer, steps)
    session.check_cancel(model, optimizer)
    model.cpu().eval()
    return {"losses": [m["total"] for m in session.history], "metrics": session.history,
            "estimated_peak_training_bytes": estimate}
