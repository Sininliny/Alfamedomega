#!/usr/bin/env python3
"""Train the corpus structure encoder: observed audio features -> AMO construction.

Reads a prepared corpus (``prepare.py build``) and learns, from ``X_audio`` and
the known performance ``conditions`` only, to predict:

- the structure family (``type_id``, 32 classes; a prediction target, never an input),
- the native operation in each of the 8 node slots,
- the presence mask of every padded tensor coordinate,
- the canonical and the rendered-instance values of ``parameters``, ``routing``,
  ``modulation`` and ``assets``.

Flat parameter indices mean different things in different families, so value
targets are normalised per (family, coordinate) and the value heads are
conditioned on the family: the true family while training, the predicted one
at inference. Coordinates that never vary inside a family are decoded from the
family statistics and carry no loss.

Training is resumable: ``last.pt`` is written atomically after every epoch (and
every ``--checkpoint-minutes`` inside an epoch), ``best.pt`` on each validation
improvement, and ``--resume`` continues exactly from the last completed step.
``--time-budget-hours`` stops cleanly before a hosted session limit (Kaggle).

Example::

    python -m training.amo.corpus.train CORPUS_DIR runs/corpus-encoder --epochs 120
    python -m training.amo.corpus.train CORPUS_DIR runs/corpus-encoder --epochs 120 --resume auto
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus.reader import load_split  # noqa: E402
from training.amo.session import atomic_write  # noqa: E402

CHECKPOINT_SCHEMA = "amo.corpus-encoder-checkpoint/1"
MODEL_SCHEMA = "amo.corpus-encoder/1"
VALUE_TENSORS = ("parameters", "routing", "modulation", "assets")
HEADS = ("canonical", "instance")
FEATURE_WIDTH = 32 * 35 + 2  # prepare.features(): duration, 32 windows x 35, log RMS
WINDOWS, WINDOW_WIDTH = 32, 35
MAX_NODES = 8


# --------------------------------------------------------------------------- data


def corpus_keys():
    keys = ["X_audio", "conditions", "type_id", "canonical_node_ops"]
    for head in HEADS:
        for name in VALUE_TENSORS:
            keys += [f"{head}_{name}", f"{head}_{name}_mask"]
    return keys


def corpus_fingerprint(root, protocol):
    root = Path(root)
    digest = hashlib.sha256()
    for name in ("shards.json", "operation-vocabulary.json"):
        digest.update((root / name).read_bytes())
    digest.update(protocol.encode())
    return digest.hexdigest()


class Standardizer:
    """Per-feature z-score with a floor so constant features stay finite."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    @classmethod
    def fit(cls, values, floor=1e-4):
        values = values.astype(np.float64)
        return cls(values.mean(0), np.maximum(values.std(0), floor))

    def __call__(self, values):
        return np.clip((values - self.mean) / self.std, -12.0, 12.0).astype(np.float32)

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class TargetStats:
    """Per (family, coordinate) target transform: optional log, centre, scale.

    A coordinate uses log space when all its values in that family are positive
    and span at least ``log_ratio`` (cutoffs, decay times, delay lengths). It is
    learnable when its spread is non-negligible; otherwise it is decoded as the
    family centre and excluded from the loss.
    """

    def __init__(self, log, center, scale, learnable):
        self.log = np.asarray(log, bool)
        self.center = np.asarray(center, np.float32)
        self.scale = np.asarray(scale, np.float32)
        self.learnable = np.asarray(learnable, bool)

    @classmethod
    def fit(cls, type_id, values_by_head, masks_by_head, types, log_ratio=8.0):
        width = values_by_head[0].shape[1]
        log = np.zeros((types, width), bool)
        center = np.zeros((types, width), np.float32)
        scale = np.ones((types, width), np.float32)
        learnable = np.zeros((types, width), bool)
        for t in range(types):
            rows = type_id == t
            if not rows.any():
                continue
            values = np.concatenate([v[rows] for v in values_by_head]).astype(np.float64)
            masks = np.concatenate([m[rows] for m in masks_by_head])
            for w in range(width):
                column = values[masks[:, w], w]
                if column.size == 0:
                    continue
                low, high = column.min(), column.max()
                use_log = low > 0 and high / low >= log_ratio
                if use_log:
                    column = np.log(column)
                spread = column.std()
                log[t, w] = use_log
                center[t, w] = column.mean()
                tolerance = 1e-6 * max(1.0, abs(column.mean()))
                learnable[t, w] = spread > tolerance
                scale[t, w] = max(spread, tolerance)
        return cls(log, center, scale, learnable)

    def encode(self, type_id, values, mask):
        log = self.log[type_id]
        safe = np.where(log & mask, np.maximum(values, 1e-30), 1.0)
        transformed = np.where(log, np.log(safe), values)
        z = (transformed - self.center[type_id]) / self.scale[type_id]
        return np.where(mask, z, 0.0).astype(np.float32)

    def weight(self, type_id, mask):
        return (mask & self.learnable[type_id]).astype(np.float32)

    def decode(self, type_id, z):
        transformed = z * self.scale[type_id] + self.center[type_id]
        transformed = np.where(self.learnable[type_id], transformed, self.center[type_id])
        return np.where(self.log[type_id], np.exp(np.clip(transformed, -80, 80)), transformed)

    def state(self):
        return {"log": self.log.tolist(), "center": self.center.tolist(),
                "scale": self.scale.tolist(), "learnable": self.learnable.tolist()}


def fit_statistics(train, types):
    stats = {"X_audio": Standardizer.fit(train["X_audio"]),
             "conditions": Standardizer.fit(train["conditions"])}
    for name in VALUE_TENSORS:
        stats[name] = TargetStats.fit(
            train["type_id"],
            [train[f"{head}_{name}"] for head in HEADS],
            [train[f"{head}_{name}_mask"] for head in HEADS],
            types,
        )
    return stats


def load_perceptual_weights(path, types, widths):
    """Per (family, coordinate) value-loss weights written by ``sensitivity.py``."""
    from training.amo.corpus.sensitivity import SCHEMA
    with np.load(path) as stored:
        if str(stored["schema"]) != SCHEMA:
            raise ValueError(f"{path} is not an {SCHEMA} file")
        weights = {name: stored[f"weight_{name}"].astype(np.float32) for name in VALUE_TENSORS}
    for name in VALUE_TENSORS:
        if weights[name].shape != (types, widths[name]):
            raise ValueError(f"{path}: weight_{name} has shape {weights[name].shape},"
                             f" this corpus needs {(types, widths[name])}")
    return weights


def encode_split(raw, stats, perceptual=None):
    """Model-ready float32/int64 arrays; all normalisation happens once here.

    ``perceptual`` (from ``load_perceptual_weights``) scales each value-loss weight
    by how audible that (family, coordinate) is; None keeps uniform weights.
    """
    type_id = raw["type_id"].astype(np.int64)
    out = {
        "x": stats["X_audio"](raw["X_audio"]),
        "conditions": stats["conditions"](raw["conditions"]),
        "type_id": type_id,
        "node_ops": raw["canonical_node_ops"].astype(np.int64),
    }
    presence = []
    for name in VALUE_TENSORS:
        presence.append(raw[f"canonical_{name}_mask"])
        for head in HEADS:
            mask = raw[f"{head}_{name}_mask"]
            out[f"{head}_{name}_z"] = stats[name].encode(type_id, raw[f"{head}_{name}"], mask)
            weight = stats[name].weight(type_id, mask)
            if perceptual is not None:
                weight = weight * perceptual[name][type_id]
            out[f"{head}_{name}_w"] = weight
    out["presence"] = np.concatenate(presence, axis=1).astype(np.float32)
    return out


# -------------------------------------------------------------------------- model


def build_model(config, widths, types, operations):
    import torch
    from torch import nn

    class CorpusEncoder(nn.Module):
        """Transformer over 32 time windows of (output, FX input) features."""

        def __init__(self):
            super().__init__()
            d = config["width"]
            self.widths = dict(widths)
            self.types = types
            self.operations = operations
            self.window = nn.Linear(2 * WINDOW_WIDTH, d)
            self.globals = nn.Linear(4 + config["conditions"], d)
            self.position = nn.Parameter(torch.zeros(1, WINDOWS + 1, d))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d, config["heads"], config["ff"], config["dropout"],
                activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, config["layers"], enable_nested_tensor=False)
            self.pool = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU())
            self.type_head = nn.Linear(d, types)
            self.ops_head = nn.Linear(d, MAX_NODES * (operations + 1))
            total = sum(widths.values())
            self.presence_head = nn.Linear(d, total)
            self.type_embedding = nn.Embedding(types, d)
            self.value_trunk = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                             nn.Linear(d, d), nn.GELU())
            self.value_head = nn.Linear(d, len(HEADS) * total)

        def tokens(self, x, conditions):
            out, inp = x[:, :FEATURE_WIDTH], x[:, FEATURE_WIDTH:]
            windows = torch.cat([
                out[:, 1:-1].reshape(-1, WINDOWS, WINDOW_WIDTH),
                inp[:, 1:-1].reshape(-1, WINDOWS, WINDOW_WIDTH)], dim=2)
            scalars = torch.cat([out[:, :1], out[:, -1:], inp[:, :1], inp[:, -1:], conditions], dim=1)
            sequence = torch.cat([self.globals(scalars)[:, None], self.window(windows)], dim=1)
            encoded = self.encoder(sequence + self.position)
            return self.pool(torch.cat([encoded[:, 0], encoded[:, 1:].mean(1)], dim=1))

        def forward(self, x, conditions, type_id=None):
            h = self.tokens(x, conditions)
            type_logits = self.type_head(h)
            if type_id is None:
                type_id = type_logits.argmax(1)
            values = self.value_head(self.value_trunk(h + self.type_embedding(type_id)))
            return {
                "type": type_logits,
                "ops": self.ops_head(h).view(-1, MAX_NODES, operations + 1),
                "presence": self.presence_head(h),
                "values": values.view(-1, len(HEADS), sum(self.widths.values())),
                "type_used": type_id,
            }

    return CorpusEncoder()


def split_values(tensor, widths):
    result, offset = {}, 0
    for name in VALUE_TENSORS:
        result[name] = tensor[..., offset:offset + widths[name]]
        offset += widths[name]
    return result


def compute_losses(output, batch, widths, weights):
    import torch
    import torch.nn.functional as F

    losses = {
        "type": F.cross_entropy(output["type"].float(), batch["type_id"]),
        "ops": F.cross_entropy(output["ops"].float().flatten(0, 1), batch["node_ops"].flatten()),
        "presence": F.binary_cross_entropy_with_logits(output["presence"].float(), batch["presence"]),
    }
    for hi, head in enumerate(HEADS):
        predicted = split_values(output["values"][:, hi].float(), widths)
        total, count = 0.0, 0.0
        for name in VALUE_TENSORS:
            w = batch[f"{head}_{name}_w"]
            error = F.smooth_l1_loss(predicted[name], batch[f"{head}_{name}_z"], reduction="none", beta=0.5)
            total = total + (error * w).sum()
            count = count + w.sum()
        losses[head] = total / torch.clamp(count, min=1.0)
    losses["total"] = sum(weights[k] * losses[k] for k in ("type", "ops", "presence", "canonical", "instance"))
    return losses


# ---------------------------------------------------------------------- utilities


def format_seconds(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def log(message, path=None):
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def learning_rate(step, total, peak, warmup, floor=0.05):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def find_resume(output, resume, search):
    if resume in (None, "", "none"):
        return None
    if resume != "auto":
        path = Path(resume)
        if not path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {path}")
        return path
    candidates = [output / "last.pt"]
    for directory in search:
        if Path(directory).is_dir():
            candidates += sorted(Path(directory).rglob("last.pt"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        return None

    def progress(path):
        try:
            with open(path.with_name("progress.json"), encoding="utf-8") as stream:
                state = json.load(stream)
            return (state["global_step"], path.stat().st_mtime)
        except (OSError, ValueError, KeyError):
            return (-1, path.stat().st_mtime)

    return max(candidates, key=progress)


def capture_rng():
    import torch
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError:
            pass  # Different GPU count: dropout noise differs, weights/optimizer are exact.


# ----------------------------------------------------------------------- training


def to_device(arrays, device):
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arrays.items()}


def take(data, index):
    return {k: v[index] for k, v in data.items()}


def evaluate(model, data, widths, weights, batch_size, amp_dtype):
    """Validation losses plus decoded metrics, with the true and the predicted family."""
    import torch

    model.eval()
    count = data["type_id"].shape[0]
    sums = {}
    stats = {"type_correct": 0, "ops_slot_correct": 0, "ops_graph_correct": 0,
             "presence_correct": 0, "presence_total": 0}
    mae = {f"{h}_{n}": [0.0, 0.0] for h in HEADS for n in VALUE_TENSORS}
    e2e = [0.0, 0.0]
    with torch.no_grad():
        for start in range(0, count, batch_size):
            batch = take(data, slice(start, start + batch_size))
            n = batch["type_id"].shape[0]
            with torch.autocast(batch["x"].device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = model(batch["x"], batch["conditions"], batch["type_id"])
                losses = compute_losses(output, batch, widths, weights)
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value) * n
            predicted_type = output["type"].argmax(1)
            stats["type_correct"] += int((predicted_type == batch["type_id"]).sum())
            ops = output["ops"].argmax(2) == batch["node_ops"]
            stats["ops_slot_correct"] += int(ops.sum())
            stats["ops_graph_correct"] += int(ops.all(1).sum())
            presence = (output["presence"] > 0) == (batch["presence"] > 0.5)
            stats["presence_correct"] += int(presence.sum())
            stats["presence_total"] += presence.numel()
            for hi, head in enumerate(HEADS):
                predicted = split_values(output["values"][:, hi].float(), widths)
                for name in VALUE_TENSORS:
                    w = batch[f"{head}_{name}_w"]
                    error = (predicted[name] - batch[f"{head}_{name}_z"]).abs() * w
                    mae[f"{head}_{name}"][0] += float(error.sum())
                    mae[f"{head}_{name}"][1] += float(w.sum())
            # End to end: values decoded with the model's own family prediction.
            # A wrong family puts the parameters in the wrong coordinate frame;
            # those rows count with the error the value head then makes.
            with torch.autocast(batch["x"].device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                free = model(batch["x"], batch["conditions"], None)
            predicted = split_values(free["values"][:, 0].float(), widths)
            right = (free["type_used"] == batch["type_id"]).float()[:, None]
            for name in VALUE_TENSORS:
                w = batch[f"canonical_{name}_w"]
                error = (predicted[name] - batch[f"canonical_{name}_z"]).abs()
                # Misclassified rows: compare against a zero (family-mean) guess at best.
                error = right * error + (1 - right) * torch.maximum(error, batch[f"canonical_{name}_z"].abs())
                e2e[0] += float((error * w).sum())
                e2e[1] += float(w.sum())
    model.train()
    metrics = {f"val_{k}": v / count for k, v in sums.items()}
    metrics["val_type_acc"] = stats["type_correct"] / count
    metrics["val_ops_slot_acc"] = stats["ops_slot_correct"] / (count * MAX_NODES)
    metrics["val_ops_graph_acc"] = stats["ops_graph_correct"] / count
    metrics["val_presence_acc"] = stats["presence_correct"] / max(1, stats["presence_total"])
    for key, (total, weight) in mae.items():
        metrics[f"val_mae_z_{key}"] = total / max(weight, 1.0)
    metrics["val_mae_z_canonical_e2e"] = e2e[0] / max(e2e[1], 1.0)
    return metrics


def save_checkpoint(path, package):
    import torch
    atomic_write(path, lambda stream: torch.save(package, stream))


def gpu_memory():
    import torch
    if not torch.cuda.is_available():
        return ""
    return f" | gpu {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB"


def train(args):
    import torch

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    root = Path(args.corpus)
    vocabulary = json.loads((root / "operation-vocabulary.json").read_text())
    operations = max(vocabulary.values())
    types = args.types
    identity = {"corpus": corpus_fingerprint(root, args.protocol), "protocol": args.protocol,
                "model": {k: getattr(args, k) for k in ("width", "layers", "heads", "ff", "dropout")},
                "types": types, "operations": operations}
    if args.perceptual_weights:
        identity["perceptual_weights"] = hashlib.sha256(Path(args.perceptual_weights).read_bytes()).hexdigest()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = None
    if device.type == "cuda" and args.amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    log(f"device {device}{' (' + torch.cuda.get_device_name(device) + ')' if device.type == 'cuda' else ''}"
        f" | torch {torch.__version__} | amp {amp_dtype} | corpus {root}", log_path)
    began = time.monotonic()
    raw_train = load_split(root, "train", args.protocol, corpus_keys())
    raw_val = load_split(root, "validation", args.protocol, corpus_keys())
    if args.limit:
        raw_train = {k: v[:args.limit] for k, v in raw_train.items()}
        raw_val = {k: v[:max(64, args.limit // 6)] for k, v in raw_val.items()}
    log(f"loaded train {len(raw_train['type_id'])} / validation {len(raw_val['type_id'])} examples"
        f" in {time.monotonic() - began:.1f}s; families {len(np.unique(raw_train['type_id']))},"
        f" operations {operations}", log_path)

    resume = find_resume(output, args.resume, args.resume_search)
    package = None
    if resume:
        package = torch.load(resume, map_location="cpu", weights_only=False)
        if package.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"{resume} is not a corpus-encoder checkpoint")
        if package["identity"] != identity:
            raise ValueError("Checkpoint corpus/protocol/model differ from this run; use a new output directory\n"
                             f"checkpoint: {package['identity']}\nthis run:   {identity}")
        stats = {"X_audio": Standardizer(**package["stats"]["X_audio"]),
                 "conditions": Standardizer(**package["stats"]["conditions"])}
        for name in VALUE_TENSORS:
            stats[name] = TargetStats(**package["stats"][name])
        log(f"resuming from {resume} at epoch {package['epoch']} step {package['step_in_epoch']}"
            f" (global step {package['global_step']})", log_path)
    else:
        stats = fit_statistics(raw_train, types)
    widths = {name: raw_train[f"canonical_{name}"].shape[1] for name in VALUE_TENSORS}
    learnable = {name: int(stats[name].learnable.sum()) for name in VALUE_TENSORS}
    log(f"value widths {widths}; learnable (family, coordinate) pairs {learnable}", log_path)

    perceptual = None
    if args.perceptual_weights:
        perceptual = load_perceptual_weights(args.perceptual_weights, types, widths)
        used = np.concatenate([perceptual[n][stats[n].learnable] for n in VALUE_TENSORS])
        log(f"perceptual value weights {args.perceptual_weights}: learnable pairs {used.size},"
            f" min {used.min():.3f} median {np.median(used):.3f} max {used.max():.3f}", log_path)
    train_data = to_device(encode_split(raw_train, stats, perceptual), device if args.data_on_device else "cpu")
    val_data = to_device(encode_split(raw_val, stats, perceptual), device if args.data_on_device else "cpu")
    del raw_train, raw_val
    if not args.data_on_device:
        train_data = {k: v.pin_memory() if device.type == "cuda" else v for k, v in train_data.items()}

    config = {"width": args.width, "layers": args.layers, "heads": args.heads, "ff": args.ff,
              "dropout": args.dropout, "conditions": int(train_data["conditions"].shape[1])}
    model = build_model(config, widths, types, operations).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    log(f"model {parameters / 1e6:.2f}M parameters {config}", log_path)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "position" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "position" not in n)]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    else:  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    weights = {"type": args.type_weight, "ops": args.ops_weight, "presence": args.presence_weight,
               "canonical": args.canonical_weight, "instance": args.instance_weight}

    count = int(train_data["type_id"].shape[0])
    steps_per_epoch = math.ceil(count / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, max(1, total_steps // 10))
    epoch, step_in_epoch, global_step, best, history = 0, 0, 0, math.inf, []
    if package:
        model.load_state_dict(package["model"])
        optimizer.load_state_dict(package["optimizer"])
        if package.get("scaler"):
            scaler.load_state_dict(package["scaler"])
        epoch, step_in_epoch, global_step = package["epoch"], package["step_in_epoch"], package["global_step"]
        best, history = package["best"], package["history"]
        restore_rng(package["rng"])
        if epoch >= args.epochs:
            log(f"checkpoint already finished {epoch} epochs; raise --epochs to continue", log_path)
    log(f"{count} train examples, batch {args.batch_size}, {steps_per_epoch} steps/epoch,"
        f" {args.epochs} epochs = {total_steps} steps (warmup {warmup})", log_path)

    def checkpoint(name, epoch_value, step_value):
        package = {"schema": CHECKPOINT_SCHEMA, "identity": identity, "config": config, "widths": widths,
                   "epoch": epoch_value, "step_in_epoch": step_value, "global_step": global_step,
                   "best": best, "history": history, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "rng": capture_rng(), "args": vars(args),
                   "stats": {k: v.state() for k, v in stats.items()}}
        save_checkpoint(output / name, package)
        if name == "last.pt":
            (output / "progress.json").write_text(json.dumps(
                {"epoch": epoch_value, "step_in_epoch": step_value, "global_step": global_step,
                 "epochs": args.epochs, "best_val": best}, indent=2))

    run_start = time.monotonic()
    steps_this_run = 0
    budget = args.time_budget_hours * 3600 if args.time_budget_hours else math.inf
    last_save = time.monotonic()
    epoch_seconds = []
    stopped_early = False
    model.train()
    while epoch < args.epochs:
        generator = torch.Generator().manual_seed(args.seed * 100003 + epoch)
        order = torch.randperm(count, generator=generator)
        epoch_start = time.monotonic()
        window = {"loss": 0.0, "n": 0, "samples": 0, "t": time.monotonic(), "grad": 0.0}
        sums = {}
        seen = 0
        for step in range(step_in_epoch, steps_per_epoch):
            index = order[step * args.batch_size:(step + 1) * args.batch_size]
            batch = take(train_data, index.to(train_data["x"].device))
            if batch["x"].device != device:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            lr = learning_rate(global_step, total_steps, args.lr, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output_ = model(batch["x"], batch["conditions"], batch["type_id"])
                losses = compute_losses(output_, batch, widths, weights)
            if not torch.isfinite(losses["total"]):
                checkpoint("nonfinite.pt", epoch, step)
                raise RuntimeError(f"Nonfinite loss at epoch {epoch} step {step}: "
                                   f"{ {k: float(v) for k, v in losses.items()} }")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).detach()
            scaler.step(optimizer)
            scaler.update()
            n = int(index.shape[0])
            global_step += 1
            steps_this_run += 1
            seen += n
            # Accumulate on the device; reading values back every step would stall the GPU.
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + value.detach() * n
            window["loss"] += losses["total"].detach()
            window["grad"] += torch.nan_to_num(grad, nan=0.0, posinf=0.0)
            window["n"] += 1
            window["samples"] += n
            if (step + 1) % args.log_every == 0 or step + 1 == steps_per_epoch:
                elapsed = time.monotonic() - window["t"]
                rate = window["samples"] / max(elapsed, 1e-9)
                per_step = (time.monotonic() - epoch_start) / max(1, step + 1 - step_in_epoch)
                epoch_eta = per_step * (steps_per_epoch - step - 1)
                remaining = (args.epochs - epoch - 1) * (np.mean(epoch_seconds) if epoch_seconds
                                                          else per_step * steps_per_epoch)
                parts = " ".join(f"{k} {float(v):.4f}" for k, v in losses.items() if k != "total")
                log(f"ep {epoch + 1}/{args.epochs} step {step + 1}/{steps_per_epoch}"
                    f" | loss {float(window['loss']) / window['n']:.4f} ({parts})"
                    f" | lr {lr:.2e} | grad {float(window['grad']) / window['n']:.3f}"
                    f" | {rate:,.0f} samples/s | epoch ETA {format_seconds(epoch_eta)}"
                    f" | total ETA {format_seconds(epoch_eta + remaining)}{gpu_memory()}", log_path)
                window = {"loss": 0.0, "n": 0, "samples": 0, "t": time.monotonic(), "grad": 0.0}
            if args.checkpoint_minutes and time.monotonic() - last_save > args.checkpoint_minutes * 60 \
                    and step + 1 < steps_per_epoch:
                checkpoint("last.pt", epoch, step + 1)
                last_save = time.monotonic()
                log(f"checkpoint last.pt (mid-epoch {epoch + 1}, step {step + 1})", log_path)
            if time.monotonic() - run_start > budget:
                checkpoint("last.pt", epoch, step + 1)
                log(f"time budget {args.time_budget_hours}h reached mid-epoch; saved last.pt,"
                    f" continue with --resume auto", log_path)
                return 2
            if args.max_steps and steps_this_run >= args.max_steps:
                checkpoint("last.pt", epoch, step + 1)
                log(f"--max-steps {args.max_steps} reached; saved last.pt", log_path)
                return 2
        train_seconds = time.monotonic() - epoch_start
        metrics = {f"train_{k}": float(v) / max(seen, 1) for k, v in sums.items()}
        val_start = time.monotonic()
        metrics.update(evaluate(model, val_data, widths, weights, args.eval_batch_size, amp_dtype))
        epoch += 1
        step_in_epoch = 0
        metrics.update({"epoch": epoch, "global_step": global_step, "lr": lr,
                        "epoch_seconds": time.monotonic() - epoch_start,
                        "train_samples_per_second": seen / max(train_seconds, 1e-9),
                        "validation_seconds": time.monotonic() - val_start})
        epoch_seconds.append(metrics["epoch_seconds"])
        history.append(metrics)
        improved = metrics["val_total"] < best
        if improved:
            best = metrics["val_total"]
        with open(output / "metrics.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        checkpoint("last.pt", epoch, 0)
        last_save = time.monotonic()
        if improved:
            checkpoint("best.pt", epoch, 0)
        if args.keep_every and epoch % args.keep_every == 0:
            checkpoint(f"epoch-{epoch:04d}.pt", epoch, 0)
        remaining = (args.epochs - epoch) * float(np.mean(epoch_seconds[-5:]))
        log(f"== epoch {epoch}/{args.epochs} done in {format_seconds(metrics['epoch_seconds'])}"
            f" | train {metrics['train_total']:.4f} | val {metrics['val_total']:.4f}"
            f"{' *best*' if improved else ''} (best {best:.4f})"
            f" | type acc {metrics['val_type_acc']:.3f} | ops acc slot {metrics['val_ops_slot_acc']:.3f}"
            f" graph {metrics['val_ops_graph_acc']:.3f} | presence {metrics['val_presence_acc']:.4f}"
            f" | MAE(z) canonical {metrics['val_canonical']:.4f} instance {metrics['val_instance']:.4f}"
            f" e2e {metrics['val_mae_z_canonical_e2e']:.4f}"
            f" | {metrics['train_samples_per_second']:,.0f} samples/s | ETA {format_seconds(remaining)}"
            f"{gpu_memory()} | saved last.pt{' best.pt' if improved else ''}", log_path)
        log("   per-tensor val MAE(z): " + ", ".join(
            f"{k[10:]} {v:.3f}" for k, v in metrics.items() if k.startswith("val_mae_z_") and "e2e" not in k),
            log_path)
        if args.patience:
            best_epoch = min(history, key=lambda row: row["val_total"])["epoch"]
            if epoch - best_epoch >= args.patience and epoch < args.epochs:
                log(f"early stop: val has not improved for {epoch - best_epoch} epochs"
                    f" (best {best:.4f} at epoch {best_epoch})", log_path)
                break
        if time.monotonic() - run_start + float(np.mean(epoch_seconds[-3:])) > budget and epoch < args.epochs:
            log(f"time budget {args.time_budget_hours}h would be exceeded by the next epoch;"
                f" stopping at epoch {epoch}, continue with --resume auto", log_path)
            stopped_early = True
            break
    if stopped_early:
        return 2
    export(output, model, config, widths, stats, vocabulary, types, history, best, log_path)
    return 0


def export(output, model, config, widths, stats, vocabulary, types, history, best, log_path):
    """Inference package: best weights plus everything needed to decode raw tensors."""
    import torch
    best_path = output / "best.pt"
    source = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None
    package = {"schema": MODEL_SCHEMA, "config": config, "widths": widths, "types": types,
               "operation_vocabulary": vocabulary, "heads": list(HEADS), "value_tensors": list(VALUE_TENSORS),
               "model": source["model"] if source else model.state_dict(),
               "stats": {k: v.state() for k, v in stats.items()},
               "epoch": source["epoch"] if source else len(history),
               "validation": next((h for h in history if h["val_total"] == best), history[-1] if history else {})}
    save_checkpoint(output / "model.pt", package)
    log(f"exported model.pt from epoch {package['epoch']} (val {best:.4f})", log_path)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path, help="prepared corpus directory (contains shards.json)")
    parser.add_argument("output", type=Path, help="run directory for checkpoints, logs and model.pt")
    parser.add_argument("--protocol", choices=("originals", "unseen_topology"), default="originals")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--types", type=int, default=32)
    parser.add_argument("--type-weight", type=float, default=1.0)
    parser.add_argument("--ops-weight", type=float, default=0.5)
    parser.add_argument("--presence-weight", type=float, default=0.2)
    parser.add_argument("--canonical-weight", type=float, default=2.0)
    parser.add_argument("--instance-weight", type=float, default=1.0)
    parser.add_argument("--perceptual-weights", type=Path, default=None,
                        help="sensitivity.npz from sensitivity.py: weight the value loss by audibility"
                             " (validation loss is weighted the same way; evaluate.py stays unweighted)")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:N")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--data-on-device", action=argparse.BooleanOptionalAction, default=True,
                        help="keep the encoded corpus in GPU memory (about 2 GiB for the full corpus)")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--resume", default=None,
                        help="'auto' (OUTPUT/last.pt, else newest last.pt under --resume-search) or a path")
    parser.add_argument("--resume-search", nargs="*", default=["/kaggle/input"])
    parser.add_argument("--checkpoint-minutes", type=float, default=20.0,
                        help="also save last.pt inside an epoch this often (0 disables)")
    parser.add_argument("--patience", type=int, default=0,
                        help="stop and export once val_total has not improved for N epochs (0 = off)")
    parser.add_argument("--keep-every", type=int, default=0, help="keep epoch-NNNN.pt every N epochs")
    parser.add_argument("--time-budget-hours", type=float, default=0.0,
                        help="stop cleanly (exit 2) before this wall time, e.g. 11.5 on Kaggle")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N steps in this run (smoke tests)")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N training rows (smoke tests)")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.width % args.heads:
        parser.error("--width must be divisible by --heads")
    return args


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    return train(parse(argv))


if __name__ == "__main__":
    raise SystemExit(main())
