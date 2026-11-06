#!/usr/bin/env python3
"""Train the mix -> multi-instrument preset model (MIX-DESIGN.md §4).

Input: the Mid/Side log-band spectrogram of a mixed phrase (``mixscene`` corpus)
plus a count token (0 = auto, 1..8 = the user's instrument count). Output: 8 slots,
each an instrument preset (two-layer construction, insert FX, note echo), its gain,
its notes (piano roll) and an auxiliary stem; a bus FX preset; and, for the auto
token, the instrument count.

Training picks, per example, either the auto token with the L0 target (MDL: the
fewest-instrument exact explanation) or a ladder level and its instrument count as
the token. Slots are matched to target units with the Hungarian algorithm.

Resumable like ``train.py``: ``last.pt`` + ``progress.json`` every epoch and every
``--checkpoint-minutes``, ``best.pt`` on validation improvement, ``--resume auto``,
``--time-budget-hours`` (exit code 2 = stopped early with ``last.pt`` saved).

Example::

    python -m training.amo.corpus.mixtrain train MIX_CORPUS runs/mix --epochs 40
    python -m training.amo.corpus.mixtrain evaluate MIX_CORPUS runs/mix/model.pt --resynth 32
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus import mixscene, mixtargets as T  # noqa: E402
from training.amo.corpus.train import (  # noqa: E402
    capture_rng, find_resume, format_seconds, learning_rate, log, restore_rng, save_checkpoint)
from training.amo.session import atomic_write  # noqa: E402

MODEL_SCHEMA = "amo.mix-decomposer/1"
CHECKPOINT_SCHEMA = "amo.mix-decomposer-checkpoint/1"
SLOTS = mixscene.MAX_INSTRUMENTS
FEATURES = mixscene.MID_BANDS + mixscene.SIDE_BANDS
FRAMES = mixscene.MAX_FRAMES
STEM_FRAMES = mixscene.MAX_STEM_FRAMES
STEM_FLOOR = -6.0  # log10 power below which stems are not supervised in detail
N_FAMILIES = len(T.FAMILIES)
HEADS = {  # name -> width of the per-slot output
    "exists": 1, "family1": N_FAMILIES, "u1": N_FAMILIES * T.U_DIMS, "family2": N_FAMILIES + 1,
    "u2": N_FAMILIES * T.U_DIMS, "transpose2": len(T.LAYER_TRANSPOSE),
    "insert_present": len(T.INSERT_FX), "insert_values": T.INSERT_WIDTH,
    "echo_present": 1, "echo_taps": 3, "echo_transpose": len(T.ECHO_TRANSPOSE),
    "echo_values": len(T.ECHO_PARAMS), "gain": 1,
}
ROLL = len(T.ROLL_CHANNELS)


# --------------------------------------------------------------------------- data


def load_corpus(root):
    """All scenes of a mixscene corpus: features in memory (fp16), specs as dicts."""
    root = Path(root)
    mixes, frames, stems, levels, scenes = [], [], [], [], []
    for shard in sorted((root / "shards").iterdir()):
        if not (shard / "complete.json").exists():
            continue
        data = np.load(shard / "scenes.npz")
        mixes.append(data["mix"])
        frames.append(data["frames"])
        stems.append(data["stems"])
        levels.append(data["atom_levels"])
        with open(shard / "scenes.jsonl", encoding="utf-8") as stream:
            scenes += [json.loads(line) for line in stream]
    if not scenes:
        raise SystemExit(f"no complete shards under {root / 'shards'}")
    return {"mix": np.concatenate(mixes), "frames": np.concatenate(frames),
            "stems": np.concatenate(stems), "atom_levels": np.concatenate(levels), "scenes": scenes}


def split_indices(corpus, split):
    return [i for i, s in enumerate(corpus["scenes"]) if s["split"] == split]


def example(corpus, index, rng, auto_probability):
    """Inputs and targets of one scene with a sampled count token and ladder level."""
    scene = corpus["scenes"][index]
    if rng.random() < auto_probability:
        level, token = "L0", 0
    else:
        level = T.LEVELS[int(rng.integers(3))]
        token = None
    targets = T.scene_targets(scene, level)
    if token is None:
        token = targets["count"]
    targets["stem"] = np.maximum(T.unit_stems(corpus["stems"][index], corpus["atom_levels"][index], level),
                                 STEM_FLOOR)
    targets["count_class"] = targets["count"] - 1
    targets["token"] = token
    targets["auto"] = float(token == 0)
    targets["mix"] = corpus["mix"][index].astype(np.float32)
    targets["frames"] = int(corpus["frames"][index])
    return targets


def collate(items):
    batch = {}
    for key in items[0]:
        value = np.stack([np.asarray(item[key]) for item in items])
        if value.dtype == np.float64:
            value = value.astype(np.float32)
        batch[key] = value.astype(np.float16) if key == "roll" else value  # halves host->GPU traffic
    return batch


class Dataset:
    """torch-style dataset; the level/token draw is seeded by (seed, epoch, index)."""

    def __init__(self, corpus, indices, seed, auto_probability, epoch=0):
        self.corpus, self.indices, self.seed = corpus, indices, seed
        self.auto_probability, self.epoch = auto_probability, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        rng = np.random.default_rng([self.seed, self.epoch, index])
        return example(self.corpus, index, rng, self.auto_probability)


# -------------------------------------------------------------------------- model


def build_model(config):
    import torch
    from torch import nn

    d = config["width"]

    class MixDecomposer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.Linear(FEATURES, d), nn.GELU(), nn.Linear(d, d))
            self.local = nn.Conv1d(d, d, 5, padding=2)
            self.position = nn.Parameter(torch.randn(FRAMES + 1, d) * 0.02)
            self.count_token = nn.Embedding(SLOTS + 1, d)
            layer = nn.TransformerEncoderLayer(d, config["heads"], 4 * d, config["dropout"],
                                               batch_first=True, norm_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, config["layers"], enable_nested_tensor=False)
            self.queries = nn.Parameter(torch.randn(SLOTS, d) * 0.02)
            dec = nn.TransformerDecoderLayer(d, config["heads"], 4 * d, config["dropout"],
                                             batch_first=True, norm_first=True, activation="gelu")
            self.decoder = nn.TransformerDecoder(dec, config["decoder_layers"])
            self.slot_heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, w))
                                             for k, w in HEADS.items()})
            self.frame_proj = nn.Linear(d, d)
            self.slot_proj = nn.Linear(d, d)
            self.roll_head = nn.Linear(d, T.PITCHES * ROLL)
            self.stem_frame = nn.Linear(d, d)
            self.stem_slot = nn.Linear(d, d)
            self.stem_head = nn.Linear(d, mixscene.STEM_BANDS)
            self.count_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, SLOTS))
            self.bus_head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                          nn.Linear(d, len(T.BUS_FX) + T.BUS_WIDTH))

        def forward(self, mix, frames, token):
            batch = mix.shape[0]
            valid = torch.arange(FRAMES, device=mix.device)[None] < frames[:, None]
            x = self.input(mix)
            x = x + self.local(x.transpose(1, 2)).transpose(1, 2)
            x = torch.cat([self.count_token(token)[:, None], x], 1) + self.position[None]
            padding = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=mix.device), ~valid], 1)
            h = self.encoder(x, src_key_padding_mask=padding)
            global_, frames_h = h[:, 0], h[:, 1:]
            q = self.decoder(self.queries[None].expand(batch, -1, -1), h, memory_key_padding_mask=padding)
            out = {k: head(q) for k, head in self.slot_heads.items()}
            out["u1"] = out["u1"].view(batch, SLOTS, N_FAMILIES, T.U_DIMS)
            out["u2"] = out["u2"].view(batch, SLOTS, N_FAMILIES, T.U_DIMS)
            for key in ("exists", "echo_present", "gain"):
                out[key] = out[key].squeeze(-1)
            # Per (slot, frame) product of projections, so the roll/stem heads stay small.
            f = self.frame_proj(frames_h)  # B,T,d
            s = self.slot_proj(q)  # B,S,d
            roll = self.roll_head(torch.relu(f[:, None] * s[:, :, None]))  # B,S,T,P*R
            out["roll"] = roll.view(batch, SLOTS, FRAMES, T.PITCHES, ROLL)
            pooled = torch.nn.functional.adaptive_avg_pool1d(frames_h.transpose(1, 2), STEM_FRAMES).transpose(1, 2)
            stem = self.stem_head(torch.relu(self.stem_frame(pooled)[:, None] * self.stem_slot(q)[:, :, None]))
            out["stem"] = stem  # B,S,STEM_FRAMES,BANDS (log10 power, floor STEM_FLOOR)
            out["count"] = self.count_head(global_)
            bus = self.bus_head(global_ + frames_h.masked_fill(~valid[..., None], 0).sum(1)
                                / frames.clamp(min=1)[:, None])
            out["bus_present"], out["bus_values"] = bus[:, : len(T.BUS_FX)], bus[:, len(T.BUS_FX):]
            return out

    return MixDecomposer()


# ------------------------------------------------------------------------- losses


def pick_family(u, family):
    """u [B,S,F,D] gathered at family [B,S] -> [B,S,D]."""
    index = family.long().clamp(0, N_FAMILIES - 1)[..., None, None].expand(-1, -1, 1, u.shape[-1])
    return u.gather(2, index).squeeze(2)


def match(out, batch):
    """Hungarian assignment per example: permutation that reorders predictions onto targets."""
    import torch
    from scipy.optimize import linear_sum_assignment

    with torch.no_grad():
        logp = torch.log_softmax(out["family1"].float(), -1)  # B,S,F
        family = batch["family1"].long()  # B,S
        cost_family = -logp.gather(2, family[:, None, :].expand(-1, SLOTS, -1))  # B,pred,target
        stem = out["stem"].float()
        cost_stem = (stem[:, :, None] - batch["stem"][:, None]).abs().mean((-1, -2))  # B,pred,target
        onset_p = torch.sigmoid(out["roll"][..., 0].float()).flatten(2)  # B,S,T*P
        onset_t = batch["roll"][..., 0].float().flatten(2)
        overlap = torch.einsum("bin,bjn->bij", onset_p, onset_t)
        cost_notes = 1 - 2 * overlap / (onset_p.sum(-1)[:, :, None] + onset_t.sum(-1)[:, None] + 1e-6)
        cost = (cost_family + cost_stem + 2 * cost_notes).cpu().numpy()
    counts = batch["count"].cpu().numpy()
    permutation = np.zeros((len(counts), SLOTS), np.int64)
    for b, k in enumerate(counts):
        rows, cols = linear_sum_assignment(cost[b][:, :k])
        order = list(rows[np.argsort(cols)])  # prediction slot for target 0..k-1
        order += [i for i in range(SLOTS) if i not in order]
        permutation[b] = order
    return torch.from_numpy(permutation).to(out["exists"].device)


def reorder(out, permutation):
    import torch

    result = {}
    for key, value in out.items():
        if value.dim() >= 2 and value.shape[1] == SLOTS and key not in ("count", "bus_present", "bus_values"):
            index = permutation.view(permutation.shape + (1,) * (value.dim() - 2)).expand(-1, -1, *value.shape[2:])
            result[key] = torch.gather(value, 1, index)
        else:
            result[key] = value
    return result


def compute_losses(out, batch, weights):
    import torch
    import torch.nn.functional as F

    out = reorder(out, match(out, batch))
    exists = batch["exists"]  # B,S float
    n = exists.sum().clamp(min=1)

    def slot_mean(value):  # value B,S
        return (value * exists).sum() / n

    losses = {}
    losses["exists"] = F.binary_cross_entropy_with_logits(out["exists"].float(), exists)
    ce = lambda key, target: F.cross_entropy(out[key].float().flatten(0, 1), target.long().flatten(),
                                             reduction="none").view(target.shape)
    losses["family1"] = slot_mean(ce("family1", batch["family1"]))
    u1 = pick_family(out["u1"].float(), batch["family1"])
    m1 = batch["u1_mask"].float()
    losses["u1"] = slot_mean(((torch.sigmoid(u1) - batch["u1"]).abs() * m1).sum(-1) / m1.sum(-1).clamp(min=1))
    losses["family2"] = slot_mean(ce("family2", batch["family2"]))
    layered = (batch["family2"] != T.NO_LAYER).float() * exists
    u2 = pick_family(out["u2"].float(), batch["family2"].clamp(max=N_FAMILIES - 1))
    m2 = batch["u2_mask"].float()
    term = ((torch.sigmoid(u2) - batch["u2"]).abs() * m2).sum(-1) / m2.sum(-1).clamp(min=1)
    losses["u2"] = (term * layered).sum() / layered.sum().clamp(min=1)
    losses["transpose2"] = (ce("transpose2", batch["transpose2"]) * layered).sum() / layered.sum().clamp(min=1)
    present = batch["insert_present"]
    losses["insert_present"] = slot_mean(F.binary_cross_entropy_with_logits(
        out["insert_present"].float(), present, reduction="none").mean(-1))
    value_mask = torch.zeros_like(batch["insert_values"])
    for index, kind in enumerate(T.INSERT_FX):
        start, width = T.INSERT_OFFSET[kind], len(T.INSERT_PARAMS[kind])
        value_mask[..., start:start + width] = present[..., index:index + 1]
    error = (torch.sigmoid(out["insert_values"].float()) - batch["insert_values"]).abs() * value_mask
    losses["insert_values"] = (error.sum(-1) * exists).sum() / (value_mask.sum(-1) * exists).sum().clamp(min=1)
    echo = batch["echo_present"] * exists
    losses["echo_present"] = slot_mean(F.binary_cross_entropy_with_logits(
        out["echo_present"].float(), batch["echo_present"], reduction="none"))
    echo_terms = (ce("echo_taps", batch["echo_taps"]) + ce("echo_transpose", batch["echo_transpose"])
                  + (torch.sigmoid(out["echo_values"].float()) - batch["echo_values"]).abs().sum(-1))
    losses["echo_values"] = (echo_terms * echo).sum() / echo.sum().clamp(min=1)
    losses["gain"] = slot_mean((torch.sigmoid(out["gain"].float()) - batch["gain"]).abs())

    roll, target = out["roll"].float(), batch["roll"].float()
    frame_valid = (torch.arange(FRAMES, device=roll.device)[None] < batch["frames"][:, None]).float()
    mask = exists[:, :, None, None] * frame_valid[:, None, :, None]  # B,S,T,1
    onset_bce = F.binary_cross_entropy_with_logits(roll[..., 0], target[..., 0], reduction="none",
                                                   pos_weight=torch.tensor(weights["onset_pos"], device=roll.device))
    sustain_bce = F.binary_cross_entropy_with_logits(roll[..., 1], target[..., 1], reduction="none")
    cells = (mask.sum() * T.PITCHES).clamp(min=1)
    losses["onset"] = (onset_bce * mask).sum() / cells
    losses["sustain"] = (sustain_bce * mask).sum() / cells
    on = target[..., 0] * mask
    held = target[..., 1] * mask
    losses["velocity"] = ((torch.sigmoid(roll[..., 2]) - target[..., 2]).abs() * held).sum() / held.sum().clamp(min=1)
    timing = (torch.sigmoid(roll[..., 3]) - target[..., 3]).abs() + (torch.sigmoid(roll[..., 4]) - target[..., 4]).abs()
    losses["timing"] = (timing * on).sum() / on.sum().clamp(min=1)

    stem_error = (out["stem"].float().clamp(min=STEM_FLOOR) - batch["stem"]).abs().mean((-1, -2))
    losses["stem"] = slot_mean(stem_error)
    auto = batch["auto"]
    count_ce = F.cross_entropy(out["count"].float(), batch["count_class"].long(), reduction="none")
    losses["count"] = (count_ce * auto).sum() / auto.sum().clamp(min=1)
    bus_present = batch["bus_present"]
    losses["bus_present"] = F.binary_cross_entropy_with_logits(out["bus_present"].float(), bus_present)
    bus_mask = torch.zeros_like(batch["bus_values"])
    for index, kind in enumerate(T.BUS_FX):
        start, width = T.BUS_OFFSET[kind], len(T.BUS_PARAMS[kind])
        bus_mask[:, start:start + width] = bus_present[:, index:index + 1]
    bus_error = (torch.sigmoid(out["bus_values"].float()) - batch["bus_values"]).abs() * bus_mask
    losses["bus_values"] = bus_error.sum() / bus_mask.sum().clamp(min=1)

    total = sum(weights.get(k, 1.0) * v for k, v in losses.items())
    with torch.no_grad():
        family_hit = (out["family1"].argmax(-1) == batch["family1"].long()).float()
        metrics = {"family_acc": float(slot_mean(family_hit)),
                   "count_acc_auto": float(((out["count"].argmax(-1) == batch["count_class"].long()).float()
                                            * auto).sum() / auto.sum().clamp(min=1)),
                   "exists_acc": float(((out["exists"] > 0).float() == exists).float().mean())}
    return total, {k: float(v.detach()) for k, v in losses.items()}, metrics


DEFAULT_WEIGHTS = {"onset_pos": 30.0, "u1": 2.0, "u2": 1.0, "family2": 0.5, "transpose2": 0.3,
                   "sustain": 0.5, "velocity": 0.3, "timing": 0.5, "stem": 0.5, "count": 0.5}


# ----------------------------------------------------------------------- training


def to_device(batch, device):
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in batch.items()}


def model_inputs(batch):
    return batch["mix"], batch["frames"].long(), batch["token"].long()


def loader(dataset, batch_size, shuffle, workers, seed):
    import torch
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                       collate_fn=collate, generator=generator, drop_last=shuffle,
                                       persistent_workers=False, pin_memory=True)


def run_validation(model, data, device, amp_dtype, weights, batch_size, workers, limit):
    import torch
    model.eval()
    sums, count = {}, 0
    with torch.no_grad():
        for i, batch in enumerate(loader(data, batch_size, False, workers, 0)):
            if limit and i >= limit:
                break
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(*model_inputs(batch))
            total, losses, metrics = compute_losses(out, batch, weights)
            for k, v in {"loss": float(total.detach()), **losses, **metrics}.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
    model.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


def train(args):
    import torch

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    began = time.monotonic()
    corpus = load_corpus(args.corpus)
    train_set = Dataset(corpus, split_indices(corpus, "train"), args.seed, args.auto_probability)
    val_set = Dataset(corpus, split_indices(corpus, "validation"), args.seed + 1, args.auto_probability)
    log(f"corpus {args.corpus}: {len(train_set)} train / {len(val_set)} validation scenes", log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" and not args.no_amp else None
    config = {"width": args.width, "heads": args.heads, "layers": args.layers,
              "decoder_layers": args.decoder_layers, "dropout": args.dropout}
    weights = {**DEFAULT_WEIGHTS, **json.loads(args.loss_weights or "{}")}
    model = build_model(config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    log(f"model {parameters / 1e6:.2f} M parameters on {device}", log_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=amp_dtype is not None)
    steps_per_epoch = max(1, len(train_set) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    epoch, step, best, history, stale = 0, 0, math.inf, [], 0
    resume = find_resume(output, args.resume, args.resume_search or [])
    if resume:
        package = torch.load(resume, map_location="cpu", weights_only=False)  # RNG states stay CPU tensors
        if package.get("schema") != CHECKPOINT_SCHEMA or package["config"] != config:
            raise SystemExit(f"{resume} has another schema or model size; start a new run directory")
        model.load_state_dict(package["model"])
        optimizer.load_state_dict(package["optimizer"])
        scaler.load_state_dict(package["scaler"])
        epoch, step, best, history, stale = (package["epoch"], package["step"], package["best"],
                                            package["history"], package.get("stale", 0))
        restore_rng(package["rng"])
        log(f"resumed {resume} at epoch {epoch} step {step}", log_path)

    def checkpoint(name, epoch_value, position):
        package = {"schema": CHECKPOINT_SCHEMA, "config": config, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch_value,
                   "step": step, "position": position, "best": best, "history": history, "stale": stale,
                   "rng": capture_rng(), "weights": weights}
        save_checkpoint(output / name, package)
        if name == "last.pt":
            atomic_write(output / "progress.json", lambda s: s.write(
                json.dumps({"global_step": step, "epoch": epoch_value}).encode()))

    budget = args.time_budget_hours * 3600 if args.time_budget_hours else math.inf
    last_save = time.monotonic()
    stopped = False
    while epoch < args.epochs and not stopped:
        train_set.epoch = epoch
        skip = step - epoch * steps_per_epoch  # batches of this epoch done before an interruption
        sums, seen = {}, 0
        epoch_began = time.monotonic()
        for i, batch in enumerate(loader(train_set, args.batch_size, True, args.workers, args.seed + epoch)):
            if i >= steps_per_epoch:
                break
            if i < skip:
                continue
            batch = to_device(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total_steps, args.lr, args.warmup)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(*model_inputs(batch))
            total, losses, metrics = compute_losses(out, batch, weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            seen += 1
            for k, v in {"loss": float(total.detach()), **losses, **metrics}.items():
                sums[k] = sums.get(k, 0.0) + v
            if step % args.log_every == 0:
                rate = seen / max(time.monotonic() - epoch_began, 1e-6)
                log(f"epoch {epoch} step {step}/{total_steps} loss {sums['loss'] / seen:.4f} "
                    f"family {sums['family_acc'] / seen:.3f} count {sums['count_acc_auto'] / seen:.3f} "
                    f"{rate:.2f} it/s eta {format_seconds((total_steps - step) / max(rate, 1e-6))}", log_path)
            if time.monotonic() - last_save > args.checkpoint_minutes * 60:
                checkpoint("last.pt", epoch, i + 1)
                last_save = time.monotonic()
            if time.monotonic() - began > budget:
                stopped = True
                break
        if stopped:
            checkpoint("last.pt", epoch, None)
            log("time budget reached; last.pt saved", log_path)
            return 2
        validation = run_validation(model, val_set, device, amp_dtype, weights, args.batch_size,
                                    args.workers, args.val_batches)
        record = {"epoch": epoch, "step": step, "train": {k: v / max(seen, 1) for k, v in sums.items()},
                  "validation": validation, "lr": optimizer.param_groups[0]["lr"],
                  "minutes": (time.monotonic() - began) / 60}
        history.append(record)
        with open(output / "metrics.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log(f"epoch {epoch} validation loss {validation['loss']:.4f} family {validation['family_acc']:.3f} "
            f"count(auto) {validation['count_acc_auto']:.3f} onset {validation['onset']:.4f}", log_path)
        epoch += 1
        if validation["loss"] < best:
            best, stale = validation["loss"], 0
            checkpoint("best.pt", epoch, None)
        else:
            stale += 1
        checkpoint("last.pt", epoch, None)
        last_save = time.monotonic()
        if args.patience and stale >= args.patience:
            log(f"early stop: no validation improvement for {stale} epochs", log_path)
            break
    best_package = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    export = {"schema": MODEL_SCHEMA, "config": config, "model": best_package["model"],
              "epoch": best_package["epoch"], "best_validation": best, "weights": weights,
              "mixscene": {"schema": mixscene.SCHEMA, "sample_rate": mixscene.SAMPLE_RATE, "hop": mixscene.HOP,
                           "mid_bands": mixscene.MID_BANDS, "side_bands": mixscene.SIDE_BANDS}}
    save_checkpoint(output / "model.pt", export)
    log(f"exported model.pt (best validation {best:.4f})", log_path)
    return 0


# --------------------------------------------------------------------- inference


def load_model(path, device="cpu"):
    import torch
    package = torch.load(path, map_location=device, weights_only=False)
    if package.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"{path} is not an {MODEL_SCHEMA} model")
    model = build_model(package["config"]).to(device)
    model.load_state_dict(package["model"])
    model.eval()
    return model


def predict(model, mix_features, frames, count=0):
    """Slot predictions (numpy) for one scene; ``count`` 0 = auto, else exactly ``count`` slots."""
    import torch
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.from_numpy(mix_features[None].astype(np.float32)).to(device),
                    torch.tensor([frames], device=device), torch.tensor([count], device=device))
    exists = torch.sigmoid(out["exists"][0]).cpu().numpy()
    k = count if count else int(out["count"][0].argmax()) + 1
    order = list(np.argsort(-exists)[:k])
    units = []
    for s in order:
        f1 = int(out["family1"][0, s].argmax())
        f2 = int(out["family2"][0, s].argmax())
        sig = lambda key: torch.sigmoid(out[key][0, s].float()).cpu().numpy()
        roll = torch.sigmoid(out["roll"][0, s].float()).cpu().numpy()
        roll[frames:] = 0
        units.append({
            "family1": f1, "u1": torch.sigmoid(out["u1"][0, s, f1].float()).cpu().numpy(),
            "family2": f2, "u2": torch.sigmoid(out["u2"][0, s, min(f2, N_FAMILIES - 1)].float()).cpu().numpy(),
            "transpose2": int(out["transpose2"][0, s].argmax()), "insert_present": sig("insert_present"),
            "insert_values": sig("insert_values"), "echo_present": float(sig("echo_present")),
            "echo_taps": int(out["echo_taps"][0, s].argmax()),
            "echo_transpose": int(out["echo_transpose"][0, s].argmax()), "echo_values": sig("echo_values"),
            "gain": float(sig("gain")), "roll": roll, "exists": float(exists[s])})
    bus_present = torch.sigmoid(out["bus_present"][0].float()).cpu().numpy()
    bus_values = torch.sigmoid(out["bus_values"][0].float()).cpu().numpy()
    return units, bus_present, bus_values


def evaluate(args):
    """Resynthesis scores: render the decoded prediction and compare with the mix."""
    from training.amo.corpus import resynth

    model = load_model(args.model)
    corpus = load_corpus(args.corpus)
    indices = split_indices(corpus, args.split)[: args.resynth]
    rows = []
    for index in indices:
        scene = corpus["scenes"][index]
        _, mix, _ = mixscene.render_scene(scene)  # the spec renders deterministically
        target = resynth.log_spectra(mix)
        modes = {"auto": 0, **{f"fixed_{lv}": int(max(scene["ladder"][lv]) + 1) for lv in T.LEVELS}}
        row = {"index": index, "counts": {lv: int(max(scene["ladder"][lv]) + 1) for lv in T.LEVELS},
               "note_echo": any(i["note_echo"] for i in scene["instruments"])}
        for name, count in modes.items():
            units, bus_present, bus_values = predict(model, corpus["mix"][index].astype(np.float32),
                                                     int(corpus["frames"][index]), count)
            decoded = T.decode_scene(units, bus_present, bus_values, scene["seconds"])
            try:
                rendered = mixscene.render_scene(decoded)[1] if decoded["instruments"] else None
            except Exception:  # noqa: BLE001 - an unrenderable prediction scores inf
                rendered = None
            row[name] = {"count": len(units), "distance": resynth.distance(
                None if rendered is None else resynth.log_spectra(rendered), target)}
        rows.append(row)
        if args.listen_dir and len(rows) <= args.listen:
            folder = Path(args.listen_dir) / f"scene-{index:07d}"
            folder.mkdir(parents=True, exist_ok=True)
            mixscene.prepare.write_wav(folder / "input.wav", mix, mixscene.SAMPLE_RATE)
            units, bp, bv = predict(model, corpus["mix"][index].astype(np.float32), int(corpus["frames"][index]), 0)
            decoded = T.decode_scene(units, bp, bv, scene["seconds"])
            try:
                mixscene.prepare.write_wav(folder / "predicted-auto.wav", mixscene.render_scene(decoded)[1],
                                           mixscene.SAMPLE_RATE)
            except Exception:  # noqa: BLE001
                pass
            (folder / "predicted-auto.json").write_text(json.dumps(decoded, indent=1))
    summary = {"split": args.split, "scenes": len(rows)}
    for name in ["auto"] + [f"fixed_{lv}" for lv in T.LEVELS]:
        values = np.asarray([r[name]["distance"] for r in rows])
        finite = values[np.isfinite(values)]
        summary[name] = {"median_distance": float(np.median(finite)) if finite.size else None,
                         "failed": int((~np.isfinite(values)).sum())}
    summary["auto_count_accuracy_L0"] = float(np.mean([r["auto"]["count"] == r["counts"]["L0"] for r in rows]))
    echo_rows = [r for r in rows if r["note_echo"]]
    summary["echo_scenes"] = len(echo_rows)
    summary["echo_auto_count_is_L0"] = (float(np.mean([r["auto"]["count"] == r["counts"]["L0"] for r in echo_rows]))
                                        if echo_rows else None)
    result = {"summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(summary, indent=2))
    return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("corpus")
    t.add_argument("output")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--decoder-layers", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--auto-probability", type=float, default=0.3)
    t.add_argument("--loss-weights", help="JSON overrides of DEFAULT_WEIGHTS")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=20261106)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--val-batches", type=int, default=0, help="0 = whole validation split")
    t.add_argument("--checkpoint-minutes", type=float, default=20.0)
    t.add_argument("--time-budget-hours", type=float)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--resume", default="auto")
    t.add_argument("--resume-search", nargs="*")
    t.add_argument("--no-amp", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("corpus")
    e.add_argument("model")
    e.add_argument("--split", default="test")
    e.add_argument("--resynth", type=int, default=64)
    e.add_argument("--listen", type=int, default=4)
    e.add_argument("--listen-dir")
    e.add_argument("--out", default="mix-evaluation.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    random.seed(0)
    return train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
