#!/usr/bin/env python3
"""Separation network (MIX-DESIGN.md §14): mix -> 8 slot masks on every STFT bin + each slot's notes.

``build`` renders mixscene scenes and stores a random 4 s window of the mix and of every L2 atom
passed alone through the scene's bus (mono Mid, int16) with the atom's notes. ``train`` fits a
band-split network with permutation-invariant (Hungarian) training and remix augmentation;
``evaluate`` reports per-track SI-SDR, log-spectral distance, count accuracy and note F1.

    python -m training.amo.corpus.mixsep build CORPUS --first 300000 --scenes 10000
    python -m training.amo.corpus.mixsep train runs/sep --corpus CORPUS_A CORPUS_B --epochs 60
    python -m training.amo.corpus.mixsep evaluate runs/sep/model.pt --corpus CORPUS_A CORPUS_B
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus import mixscene, mixtargets as T, prepare  # noqa: E402

SCHEMA = "amo.mix-sep/1"
MODEL_SCHEMA = "amo.separator/1"
CHECKPOINT_SCHEMA = "amo.separator-checkpoint/1"
SR = mixscene.SAMPLE_RATE
N_FFT, HOP = mixscene.N_FFT, mixscene.HOP  # 2048 / 512, the roll's hop too (mixtargets.piano_roll)
BINS = N_FFT // 2 + 1
WINDOW = 4 * SR
FRAMES = 1 + WINDOW // HOP  # torch.stft(center=True)
SLOTS = mixscene.MAX_INSTRUMENTS
ROLL = len(T.ROLL_CHANNELS)
# 48 sub-bands, narrow at the bottom: 12x4, 12x8, 12x16, 11x48 bins, then the rest (161).
BAND_WIDTHS = [4] * 12 + [8] * 12 + [16] * 12 + [48] * 11
BAND_WIDTHS.append(BINS - sum(BAND_WIDTHS))
DEFAULT_WEIGHTS = {"separation": 1.0, "residual": 0.5, "roll": 0.5, "presence": 0.1, "onset_pos": 30.0}
COMPRESS = 0.3
SILENT = 1e-3  # an atom quieter than this (RMS relative to the window's mix) is not a target


def mid(audio):
    return (audio[:, 0] + audio[:, 1]) * 0.5


def window_notes(notes, start):
    """Notes relative to a window starting at ``start`` seconds (a note begun before it keeps a
    negative onset: its sustain shows, its onset does not)."""
    end = WINDOW / SR
    return [[a - start, b, p, v] for a, b, p, v in notes if a - start < end and a + b > start]


def roll(notes, frames=FRAMES):
    """mixtargets.piano_roll, plus the sustain of notes begun before the window."""
    out = T.piano_roll([n for n in notes if n[0] >= 0], frames)
    rate = SR / HOP
    for onset, length, pitch, velocity in notes:
        p = int(pitch) - T.PITCH_LOW
        if onset >= 0 or not 0 <= p < T.PITCHES:
            continue
        end = min(frames, int(round((onset + length) * rate)))
        out[:max(end, 0), p, 1] = 1.0
        out[:max(end, 0), p, 2] = velocity
    return out


# ---------------------------------------------------------------------- build


def scene_window(seed, index, rng):
    from training.amo.corpus import mixstage2

    scene, stems, mix = mixscene.build_scene(seed, index)
    post = scene["render"]["post_bus_scale"]
    atoms = mixscene.atoms_of(scene)
    bused = [mixstage2.through_bus(s, scene["bus_fx"]) * post for s in stems]
    hops = (len(mix) - WINDOW) // HOP
    start = int(rng.integers(0, hops + 1)) * HOP if hops > 0 else 0
    piece = lambda a: np.pad(mid(a)[start:start + WINDOW], (0, max(0, WINDOW - (len(a) - start))))  # noqa: E731
    mix_w = piece(mix)
    rms = float(np.sqrt(np.mean(mix_w ** 2)))
    if rms < 1e-5:
        return None
    kept, waves = [], []
    for atom, b in zip(atoms, bused):
        w = piece(b)
        if float(np.sqrt(np.mean(w ** 2))) < SILENT * rms:
            continue
        inst = scene["instruments"][atom["instrument"]]
        shift = inst["layers"][atom["layer"]]["transpose"]
        notes = [[a, l, p + shift, v] for a, l, p, v in atom["notes"]]
        kept.append({"instrument": atom["instrument"], "layer": atom["layer"], "tap": atom["tap"],
                     "family": inst["layers"][atom["layer"]]["family"],
                     "notes": window_notes(notes, start / SR)})
        waves.append(w)
    if not kept:
        return None
    peak = max(float(np.abs(mix_w).max()), max(float(np.abs(w).max()) for w in waves))
    scale = 32000.0 / max(peak, 1e-9)
    row = {"scene": index, "split": mixscene.split_of(index), "start": start, "scale": scale, "atoms": kept,
           "L0": len({a["instrument"] for a in kept}), "bus_fx": [f["type"] for f in scene["bus_fx"]]}
    q = lambda w: np.round(w * scale).astype(np.int16)  # noqa: E731
    return q(mix_w), [q(w) for w in waves], row


def build_shard(task):
    destination, seed, start, count = task
    path = Path(destination) / "shards" / f"{start:07d}"
    if (path / "complete.json").exists():
        return json.loads((path / "complete.json").read_text())
    path.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    mixes, stems, rows = [], [], []
    for index in range(start, start + count):
        rng = np.random.default_rng([seed, index, 14])
        try:
            found = scene_window(seed, index, rng)
        except RuntimeError as error:
            print(f"scene {index}: {error}", flush=True)
            continue
        if found is None:
            continue
        m, s, row = found
        row["first_atom"] = len(stems)
        mixes.append(m)
        stems += s
        rows.append(row)
    np.save(path / "mix.npy", np.stack(mixes))
    np.save(path / "stems.npy", np.stack(stems))
    with open(path / "scenes.jsonl", "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {"start": start, "count": count, "scenes": len(rows), "atoms": len(stems),
               "seconds": time.monotonic() - began}
    prepare.dump(path / "complete.json", summary)
    return summary


def build(destination, first, scenes, per_shard, workers, seed):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tasks = [(str(destination), seed, s, min(per_shard, first + scenes - s))
             for s in range(first, first + scenes, per_shard)]
    began = time.monotonic()
    results = []
    with ProcessPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(build_shard, t) for t in tasks]):
            results.append(future.result())
            print(f"shard {results[-1]['start']} done ({len(results)}/{len(tasks)}, "
                  f"{results[-1]['atoms']} atoms)", flush=True)
    report = {"schema": SCHEMA, "seed": seed, "first": first, "scenes": scenes,
              "kept": int(sum(r["scenes"] for r in results)), "atoms": int(sum(r["atoms"] for r in results)),
              "window": WINDOW, "sample_rate": SR, "wall_seconds": time.monotonic() - began,
              "source_commit": prepare.source_commit()}
    prepare.dump(destination / "report.json", report)
    return report


def load_corpus(roots):
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    shards = sorted((s for root in roots for s in Path(root).glob("**/shards/*") if (s / "complete.json").exists()),
                    key=lambda s: s.name)
    if not shards:
        raise SystemExit(f"no complete shards under {roots}")
    if len({s.name for s in shards}) != len(shards):
        raise SystemExit("the same shard appears in two build outputs")
    mixes, stems, rows = [], [], []
    for number, shard in enumerate(shards):
        mixes.append(np.load(shard / "mix.npy", mmap_mode="r"))
        stems.append(np.load(shard / "stems.npy", mmap_mode="r"))
        with open(shard / "scenes.jsonl", encoding="utf-8") as stream:
            for local, line in enumerate(stream):
                row = json.loads(line)
                row["shard"], row["local"] = number, local
                rows.append(row)
    return {"mix": mixes, "stems": stems, "rows": rows}


def scene_audio(corpus, row):
    """(mix [WINDOW], stems [K, WINDOW]) float32 of one stored window."""
    scale = row["scale"]
    mix = corpus["mix"][row["shard"]][row["local"]].astype(np.float32) / scale
    first = row["first_atom"]
    stems = corpus["stems"][row["shard"]][first:first + len(row["atoms"])].astype(np.float32) / scale
    return mix, stems


# ------------------------------------------------------------------- dataset


class Dataset:
    """Stored windows; with probability ``remix`` a new mixture of atoms from two windows of the split."""

    def __init__(self, corpus, indices, seed, remix=0.5, epoch=0):
        self.corpus, self.indices, self.seed, self.remix, self.epoch = corpus, indices, seed, remix, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        rng = np.random.default_rng([self.seed, self.epoch, position])
        row = self.corpus["rows"][self.indices[position]]
        mix, stems = scene_audio(self.corpus, row)
        notes = [a["notes"] for a in row["atoms"]]
        if rng.random() < self.remix:
            other = self.corpus["rows"][self.indices[int(rng.integers(len(self.indices)))]]
            _, stems2 = scene_audio(self.corpus, other)
            pool = [(s, n) for s, n in zip(stems, notes)] + [(s, a["notes"]) for s, a in zip(stems2, other["atoms"])]
            take = rng.permutation(len(pool))[:int(rng.integers(1, min(SLOTS, len(pool)) + 1))]
            gains = 10 ** (rng.uniform(-6, 6, len(take)) / 20)
            stems = np.stack([pool[i][0] * g for i, g in zip(take, gains)])
            notes = [pool[i][1] for i in take]
            mix = stems.sum(0)
            peak = float(np.abs(mix).max())
            if peak > 0.95:
                mix, stems = mix * 0.95 / peak, stems * 0.95 / peak
        k = len(stems)
        order = rng.permutation(k)
        padded = np.zeros((SLOTS, WINDOW), np.float32)
        padded[:k] = stems[order]
        rolls = np.zeros((SLOTS, FRAMES, T.PITCHES, ROLL), np.float16)
        for s, i in enumerate(order):
            rolls[s] = roll(notes[i])
        present = np.zeros(SLOTS, np.float32)
        present[:k] = 1
        return {"mix": mix.astype(np.float32), "stems": padded, "roll": rolls, "present": present}


def collate(items):
    return {key: np.stack([item[key] for item in items]) for key in items[0]}


# --------------------------------------------------------------------- model


def build_model(config):
    import torch
    from torch import nn

    width = config["width"]
    edges = np.cumsum([0] + BAND_WIDTHS)

    class Block(nn.Module):
        """Bidirectional GRU along time (per band), then along bands (per frame)."""

        def __init__(self):
            super().__init__()
            self.time_norm, self.band_norm = nn.LayerNorm(width), nn.LayerNorm(width)
            self.time_rnn = nn.GRU(width, width, batch_first=True, bidirectional=True)
            self.band_rnn = nn.GRU(width, width, batch_first=True, bidirectional=True)
            self.time_out, self.band_out = nn.Linear(2 * width, width), nn.Linear(2 * width, width)

        def forward(self, h):  # B, T, K, W
            b, t, k, w = h.shape
            x = self.time_norm(h).permute(0, 2, 1, 3).reshape(b * k, t, w)
            x = self.time_out(self.time_rnn(x)[0]).view(b, k, t, w).permute(0, 2, 1, 3)
            h = h + x
            x = self.band_norm(h).reshape(b * t, k, w)
            h = h + self.band_out(self.band_rnn(x)[0]).view(b, t, k, w)
            return h

    class Separator(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
            self.embed = nn.ModuleList([nn.Sequential(nn.LayerNorm(2 * n), nn.Linear(2 * n, width))
                                        for n in BAND_WIDTHS])
            self.blocks = nn.ModuleList([Block() for _ in range(config["blocks"])])
            self.masks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.Tanh(),
                                                      nn.Linear(2 * width, (SLOTS + 1) * n)) for n in BAND_WIDTHS])
            self.slot_rnn = nn.GRU(width, width, batch_first=True, bidirectional=True)
            self.roll = nn.Linear(2 * width, T.PITCHES * ROLL)
            self.presence = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

        def spectrum(self, audio):
            return torch.stft(audio.float(), N_FFT, HOP, window=self.window, center=True, return_complex=True)

        def forward(self, mix):
            x = self.spectrum(mix).transpose(1, 2)  # B, T, F complex
            b, t, _ = x.shape
            scale = x.abs().pow(2).mean((1, 2), keepdim=True).sqrt().clamp(min=1e-6)
            xr = torch.view_as_real(x / scale)  # B, T, F, 2
            h = torch.stack([emb(xr[:, :, lo:hi].reshape(b, t, -1))
                             for emb, lo, hi in zip(self.embed, edges[:-1], edges[1:])], 2)  # B, T, K, W
            for block in self.blocks:
                h = block(h)
            logits = torch.cat([head(h[:, :, i]).view(b, t, SLOTS + 1, -1)
                                for i, head in enumerate(self.masks)], -1)  # B, T, S+1, F
            masks = logits.float().softmax(2)
            # slot features: band features pooled with each slot's mean mask in the band
            weights = torch.stack([masks[..., lo:hi].mean(-1) for lo, hi in zip(edges[:-1], edges[1:])], -1)
            weights = weights[:, :, 1:]  # B, T, S, K
            z = torch.einsum("btsk,btkw->bstw", weights.to(h.dtype), h) / weights.sum(-1).clamp(min=1e-4)[..., None]
            z = self.slot_rnn(z.reshape(b * SLOTS, t, -1))[0]  # B*S, T, 2W
            roll = self.roll(z).view(b, SLOTS, t, T.PITCHES, ROLL)
            presence = self.presence(z.mean(1)).view(b, SLOTS)
            return {"spectrum": x, "masks": masks, "roll": roll, "presence": presence}

    return Separator()


def compress(magnitude):
    return magnitude.clamp(min=1e-8).pow(COMPRESS)


def match(cost, present):
    """Hungarian matching per item: slot index of every true atom (cost [B, S, S], rows = slots)."""
    from scipy.optimize import linear_sum_assignment
    slots = []
    for c, k in zip(cost.detach().float().cpu().numpy(), present.sum(1).long().tolist()):
        rows, cols = linear_sum_assignment(c[:, :k])
        assignment = np.zeros(k, np.int64)
        assignment[cols] = rows
        slots.append(assignment)
    return slots


def compute_losses(model, out, batch, weights):
    import torch
    import torch.nn.functional as F

    x, masks = out["spectrum"], out["masks"]  # B,T,F ; B,T,S+1,F
    b = x.shape[0]
    targets = model.spectrum(batch["stems"].flatten(0, 1)).view(b, SLOTS, BINS, -1).transpose(2, 3)  # B,S,T,F
    target_c = compress(targets.abs())
    estimate_c = compress(masks[:, :, 1:].transpose(1, 2) * x.abs()[:, None])  # B,S,T,F
    with torch.no_grad():  # cost[i, slot, atom], one atom at a time to keep memory at B*S*T*F
        cost = torch.stack([(estimate_c - target_c[:, a:a + 1]).abs().mean((-1, -2)) for a in range(SLOTS)], -1)
    present = batch["present"]
    assignment = match(cost, present)
    order = torch.zeros(b, SLOTS, dtype=torch.long, device=x.device)
    matched = torch.zeros(b, SLOTS, device=x.device)
    for i, slots in enumerate(assignment):
        free = [s for s in range(SLOTS) if s not in set(slots.tolist())]
        order[i] = torch.as_tensor(list(slots) + free, device=x.device)  # atom position -> slot
        matched[i, torch.as_tensor(slots, device=x.device)] = 1
    # reorder estimates into atom order; the free slots line up with the zero targets
    estimate = torch.gather(estimate_c, 1, order[:, :, None, None].expand_as(estimate_c))
    losses = {"separation": (estimate - target_c).abs().mean()}
    residual = (x - targets.sum(1)).abs()
    losses["residual"] = (compress(masks[:, :, 0] * x.abs()) - compress(residual)).abs().mean()
    roll = torch.gather(out["roll"].float(), 1, order[:, :, None, None, None].expand_as(out["roll"]))
    target = batch["roll"].float()
    exists = present[:, :, None, None]
    onset = F.binary_cross_entropy_with_logits(roll[..., 0], target[..., 0], reduction="none",
                                               pos_weight=torch.tensor(weights["onset_pos"], device=x.device))
    sustain = F.binary_cross_entropy_with_logits(roll[..., 1], target[..., 1], reduction="none")
    cells = (exists.sum() * roll.shape[2] * T.PITCHES).clamp(min=1)
    on, held = target[..., 0] * exists, target[..., 1] * exists
    velocity = ((torch.sigmoid(roll[..., 2]) - target[..., 2]).abs() * held).sum() / held.sum().clamp(min=1)
    timing = (((torch.sigmoid(roll[..., 3]) - target[..., 3]).abs() + (torch.sigmoid(roll[..., 4]) - target[..., 4])
               .abs()) * on).sum() / on.sum().clamp(min=1)
    losses["roll"] = ((onset + sustain) * exists).sum() / cells + velocity + timing
    presence_logit = torch.gather(out["presence"].float(), 1, order)
    losses["presence"] = F.binary_cross_entropy_with_logits(presence_logit, present)
    total = sum(weights[k] * v for k, v in losses.items())
    with torch.no_grad():
        count_ok = ((out["presence"] > 0).sum(1) == present.sum(1)).float().mean()
    return total, {k: float(v.detach()) for k, v in losses.items()}, {"count_acc": float(count_ok)}


# --------------------------------------------------------------------- train


def split_indices(corpus, split):
    return [i for i, r in enumerate(corpus["rows"]) if r["split"] == split]


def to_device(batch, device):
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in batch.items()}


def loader(dataset, batch_size, shuffle, workers, seed):
    import torch
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                       collate_fn=collate, generator=generator, drop_last=shuffle, pin_memory=True,
                                       persistent_workers=False)


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
                out = model(batch["mix"])
            total, losses, metrics = compute_losses(model, out, batch, weights)
            for k, v in {"loss": float(total.detach()), **losses, **metrics}.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
    model.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


def train(args):
    import torch
    from training.amo.corpus.train import (capture_rng, find_resume, format_seconds, learning_rate, log,
                                           restore_rng, save_checkpoint)
    from training.amo.session import atomic_write

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    began = time.monotonic()
    corpus = load_corpus(args.corpus)
    limit = args.max_scenes or None
    train_set = Dataset(corpus, split_indices(corpus, "train")[:limit], args.seed, remix=args.remix)
    val_set = Dataset(corpus, split_indices(corpus, "validation")[:limit], args.seed + 1, remix=0.0)
    log(f"windows {args.corpus}: {len(train_set)} train / {len(val_set)} validation", log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" and not args.no_amp else None
    config = {"width": args.width, "blocks": args.blocks}
    weights = {**DEFAULT_WEIGHTS, **json.loads(args.loss_weights or "{}")}
    model = build_model(config).to(device)
    log(f"model {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters on {device}", log_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=amp_dtype is not None)
    steps_per_epoch = max(1, len(train_set) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    epoch, step, best, history, stale = 0, 0, math.inf, [], 0
    resume = find_resume(output, args.resume, args.resume_search or [])
    if resume:
        package = torch.load(resume, map_location="cpu", weights_only=False)
        if package.get("schema") != CHECKPOINT_SCHEMA or package["config"] != config:
            raise SystemExit(f"{resume} has another schema or model size; start a new run directory")
        model.load_state_dict(package["model"])
        optimizer.load_state_dict(package["optimizer"])
        scaler.load_state_dict(package["scaler"])
        epoch, step, best, history, stale = (package["epoch"], package["step"], package["best"],
                                            package["history"], package.get("stale", 0))
        restore_rng(package["rng"])
        log(f"resumed {resume} at epoch {epoch} step {step}", log_path)

    def checkpoint(name, epoch_value):
        save_checkpoint(output / name, {"schema": CHECKPOINT_SCHEMA, "config": config, "model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                        "epoch": epoch_value, "step": step, "best": best, "history": history,
                                        "stale": stale, "rng": capture_rng(), "weights": weights})
        if name == "last.pt":
            atomic_write(output / "progress.json", lambda s: s.write(
                json.dumps({"global_step": step, "epoch": epoch_value}).encode()))

    def export():
        best_package = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        save_checkpoint(output / "model.pt", {"schema": MODEL_SCHEMA, "config": config, "model": best_package["model"],
                                              "epoch": best_package["epoch"], "best_validation": best})
        log(f"exported model.pt (best validation {best:.4f})", log_path)

    budget = args.time_budget_hours * 3600 if args.time_budget_hours else math.inf
    last_save = time.monotonic()
    while epoch < args.epochs:
        train_set.epoch = epoch
        skip = step - epoch * steps_per_epoch
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
                out = model(batch["mix"])
            total, losses, metrics = compute_losses(model, out, batch, weights)
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
                log(f"epoch {epoch} step {step}/{total_steps} loss {sums['loss'] / seen:.4f} sep "
                    f"{sums['separation'] / seen:.4f} roll {sums['roll'] / seen:.4f} count "
                    f"{sums['count_acc'] / seen:.3f} {rate:.2f} it/s eta "
                    f"{format_seconds((total_steps - step) / max(rate, 1e-6))}", log_path)
            if time.monotonic() - last_save > args.checkpoint_minutes * 60:
                checkpoint("last.pt", epoch)
                last_save = time.monotonic()
            if time.monotonic() - began > budget:
                checkpoint("last.pt", epoch)
                log("time budget reached; last.pt saved", log_path)
                if (output / "best.pt").exists():
                    export()
                return 2
        validation = run_validation(model, val_set, device, amp_dtype, weights, args.batch_size, args.workers,
                                    args.val_batches)
        record = {"epoch": epoch, "step": step, "train": {k: v / max(seen, 1) for k, v in sums.items()},
                  "validation": validation, "lr": optimizer.param_groups[0]["lr"],
                  "minutes": (time.monotonic() - began) / 60}
        history.append(record)
        with open(output / "metrics.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log(f"epoch {epoch} validation loss {validation['loss']:.4f} sep {validation['separation']:.4f} "
            f"roll {validation['roll']:.4f} count {validation['count_acc']:.3f}", log_path)
        epoch += 1
        if validation["loss"] < best:
            best, stale = validation["loss"], 0
            checkpoint("best.pt", epoch)
        else:
            stale += 1
        checkpoint("last.pt", epoch)
        last_save = time.monotonic()
        if args.patience and stale >= args.patience:
            log(f"early stop: no validation improvement for {stale} epochs", log_path)
            break
    export()
    return 0


# ------------------------------------------------------------------ inference


class Separator:
    """mix (stereo, any length) -> slot tracks (stereo, masks from the Mid applied to L and R) + notes."""

    def __init__(self, path, device="cpu"):
        import torch
        self.torch = torch
        package = torch.load(path, map_location="cpu", weights_only=False)
        if package.get("schema") not in (MODEL_SCHEMA, CHECKPOINT_SCHEMA):
            raise SystemExit(f"{path} is not an {MODEL_SCHEMA} model")
        self.model = build_model(package["config"]).to(device)
        self.model.load_state_dict(package["model"])
        self.model.eval()
        self.device = device

    def separate(self, mix, threshold=0.5):
        """Whole clip at once (the network is recurrent, so any length works). Returns slots
        [{track (stereo), notes, presence}] for slots whose presence passes ``threshold``."""
        torch = self.torch
        with torch.no_grad():
            audio = torch.from_numpy(np.asarray(mix, np.float32)).to(self.device)
            out = self.model(audio.mean(1)[None])
            masks = out["masks"][0, :, 1:].transpose(0, 1)  # S, T, F
            stereo = torch.stft(audio.T, N_FFT, HOP, window=self.model.window, center=True,
                                return_complex=True).transpose(1, 2)  # 2, T, F
            presence = torch.sigmoid(out["presence"][0]).cpu().numpy()
            rolls = torch.sigmoid(out["roll"][0].float()).cpu().numpy()
            slots = []
            for s in range(SLOTS):
                if presence[s] < threshold:
                    continue
                spec = (stereo * masks[s][None]).transpose(1, 2)
                track = torch.istft(spec, N_FFT, HOP, window=self.model.window, center=True, length=len(mix))
                slots.append({"slot": s, "track": track.T.cpu().numpy().astype(np.float64),
                              "notes": T.roll_notes(rolls[s]), "presence": float(presence[s])})
        return slots


def si_sdr(estimate, reference):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    energy = float(np.dot(reference, reference))
    if energy < 1e-12:
        return None
    projection = np.dot(estimate, reference) / energy * reference
    noise = estimate - projection
    return float(10 * np.log10(max(np.dot(projection, projection), 1e-12) / max(np.dot(noise, noise), 1e-12)))


def note_f1(predicted, truth, tolerance=0.05):
    """Onset F1 (same pitch, onset within ``tolerance`` s), counting only notes starting in the window."""
    truth = [n for n in truth if n[0] >= 0]
    if not truth and not predicted:
        return 1.0
    used, hits = set(), 0
    for a, _, p, _ in predicted:
        for j, (b, _, q, _) in enumerate(truth):
            if j not in used and q == p and abs(a - b) <= tolerance:
                used.add(j)
                hits += 1
                break
    precision = hits / max(len(predicted), 1)
    recall = hits / max(len(truth), 1)
    return 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)


def evaluate(args):
    from scipy.optimize import linear_sum_assignment
    from training.amo.corpus import resynth

    separator = Separator(args.model)
    corpus = load_corpus(args.corpus)
    rows = []
    for index in split_indices(corpus, "test")[:args.scenes]:
        row = corpus["rows"][index]
        mix, stems = scene_audio(corpus, row)
        slots = separator.separate(np.stack([mix, mix], 1))
        k = len(stems)
        result = {"index": index, "atoms": k, "L0": row["L0"], "found": len(slots), "count_ok": len(slots) == k}
        if slots:
            tracks = [s["track"][:, 0] for s in slots]
            cost = np.array([[-(si_sdr(t, s) or -60.0) for s in stems] for t in tracks])
            r, c = linear_sum_assignment(cost)
            result["si_sdr"] = [float(-cost[i, j]) for i, j in zip(r, c)]
            result["log_D"] = [float(resynth.distance(resynth.log_spectra(np.stack([tracks[i]] * 2, 1)),
                                                      resynth.log_spectra(np.stack([stems[j]] * 2, 1))))
                               for i, j in zip(r, c)]
            result["note_f1"] = [note_f1(slots[i]["notes"], row["atoms"][j]["notes"]) for i, j in zip(r, c)]
        rows.append(result)
    flat = lambda key: [x for r in rows for x in r.get(key, [])]  # noqa: E731
    summary = {"scenes": len(rows), "count_acc": float(np.mean([r["count_ok"] for r in rows])),
               "mean_found_minus_atoms": float(np.mean([r["found"] - r["atoms"] for r in rows])),
               "median_si_sdr": float(np.median(flat("si_sdr"))) if flat("si_sdr") else None,
               "median_log_D": float(np.median(flat("log_D"))) if flat("log_D") else None,
               "mean_note_f1": float(np.mean(flat("note_f1"))) if flat("note_f1") else None}
    prepare.dump(Path(args.out), {"schema": "amo.separator-evaluation/1", "summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))
    return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("destination")
    b.add_argument("--first", type=int, default=300000)
    b.add_argument("--scenes", type=int, default=10000)
    b.add_argument("--per-shard", type=int, default=100)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--seed", type=int, default=mixscene.SEED)
    t = sub.add_parser("train")
    t.add_argument("output")
    t.add_argument("--corpus", nargs="+", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--clip", type=float, default=5.0)
    t.add_argument("--width", type=int, default=128)
    t.add_argument("--blocks", type=int, default=6)
    t.add_argument("--remix", type=float, default=0.5)
    t.add_argument("--loss-weights")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=20261113)
    t.add_argument("--log-every", type=int, default=200)
    t.add_argument("--val-batches", type=int, default=0)
    t.add_argument("--checkpoint-minutes", type=float, default=20.0)
    t.add_argument("--time-budget-hours", type=float)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--resume", default="auto")
    t.add_argument("--resume-search", nargs="*")
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--max-scenes", type=int, default=0, help="use only the first N windows per split (smoke test)")
    e = sub.add_parser("evaluate")
    e.add_argument("model")
    e.add_argument("--corpus", nargs="+", required=True)
    e.add_argument("--scenes", type=int, default=300)
    e.add_argument("--out", default="separator-evaluation.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    if args.command == "build":
        print(json.dumps(build(args.destination, args.first, args.scenes, args.per_shard, args.workers, args.seed),
                         indent=2))
        return 0
    return train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
