#!/usr/bin/env python3
"""Part encoder (MIX-DESIGN.md §13): one separated instrument part + its notes -> its preset.

``build`` renders mixscene scenes and keeps every L2 atom as a sample: the part's audio
(clean, or soft-mask separated from the mix with its true band energies), its notes and its
label (family, u, insert FX). ``train`` fits the encoder (resumable like ``mixtrain``);
``evaluate`` reports family accuracy and the part resynthesis distance on the test split.

    python -m training.amo.corpus.mixparts build CORPUS --first 200000 --scenes 20000
    python -m training.amo.corpus.mixparts train runs/parts --corpus CORPUS_A CORPUS_B --epochs 40
    python -m training.amo.corpus.mixparts evaluate runs/parts/model.pt --corpus CORPUS_A CORPUS_B --parts 512
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

SCHEMA = "amo.mix-parts/1"
MODEL_SCHEMA = "amo.part-encoder/1"
CHECKPOINT_SCHEMA = "amo.part-encoder-checkpoint/1"
HOP = 1024
MID, SIDE = mixscene.MID_BANDS, mixscene.SIDE_BANDS
FEATURES = MID + SIDE
FRAMES = int(np.ceil(mixscene.SECONDS[1] * mixscene.SAMPLE_RATE / HOP)) + 1
ROLL = 3  # onset, sustain, velocity
N_FAMILIES = len(T.FAMILIES)
CLEAN_PROBABILITY = 0.4
STEM_NOISE = 0.3  # log10 noise on the true band energies used for the masks (half the masked samples)
# insert FX index (in INSERT_FX order) of every normalised insert value
VALUE_FX = np.zeros(T.INSERT_WIDTH, np.int64)
for _kind in T.INSERT_FX:
    VALUE_FX[T.INSERT_OFFSET[_kind]:T.INSERT_OFFSET[_kind] + len(T.INSERT_PARAMS[_kind])] = T.INSERT_FX.index(_kind)
DEFAULT_WEIGHTS = {"family": 1.0, "u": 2.0, "fx_present": 0.5, "fx_values": 0.5}


# ------------------------------------------------------------------- features


def part_features(track):
    """[frames, 160] Mid/Side log band power of one part at hop 1024, relative to its own RMS."""
    rms = float(np.sqrt(np.mean(track ** 2)))
    if rms < 1e-9:
        return None
    mid = (track[:, 0] + track[:, 1]) / (2 * rms)
    side = (track[:, 0] - track[:, 1]) / (2 * rms)
    bands = np.concatenate([mixscene.band_power(mid, HOP, MID), mixscene.band_power(side, HOP, SIDE)], 1)
    out = np.full((FRAMES, FEATURES), np.log10(mixscene.LOG_FLOOR), np.float32)
    values = np.log10(bands + mixscene.LOG_FLOOR)[:FRAMES]
    out[:len(values)] = values
    return out


def roll(notes, frames=FRAMES):
    """[frames, PITCHES, 3] onset / sustain / velocity at hop 1024."""
    out = np.zeros((frames, T.PITCHES, ROLL), np.float32)
    rate = mixscene.SAMPLE_RATE / HOP
    for onset, length, pitch, velocity in notes:
        p = int(pitch) - T.PITCH_LOW
        start = int(round(onset * rate))
        if not 0 <= p < T.PITCHES or not 0 <= start < frames:
            continue
        end = max(start + 1, int(round((onset + length) * rate)))
        out[start, p, 0] = 1.0
        out[start:end, p, 1] = 1.0
        out[start:end, p, 2] = velocity
    return out


def jitter_notes(notes, rng):
    """Note-input noise standing in for the mix model's note errors (§13)."""
    out = [[a + rng.normal(0, 0.02), b, p, float(np.clip(v + rng.normal(0, 0.1), 0.05, 1.0))]
           for a, b, p, v in notes]
    if len(out) > 1 and rng.random() < 0.1:
        out.pop(int(rng.integers(len(out))))
    if out and rng.random() < 0.1:
        a, b, p, v = out[int(rng.integers(len(out)))]
        out.append([a + float(rng.uniform(0.1, 0.5)), b, p + int(rng.choice([-12, -5, 0, 7, 12])), v])
    return [[max(0.0, a), b, p, v] for a, b, p, v in sorted(out)]


def label(unit):
    """Family index, u, used-dims mask and insert FX targets of one L2 unit."""
    target = T.unit_target({**unit, "note_echo": None, "gain_db": -6.0}, 1)
    return {"family": int(target["family1"]), "u": target["u1"], "u_mask": target["u1_mask"],
            "fx_present": target["insert_present"], "fx_values": target["insert_values"]}


# ---------------------------------------------------------------------- build


def scene_parts(seed, index, rng):
    """Samples of one scene: feature matrices and rows (one per L2 atom)."""
    from training.amo.corpus import mixstage2

    scene, stems, mix = mixscene.build_scene(seed, index)
    post = scene["render"]["post_bus_scale"]
    tracks = [s * post for s in stems]
    units = T.level_units(scene, "L2")
    assert len(units) == len(tracks)
    rms = float(np.sqrt(np.mean(mix ** 2)))
    logs = np.stack([mixscene.stem_features(t, rms) for t in tracks])
    logs = np.pad(logs, ((0, 0), (0, max(0, mixscene.MAX_STEM_FRAMES - logs.shape[1])), (0, 0)),
                  constant_values=np.log10(mixscene.LOG_FLOOR))[:, :mixscene.MAX_STEM_FRAMES]
    if rng.random() < 0.5:
        logs = logs + rng.normal(0, STEM_NOISE, logs.shape)
    masked = mixstage2.separate(mix, logs) if len(tracks) > 1 else tracks
    features, rows = [], []
    for a, (unit, clean, separated) in enumerate(zip(units, tracks, masked)):
        source = "clean" if len(tracks) == 1 or rng.random() < CLEAN_PROBABILITY else "masked"
        x = part_features(clean if source == "clean" else separated)
        if x is None:
            continue
        features.append(x.astype(np.float16))
        rows.append({"scene": index, "atom": a, "split": mixscene.split_of(index), "source": source,
                     "family": unit["layers"][0]["family"], "u": unit["layers"][0]["u"],
                     "insert_fx": unit["insert_fx"], "notes": unit["notes"], "gain_db": unit["gain_db"],
                     "seconds": scene["seconds"], "frames": int(min(FRAMES, np.ceil(len(clean) / HOP)))})
    return features, rows


def build_shard(task):
    destination, seed, start, count = task
    path = Path(destination) / "shards" / f"{start:07d}"
    if (path / "complete.json").exists():
        return json.loads((path / "complete.json").read_text())
    path.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    features, rows = [], []
    for index in range(start, start + count):
        rng = np.random.default_rng([seed, index, 13])
        try:
            f, r = scene_parts(seed, index, rng)
        except RuntimeError as error:
            print(f"scene {index}: {error}", flush=True)
            continue
        features += f
        rows += r
    np.save(path / "features.npy", np.stack(features))
    with open(path / "parts.jsonl", "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {"start": start, "count": count, "parts": len(rows), "seconds": time.monotonic() - began}
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
                  f"{results[-1]['parts']} parts)", flush=True)
    report = {"schema": SCHEMA, "seed": seed, "first": first, "scenes": scenes,
              "parts": int(sum(r["parts"] for r in results)), "hop": HOP, "frames": FRAMES,
              "wall_seconds": time.monotonic() - began, "source_commit": prepare.source_commit()}
    prepare.dump(destination / "report.json", report)
    return report


class Features:
    """Read-only view over the shards' feature arrays (inputs may sit on a read-only mount)."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.offsets = np.cumsum([0] + [len(a) for a in arrays])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        shard = int(np.searchsorted(self.offsets, index, side="right")) - 1
        return self.arrays[shard][index - self.offsets[shard]]

    @property
    def shape(self):
        return (len(self),) + self.arrays[0].shape[1:]


def load_corpus(roots):
    """All parts under one or more build outputs (e.g. the two Kaggle generation kernels)."""
    roots = [roots] if isinstance(roots, (str, Path)) else roots
    shards = sorted((s for root in roots for s in Path(root).glob("**/shards/*") if (s / "complete.json").exists()),
                    key=lambda s: s.name)
    names = [s.name for s in shards]
    if len(set(names)) != len(names):
        raise SystemExit("the same shard appears in two build outputs")
    if not shards:
        raise SystemExit(f"no complete shards under {roots}")
    arrays, rows = [], []
    for shard in shards:
        arrays.append(np.load(shard / "features.npy", mmap_mode="r"))
        with open(shard / "parts.jsonl", encoding="utf-8") as stream:
            rows += [json.loads(line) for line in stream]
    return {"features": Features(arrays), "rows": rows}


# ------------------------------------------------------------------- dataset


class Dataset:
    def __init__(self, corpus, indices, seed, jitter=True, epoch=0):
        self.corpus, self.indices, self.seed, self.jitter, self.epoch = corpus, indices, seed, jitter, epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        row = self.corpus["rows"][index]
        rng = np.random.default_rng([self.seed, self.epoch, index])
        notes = jitter_notes(row["notes"], rng) if self.jitter else row["notes"]
        return {"x": self.corpus["features"][index].astype(np.float32), "roll": roll(notes).astype(np.float16),
                "frames": row["frames"], **label(row_unit(row))}


def row_unit(row):
    return {"layers": [{"family": row["family"], "u": row["u"], "transpose": 0}], "insert_fx": row["insert_fx"],
            "notes": row["notes"]}


def collate(items):
    batch = {}
    for key in items[0]:
        value = np.stack([np.asarray(item[key]) for item in items])
        batch[key] = value.astype(np.float32) if value.dtype == np.float64 else value
    return batch


# --------------------------------------------------------------------- model


def build_model(config):
    import torch
    from torch import nn

    width, dropout = config["width"], config["dropout"]

    class PartEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.roll = nn.Linear(T.PITCHES * ROLL, 64)
            self.input = nn.Linear(FEATURES + 64, width)
            self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                                                      nn.Conv1d(width, width, 3, stride=2, padding=1), nn.GELU())
                                        for _ in range(config["conv_blocks"])])
            self.norm = nn.LayerNorm(width)
            layer = nn.TransformerEncoderLayer(width, config["heads"], 4 * width, dropout, batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, config["layers"])
            self.pool = nn.Linear(width, 1)
            self.trunk = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))
            self.family = nn.Linear(width, N_FAMILIES)
            self.u = nn.Linear(width, N_FAMILIES * T.U_DIMS)
            self.fx_present = nn.Linear(width, len(T.INSERT_FX))
            self.fx_values = nn.Linear(width, T.INSERT_WIDTH)

        def forward(self, x, notes, frames):
            b, t = x.shape[:2]
            h = self.input(torch.cat([x, self.roll(notes.reshape(b, t, -1).to(x.dtype))], -1))
            h = h.transpose(1, 2)
            for conv in self.convs:
                h = conv(h)
            h = self.norm(h.transpose(1, 2))
            steps = h.shape[1]
            valid = (torch.arange(steps, device=x.device)[None] * (t / steps)) < frames[:, None].to(x.device)
            h = self.encoder(h, src_key_padding_mask=~valid)
            weights = self.pool(h).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)
            z = self.trunk((weights[..., None] * h).sum(1))
            return {"family": self.family(z), "u": self.u(z).view(b, N_FAMILIES, T.U_DIMS),
                    "fx_present": self.fx_present(z), "fx_values": self.fx_values(z)}

    return PartEncoder()


def compute_losses(out, batch, weights):
    import torch
    import torch.nn.functional as F

    family = batch["family"].long()
    losses = {"family": F.cross_entropy(out["family"].float(), family)}
    u = torch.sigmoid(out["u"].float()[torch.arange(len(family), device=family.device), family])
    mask = batch["u_mask"].float()
    losses["u"] = ((u - batch["u"]).abs() * mask).sum() / mask.sum().clamp(min=1)
    present = batch["fx_present"]
    losses["fx_present"] = F.binary_cross_entropy_with_logits(out["fx_present"].float(), present)
    value_mask = present[:, torch.as_tensor(VALUE_FX, device=present.device)]
    values = torch.sigmoid(out["fx_values"].float())
    losses["fx_values"] = ((values - batch["fx_values"]).abs() * value_mask).sum() / value_mask.sum().clamp(min=1)
    total = sum(weights[k] * v for k, v in losses.items())
    metrics = {"family_acc": float((out["family"].argmax(-1) == family).float().mean()),
               "fx_acc": float(((out["fx_present"] > 0).float() == present).float().mean())}
    return total, {k: float(v.detach()) for k, v in losses.items()}, metrics


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
                                       collate_fn=collate, generator=generator, drop_last=shuffle, pin_memory=True)


def model_inputs(batch):
    return batch["x"], batch["roll"], batch["frames"].long()


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
    from training.amo.corpus.train import (capture_rng, find_resume, format_seconds, learning_rate, log,
                                           restore_rng, save_checkpoint)
    from training.amo.session import atomic_write

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    began = time.monotonic()
    corpus = load_corpus(args.corpus)
    limit = args.max_parts or None  # smoke runs
    train_set = Dataset(corpus, split_indices(corpus, "train")[:limit], args.seed)
    val_set = Dataset(corpus, split_indices(corpus, "validation")[:limit], args.seed + 1, jitter=False)
    log(f"parts {args.corpus}: {len(train_set)} train / {len(val_set)} validation", log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" and not args.no_amp else None
    config = {"width": args.width, "heads": args.heads, "layers": args.layers, "conv_blocks": args.conv_blocks,
              "dropout": args.dropout}
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
                                              "epoch": best_package["epoch"], "best_validation": best,
                                              "hop": HOP, "frames": FRAMES, "families": T.FAMILIES})
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
                log(f"epoch {epoch} step {step}/{total_steps} loss {sums['loss'] / seen:.4f} family "
                    f"{sums['family_acc'] / seen:.3f} u {sums['u'] / seen:.4f} {rate:.2f} it/s "
                    f"eta {format_seconds((total_steps - step) / max(rate, 1e-6))}", log_path)
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
        log(f"epoch {epoch} validation loss {validation['loss']:.4f} family {validation['family_acc']:.3f} "
            f"u {validation['u']:.4f} fx {validation['fx_acc']:.3f}", log_path)
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


class PartEncoder:
    """Inference wrapper: part features + notes -> candidate instruments (top families)."""

    def __init__(self, path):
        import torch
        self.torch = torch
        package = torch.load(path, map_location="cpu", weights_only=False)
        if package.get("schema") not in (MODEL_SCHEMA, CHECKPOINT_SCHEMA):
            raise SystemExit(f"{path} is not an {MODEL_SCHEMA} model")
        self.model = build_model(package["config"])
        self.model.load_state_dict(package["model"])
        self.model.eval()

    def predict(self, x, notes, frames, top=2):
        torch = self.torch
        with torch.no_grad():
            out = self.model(torch.from_numpy(np.asarray(x, np.float32)[None]),
                             torch.from_numpy(roll(notes)[None]), torch.tensor([int(frames)]))
        probabilities = torch.softmax(out["family"][0].float(), -1).numpy()
        chain = decode_chain(torch.sigmoid(out["fx_present"][0].float()).numpy(),
                             torch.sigmoid(out["fx_values"][0].float()).numpy())
        return [{"family": T.FAMILIES[int(f)], "u": torch.sigmoid(out["u"][0, int(f)].float()).numpy().tolist(),
                 "insert_fx": chain, "probability": float(probabilities[f])}
                for f in np.argsort(-probabilities)[:top]]

    def guess(self, track, notes, top=2):
        x = part_features(track)
        return [] if x is None else self.predict(x, notes, min(FRAMES, np.ceil(len(track) / HOP)), top)


def decode_chain(present, values):
    """Insert FX chain from presence probabilities and normalised values (mixtargets conventions)."""
    unit = {"family1": 0, "u1": np.zeros(T.U_DIMS), "family2": T.NO_LAYER, "u2": None, "transpose2": 0,
            "insert_present": present, "insert_values": values, "echo_present": 0.0, "echo_taps": 0,
            "echo_transpose": 0, "echo_values": None, "gain": 0.5,
            "roll": np.zeros((1, T.PITCHES, len(T.ROLL_CHANNELS)))}
    return T.decode_unit(unit)["insert_fx"]


def part_document(guess):
    """Construction of a guessed part: family document with its insert chain."""
    doc, _, _, _ = prepare.make_document(guess["family"], np.asarray(guess["u"]))
    return mixscene.insert_chain(doc, guess["insert_fx"])


def part_distance(guess, notes, target, frames):
    from training.amo.corpus import resynth
    try:
        audio = mixscene.render_document(part_document(guess), frames, notes)
        return float(resynth.distance(resynth.log_spectra(audio), target))
    except Exception:  # noqa: BLE001 - an invalid guessed construction scores inf
        return math.inf


def evaluate(args):
    from training.amo.corpus import resynth

    encoder = PartEncoder(args.model)
    corpus = load_corpus(args.corpus)
    rows = []
    for index in split_indices(corpus, "test")[:args.parts]:
        row = corpus["rows"][index]
        frames = int(round(row["seconds"] * mixscene.SAMPLE_RATE))
        # Reference: the clean part re-rendered from its label (the stored input may be the masked one).
        truth = mixscene.render_document(mixscene.layer_document(row_unit(row), 0), frames, row["notes"])
        target = resynth.log_spectra(truth)
        guesses = encoder.predict(corpus["features"][index], row["notes"], row["frames"])
        rows.append({"index": index, "source": row["source"], "family": row["family"],
                     "guess": guesses[0]["family"], "family_ok": guesses[0]["family"] == row["family"],
                     "top2_ok": row["family"] in [g["family"] for g in guesses],
                     "D": min(part_distance(g, row["notes"], target, frames) for g in guesses)})
    summary = {}
    for source in ("clean", "masked", "all"):
        group = [r for r in rows if source == "all" or r["source"] == source]
        if not group:
            continue
        d = np.asarray([r["D"] for r in group])
        summary[source] = {"parts": len(group), "family_acc": float(np.mean([r["family_ok"] for r in group])),
                           "top2_acc": float(np.mean([r["top2_ok"] for r in group])),
                           "median_D": float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else None,
                           "invalid": int((~np.isfinite(d)).sum())}
    for r in rows:
        r["D"] = r["D"] if math.isfinite(r["D"]) else None
    prepare.dump(Path(args.out), {"schema": "amo.part-encoder-evaluation/1", "summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))
    return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("destination")
    b.add_argument("--first", type=int, default=200000)
    b.add_argument("--scenes", type=int, default=20000)
    b.add_argument("--per-shard", type=int, default=200)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--seed", type=int, default=mixscene.SEED)
    t = sub.add_parser("train")
    t.add_argument("output")
    t.add_argument("--corpus", nargs="+", required=True)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--conv-blocks", type=int, default=2)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--loss-weights")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=20261112)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--val-batches", type=int, default=0)
    t.add_argument("--checkpoint-minutes", type=float, default=20.0)
    t.add_argument("--time-budget-hours", type=float)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--resume", default="auto")
    t.add_argument("--resume-search", nargs="*")
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--max-parts", type=int, default=0, help="use only the first N parts per split (smoke test)")
    e = sub.add_parser("evaluate")
    e.add_argument("model")
    e.add_argument("--corpus", nargs="+", required=True)
    e.add_argument("--parts", type=int, default=512)
    e.add_argument("--out", default="parts-evaluation.json")
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
