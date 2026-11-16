"""Debug probes for the separation network (MIX-DESIGN.md §14): does the loss / matching / metric path work?

1. ideal masks through compute_losses: the SI-SDR metric must come out high and the loss low;
2. free masks (one logit tensor optimised directly, no network) on one batch: the loss must be able to fit;
3. the network on the same fixed batch, with gradient norms per part.

    python -m training.amo.corpus.mixsep_debug --corpus ROOT [ROOT ...]
"""

import argparse
import json

import numpy as np

from training.amo.corpus import mixsep as M


def batch_for(corpus, indices, min_level_db, device):
    import torch
    data = M.Dataset(corpus, indices, 7, remix=0.0, min_level_db=min_level_db)
    items = M.collate([data[i] for i in range(len(indices))])
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in items.items()}


def probe(corpus, indices, min_level_db, kind, device, steps):
    import torch
    torch.manual_seed(0)
    model = M.build_model({"width": 128, "blocks": 6}).to(device)
    batch = batch_for(corpus, indices, min_level_db, device)
    weights = {**M.DEFAULT_WEIGHTS, "roll": 0.0, "presence": 0.0, "kind": kind}
    with torch.no_grad():
        x = model.spectrum(batch["mix"]).transpose(1, 2)
        b, t, f = x.shape
        targets = model.spectrum(batch["stems"].flatten(0, 1)).view(b, M.SLOTS, M.BINS, -1).transpose(2, 3)
    zeros = {"roll": torch.zeros(b, M.SLOTS, t, M.T.PITCHES, M.ROLL, device=device),
             "presence": torch.zeros(b, M.SLOTS, device=device)}
    report = {"kind": kind, "min_level_db": min_level_db, "atoms": batch["present"].sum(1).tolist()}

    # 1. ideal masks (|S_k| / sum |S|, residual 0), slots in atom order
    mags = targets.abs()
    ideal = mags / mags.sum(1, keepdim=True).clamp(min=1e-8)
    masks = torch.cat([torch.zeros(b, 1, t, f, device=device), ideal], 1).transpose(1, 2)  # B,T,S+1,F
    _, losses, metrics = M.compute_losses(model, {"spectrum": x, "masks": masks, **zeros}, batch, weights)
    uniform = torch.full_like(masks, 1.0 / (M.SLOTS + 1))
    _, ulosses, umetrics = M.compute_losses(model, {"spectrum": x, "masks": uniform, **zeros}, batch, weights)
    report["ideal"] = {**losses, **metrics}
    report["uniform"] = {**ulosses, **umetrics}
    print(json.dumps({"probe": "ideal/uniform", **report}), flush=True)

    # 2. free masks
    logits = torch.zeros(b, t, M.SLOTS + 1, f, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([logits], lr=0.05)
    for step in range(steps):
        total, losses, metrics = M.compute_losses(model, {"spectrum": x, "masks": logits.softmax(2), **zeros},
                                                  batch, weights)
        optimiser.zero_grad()
        total.backward()
        optimiser.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(json.dumps({"probe": "free", "kind": kind, "step": step, **losses, **metrics}), flush=True)

    # 3. the network on the same batch
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for step in range(steps):
        out = model(batch["mix"])
        total, losses, metrics = M.compute_losses(model, out, batch, weights)
        optimiser.zero_grad()
        total.backward()
        norms = {name: float(sum(p.grad.norm() ** 2 for p in part.parameters() if p.grad is not None) ** 0.5)
                 for name, part in (("embed", model.embed), ("blocks", model.blocks), ("masks", model.masks))}
        optimiser.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            m = out["masks"].detach()
            print(json.dumps({"probe": "network", "kind": kind, "step": step, **losses, **metrics,
                              "grad": norms, "mask_max_mean": float(m.max(2).values.mean()),
                              "residual_share": float(m[:, :, 0].mean())}), flush=True)


def loop_probe(corpus, windows, amp, batch_size, device, steps, shuffle=True):
    """The training loop's own data path (Dataset, loader, per-epoch atom order) and AMP choice."""
    import torch
    torch.manual_seed(0)
    model = M.build_model({"width": 128, "blocks": 6}).to(device)
    data = M.Dataset(corpus, M.split_indices(corpus, "train")[:windows], 7, remix=0.0)
    weights = {**M.DEFAULT_WEIGHTS, "roll": 0.0, "presence": 0.0, "kind": "complex"}
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler(enabled=amp)
    step = 0
    while step < steps:
        data.epoch = step
        for batch in M.loader(data, batch_size, shuffle, 2, step):
            batch = M.to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = model(batch["mix"])
            total, losses, metrics = M.compute_losses(model, out, batch, weights)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()
            if step % max(1, steps // 10) == 0 or step == steps - 1:
                print(json.dumps({"probe": "loop", "windows": windows, "amp": amp, "batch": batch_size,
                                  "shuffle": shuffle, "step": step, **losses, **metrics}), flush=True)
            step += 1
            if step >= steps:
                break


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--loop", action="store_true", help="training-loop probes (data path, AMP)")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = M.load_corpus(args.corpus)
    indices = M.split_indices(corpus, "train")[:args.batch]
    if args.loop:
        for windows, amp, batch_size in ((4, False, 4), (4, True, 4), (64, False, 8), (64, True, 8)):
            loop_probe(corpus, windows, amp, batch_size, device, args.steps)
        return 0
    for min_level_db in (-12.0, None):
        for kind in ("compressed", "complex"):
            probe(corpus, indices, min_level_db, kind, device, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
