"""Bounded reader separating observed inputs, construction supervision and IDs."""

from .prepare import ShardReader
import numpy as np

HELD_OUT_TOPOLOGIES = {6, 21, 28, 30}


def batches(root, split="train", batch_size=64, protocol="originals", as_torch=False):
    """Never feed structure/type IDs into an unknown-WAV encoder by accident.

    `originals` uses the original-group 75/12.5/12.5 split. `unseen_topology`
    withholds four entire graph families. Choose one before training/evaluation.
    """
    if protocol not in ("originals", "unseen_topology"):
        raise ValueError("Unknown evaluation protocol")
    if split == "topology_test":
        if protocol != "unseen_topology":
            raise ValueError("Topology test requires unseen_topology protocol")
        source_splits = ("train", "validation", "test")
    else:
        source_splits = (split,)
    for source_split in source_splits:
        for raw in ShardReader(root, source_split).batches(batch_size):
            selected = np.ones(len(raw["id"]), np.bool_)
            if protocol == "unseen_topology":
                held = np.isin(raw["type_id"], list(HELD_OUT_TOPOLOGIES))
                selected = held if split == "topology_test" else ~held
            if not selected.any():
                continue
            inputs = {
                k: raw[k][selected] for k in ("X_audio", "conditions", "fx_input_mask")
            }
            targets = {
                k: raw[k][selected]
                for k in raw
                if k.startswith(("canonical_", "instance_")) and k != "canonical_id"
            }
            metadata = {
                k: raw[k][selected] for k in ("id", "canonical_id", "type_id", "split")
            }
            if as_torch:
                import torch

                inputs = {k: torch.from_numpy(v.copy()) for k, v in inputs.items()}
                targets = {k: torch.from_numpy(v.copy()) for k, v in targets.items()}
            yield {"inputs": inputs, "targets": targets, "metadata": metadata}
