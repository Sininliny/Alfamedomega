"""Learned control ranking: evaluate a context-importance model for one X layer of a sound.

The ranking is written into the sound as `control_layers[layer].ranking` with the model's
SHA-256, so the plugin can order controls by it and say which model produced it. The
features describe the musical context explicitly; nothing here invents them.
"""
import hashlib
import json
import math
from pathlib import Path

from .local_learning import ContextImportanceModel, load_local_model
from .session import atomic_json
from .schema import schema_is


def rank_controls(sound: Path, model: Path, features: Path, output: Path, *, layer: int,
                  available_ram_bytes: int = 1 << 30) -> dict:
    sound, model, features, output = (Path(p) for p in (sound, model, features, output))
    if output.resolve() in {sound.resolve(), model.resolve(), features.resolve()}:
        raise ValueError("Ranked sound must be saved separately from its inputs")
    document = json.loads(sound.read_text(encoding="utf-8"))
    if not schema_is(document.get("schema"), "amo.tensor/1"):
        raise ValueError("An amo.tensor/1 sound is required")
    layers = document.get("control_layers") or []
    if not 0 <= layer < len(layers):
        raise ValueError(f"The sound has {len(layers)} control layers; layer {layer} does not exist")
    controls = len(layers[layer]["values"])
    importance_model = load_local_model(model, available_ram_bytes=available_ram_bytes)
    if not isinstance(importance_model, ContextImportanceModel):
        raise ValueError("A context importance model is required")
    if importance_model.output_width != controls:
        raise ValueError(f"The model ranks {importance_model.output_width} dimensions but layer {layer} "
                         f"has {controls} controls")
    context = json.loads(features.read_text(encoding="utf-8"))
    importance = importance_model.predict_importance(context)
    if any(not math.isfinite(v) for v in importance):
        raise ValueError("The model produced non-finite importance")
    layers[layer]["ranking"] = {"model": hashlib.sha256(model.read_bytes()).hexdigest(),
                                "name": model.stem, "importance": importance}
    atomic_json(output, document)
    order = sorted(range(controls), key=lambda i: -importance[i])
    return {"sound": str(output), "layer": layer, "order": order, "importance": importance}
