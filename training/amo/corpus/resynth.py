"""Resynthesis evaluation: render a predicted construction and compare it with the target.

Family accuracy counts a prediction as wrong whenever the label differs, even when
two families sound the same (``fm_lowpass`` with its cutoff above the content is
``sine_fm``). This module scores what matters for the instrument instead: the
predicted construction is rendered by the native runtime with the example's own
performance plan and compared with the rendered target by a gain-normalised
multi-resolution log-magnitude STFT distance (both channels, so stereo
width counts).

Everything here is numpy plus the native library; ``evaluate.py --resynth``
supplies the decoded predictions from the torch model.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from training.amo.corpus import prepare
from training.amo.native import NativeRuntime

VALUE_TENSORS = ("parameters", "routing", "modulation", "assets")
FFT_SIZES = (2048, 1024, 512, 256)


def split_rows(root, split):
    """index.jsonl rows of one split, keyed by example id."""
    root = Path(root)
    code = {"train": "train", "validation": "validation", "test": "test"}[split]
    rows = {}
    for shard in json.loads((root / "shards.json").read_text()):
        with open(root / shard["shard"] / "index.jsonl", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["split"] == code:
                    rows[row["id"]] = row
    return rows


_TEMPLATES = {}


def template(type_id):
    """Structure and coordinate bounds of a family; both are fixed per family."""
    if type_id not in _TEMPLATES:
        doc, bounds, _, _ = prepare.make_document(prepare.NAMES[type_id], np.full(64, 0.5))
        _TEMPLATES[type_id] = (doc, bounds)
    return _TEMPLATES[type_id]


def assemble(type_id, values):
    """Construction document of family ``type_id`` carrying padded raw ``values``.

    Values are clipped to the family's declared bounds (assets to [-1, 1]); the
    activation tensor, graph and clocks come from the family template.
    """
    base, bounds = template(type_id)
    doc = json.loads(json.dumps(base))
    for name in ("parameters", "routing", "modulation"):
        tensor = doc["tensors"][name]
        count = len(tensor["values"])
        low = np.array([b[0] for b in bounds[name]], np.float64)
        high = np.array([b[1] for b in bounds[name]], np.float64)
        tensor["values"] = np.clip(np.asarray(values[name][:count], np.float64), low, high).tolist()
    flat = np.clip(np.asarray(values["assets"], np.float64), -1.0, 1.0)
    offset = 0
    for node in doc["instances"]:
        for item in node["assets"]:
            count = len(item["values"])
            item["values"] = flat[offset:offset + count].tolist()
            offset += count
    return doc


def render(doc, performance):
    """Stereo render of ``doc`` for one performance plan, or None if the runtime rejects it."""
    audio, midi = prepare.inputs_for(performance)
    try:
        with NativeRuntime(doc, performance["sample_rate"], prepare.LIB) as runtime:
            output = prepare.audible(doc, runtime.render(audio=audio, midi=midi))
    except ValueError:
        return None
    return output if np.isfinite(output).all() else None


def log_spectra(audio):
    """Log-magnitude STFTs of each channel (signal scaled to unit RMS) at every FFT size."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-9:
        return None
    result = []
    for channel in (audio / rms).T:
        channel = np.ascontiguousarray(channel)
        for size in FFT_SIZES:
            hop = size // 4
            padded = np.pad(channel, (size // 2, size // 2))
            count = 1 + (len(padded) - size) // hop
            frames = np.lib.stride_tricks.as_strided(
                padded, (count, size), (padded.strides[0] * hop, padded.strides[0]))
            result.append(np.log(np.abs(np.fft.rfft(frames * np.hanning(size), axis=1)) + 1e-5))
    return result


def distance(predicted, target):
    """Mean L1 between log spectra over all resolutions; inf for a silent or failed render."""
    if predicted is None or target is None:
        return float("inf")
    return float(np.mean([np.mean(np.abs(a - b)) for a, b in zip(predicted, target)]))


def score(job):
    """Render the target and every candidate of one example; returns candidate distances."""
    target_doc, performance, candidates = job
    target = log_spectra(render(target_doc, performance))
    scores = {}
    for name, (type_id, values) in candidates.items():
        audio = render(assemble(type_id, values), performance)
        scores[name] = distance(None if audio is None else log_spectra(audio), target)
    return scores


def score_all(jobs, workers):
    if workers <= 1:
        return [score(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(score, jobs, chunksize=8))


def summarise(values):
    finite = np.asarray([v for v in values if np.isfinite(v)])
    return {"count": len(values), "failed": len(values) - len(finite),
            "median": float(np.median(finite)) if len(finite) else None,
            "mean": float(np.mean(finite)) if len(finite) else None}
