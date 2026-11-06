#!/usr/bin/env python3
"""WAV to Preset: turn a one-shot sample into a playable Alfamedomega preset.

The corpus encoder guesses which of our instrument structures, with which parameters,
best explains the sample (``resynth_wav``). The guessed construction is monophonic and
reads MIDI elements 0..2 (frequency, velocity, gate), so this module replicates it once per
voice, points each copy at its own voice row of the plugin's ``voice_indexed`` MIDI layout
(10 elements per voice), mixes the copies, adds ``amo.preset/1`` metadata and checks the
result with the native compiler. The ``.amo.json`` it writes appears in the plugin's sound
library (``%APPDATA%/Alfamedomega/TensorPresets`` on Windows).

    python -m training.amo.corpus.wav_to_preset MODEL kick.wav pad.wav --out DIR
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.amo.corpus import prepare, resynth, resynth_wav  # noqa: E402
from training.amo.native import NativeRuntime  # noqa: E402

VOICE_STRIDE = 10  # plugin voice_indexed row: freq, velocity, gate, ..., channel, note
SUM_KEYS = ("active_primitives", "state_elements", "parameter_elements", "routing_elements",
            "modulation_elements", "output_elements", "stored_tensor_elements", "projection_terms",
            "sparse_state_writes")


def default_library():
    """The plugin's PresetStore::defaultDirectory()."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "Alfamedomega" / "TensorPresets"


def output_count(doc):
    with NativeRuntime(doc, 48000.0, prepare.LIB) as runtime:
        return runtime.output_count


def polyphonic(doc, voices, gain):
    """``voices`` independent copies of a monophonic construction, one per MIDI voice row."""
    if doc.get("shared_initial_state"):
        raise ValueError("constructions with shared initial state cannot be replicated per voice")
    sizes = {name: len(doc["tensors"][name]["values"]) for name in doc["tensors"]}
    states = sum(node["state_size"] for node in doc["instances"])
    outputs = output_count(doc)
    nodes = len(doc["instances"])
    result = copy.deepcopy(doc)
    for name, tensor in result["tensors"].items():
        tensor["values"] = list(tensor["values"]) * voices
        tensor["presence"] = list(tensor.get("presence", [True] * sizes[name])) * voices
        tensor["shape"] = [sizes[name] * voices]
    result["parameter_locks"] = list(doc.get("parameter_locks", [])) * voices

    def shifted(link, v):
        link = dict(link)
        link["weight_element"] += v * sizes[link["weights"]]
        source = link["source"]
        if source == "midi":
            link["source_element"] += v * VOICE_STRIDE
        elif source == "state":
            link["source_element"] += v * states
        elif source == "output_history":
            link["source_element"] += v * outputs
        elif source != "audio":
            raise ValueError(f"cannot replicate a link reading {source!r}")
        return link

    instances, projections, mix = [], [], []
    for v in range(voices):
        for node in doc["instances"]:
            node = copy.deepcopy(node)
            node["instance_id"] = f"{node['instance_id']}-v{v}"
            node["activation"] += v * sizes["activation"]
            node["parameters"] = {"offset": node["parameters"]["offset"] + v * sizes["parameters"],
                                  "count": node["parameters"]["count"]}
            node["audio_bindings"] = [shifted(b, v) for b in node["audio_bindings"]]
            instances.append(node)
        for p in doc["projections"]:
            p = shifted(p, v)
            p["parameter_element"] += v * sizes["parameters"]
            projections.append(p)
        mix += [[index + v * outputs, channel, g * gain] for index, channel, g in doc["outputMix"]]
    result["instances"], result["projections"], result["outputMix"] = instances, projections, mix
    budget = dict(doc.get("budget", {}))
    for key in SUM_KEYS:
        if key in budget:
            budget[key] *= voices
    result["budget"] = budget
    assert len(instances) == nodes * voices
    return result


def release_seconds(doc):
    """Voice tail after note-off: the ADSR release (x3 for the exponential tail), else 2.5 s."""
    for node in doc["instances"]:
        if node["primitive"] == "control.adsr":
            release = doc["tensors"]["parameters"]["values"][node["parameters"]["offset"] + 4]
            return float(np.clip(3 * release + 0.2, 0.3, 4.0))
    return 2.5


def make_preset(mono, name, voices=8, gain=0.5, source=None, guess=None):
    doc = polyphonic(mono, voices, gain)
    doc.pop("dataset_structure", None)
    doc["performance"] = {"mode": "instrument", "midi_layout": "voice_indexed", "voice_capacity": voices,
                          "release_seconds": release_seconds(mono), "bend_range_semitones": 2}
    doc["preferred_mode"] = 0
    doc["preset"] = {"schema": "amo.preset/1", "id": str(uuid.uuid4()), "name": name, "category": "WAV import"}
    doc["description"] = (f"Imported from {source} by the AMO structure encoder: nearest {guess} construction"
                          if source else "Imported by the AMO structure encoder")
    doc["authoring"] = {"kind": "wav to preset (corpus encoder guess)", "family": guess, "source": source,
                        "semantic_mappings": False}
    with NativeRuntime(doc, 48000.0, prepare.LIB):
        pass  # the native compiler must accept the program, or nothing is written
    return doc


def playable(preset, rate=48000.0):
    """The plugin runs every voice row, including idle ones (frequency 0): all must render."""
    voices = preset["performance"]["voice_capacity"]
    frames = int(rate * 0.25)
    idle = np.zeros((frames, voices * VOICE_STRIDE), np.float32)
    played = idle.copy()
    for v, hz in enumerate((55.0, 220.0, 880.0)[:voices]):
        played[:, v * VOICE_STRIDE:v * VOICE_STRIDE + 3] = (hz, 0.8, 1.0)
    try:
        with NativeRuntime(preset, rate, prepare.LIB) as runtime:
            runtime.render(midi=idle)
            runtime.render(midi=played)
        return True
    except ValueError:
        return False


def check_voice(mono, poly, sr=24000, seconds=1.0):
    """Voice 0 of the polyphonic program must sound exactly like the monophonic one."""
    frames = int(sr * seconds)
    t = np.arange(frames) / sr
    row = np.zeros((frames, 3), np.float32)
    row[:, 0], row[:, 1], row[:, 2] = 220.0, 0.8, t < 0.5
    voices = poly["performance"]["voice_capacity"]
    wide = np.zeros((frames, voices * VOICE_STRIDE), np.float32)
    wide[:, :3] = row
    with NativeRuntime(mono, sr, prepare.LIB) as runtime:
        a = prepare.audible(mono, runtime.render(midi=row))
    with NativeRuntime(poly, sr, prepare.LIB) as runtime:
        b = prepare.audible(poly, runtime.render(midi=wide))
    gain = poly["outputMix"][0][2] / mono["outputMix"][0][2]
    return float(np.max(np.abs(a * gain - b)))


def unique_path(folder, stem):
    path = folder / f"{stem}.amo.json"
    n = 2
    while path.exists():
        path = folder / f"{stem} {n}.amo.json"
        n += 1
    return path


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="corpus encoder model.pt")
    parser.add_argument("inputs", nargs="+", type=Path, help="WAV files (one-shots work best)")
    parser.add_argument("--out", type=Path, default=None, help="preset folder (default: the plugin's library)")
    parser.add_argument("--note", default="auto", help="MIDI note of the samples, or auto")
    parser.add_argument("--voices", type=int, default=8)
    parser.add_argument("--preview", action="store_true", help="also write <name>.preview.wav next to each preset")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(max(1, args.threads))
    encoder = resynth_wav.Encoder(args.model)
    out = args.out or default_library()
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for path in args.inputs:
        try:
            # The waveguide's fixed ring cannot hold low notes at 48 kHz or idle (0 Hz) voices.
            excluded = [prepare.NAMES.index("impulse_waveguide")]
            while True:  # a family that cannot play idle voices or low notes is skipped for the next best
                guess = resynth_wav.guess_wav(encoder, path, args.note, exclude=excluded)
                target = unique_path(out, path.stem)
                preset = make_preset(guess["construction"], target.name[:-len(".amo.json")], args.voices,
                                     source=path.name,
                                     guess=guess["info"]["guessed_family"])
                if playable(preset):
                    break
                excluded.append(prepare.NAMES.index(guess["info"]["guessed_family"]))
                guess["info"]["warnings"].append(
                    f"{guess['info']['guessed_family']} cannot play every note in the plugin; used the next best family")
                if len(excluded) >= 5:
                    raise ValueError("no playable construction among the best guesses")
            target.write_text(json.dumps(preset, indent=1), encoding="utf-8")
            if args.preview and guess["resynth"] is not None:
                peak = float(np.abs(guess["resynth"]).max())
                if peak > 1e-6:
                    prepare.write_wav(target.with_suffix("").with_suffix(".preview.wav"),
                                      guess["resynth"] * (0.5 / peak), resynth_wav.SAMPLE_RATE)
            entry = {"input": str(path), "preset": str(target), **guess["info"]}
            print(f"OK {path.name} -> {target.name} ({entry['guessed_family']}, distance {entry['spectral_distance']})"
                  + "".join(f"\n  warning: {w}" for w in guess["info"]["warnings"]), flush=True)
        except Exception as error:  # one bad file must not stop the batch
            entry = {"input": str(path), "error": str(error)}
            print(f"FAILED {path.name}: {error}", flush=True)
        report.append(entry)
    print("RESULT " + json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if all("preset" in r for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
