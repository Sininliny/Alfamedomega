"""Stand-in for training/alfamedomega_train.py with the same CLI and outputs, minus PyTorch."""
import argparse
import json
import os
import sys

GRAPHS = ["patch_encoder", "audio_encoder", "instance_step", "voice_core"] + [
    f"expert_{m}" for m in ["subtractive", "wavetable", "additive", "spectral", "fm", "physical",
                            "granular", "sampling", "hybrid"]]

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd")
tr = sub.add_parser("train")
for flag in ("--tier", "--stage", "--out", "--init", "--teacher", "--data", "--device", "--lr"):
    tr.add_argument(flag)
for flag in ("--steps", "--batch", "--accum", "--seed"):
    tr.add_argument(flag, type=int)
tr.add_argument("--clip", type=float)
tr.add_argument("--smoke", action="store_true")
ex = sub.add_parser("export")
ex.add_argument("--ckpt")
ex.add_argument("--tier")
ex.add_argument("--out")
ex.add_argument("--version", default="0.1.0")
ex.add_argument("--int8", action="store_true")
a = ap.parse_args()

if os.environ.get("FAKE_TRAIN_FAIL"):
    print("boom", file=sys.stderr)
    sys.exit(3)
if a.cmd == "train":
    steps = min(a.steps, 6) if a.smoke else a.steps
    print(f"tier {a.tier}, stage {a.stage}, device cpu")
    for s in range(1, steps + 1):
        print(f"step {s}/{steps} loss 1.0000 mrstft 0.9 gnorm 1.00 lr 3.00e-04 0.01s/step [fm]", flush=True)
    path = os.path.join(a.out, f"{a.tier}_{a.stage}.pt")
    open(path, "wb").write(b"ckpt")
    print(f"saved {path}")
else:
    # Like the real export: a Models folder with <tier>-<version>/ and the host's manifest.json.
    import hashlib
    folder = f"{a.tier}-{a.version}"
    out = os.path.join(a.out, folder)
    os.makedirs(out, exist_ok=True)
    for g in GRAPHS:
        open(os.path.join(out, g + ".onnx"), "wb").write(os.urandom(256) + g.encode())
    open(os.path.join(out, "grain_bank.npy"), "wb").write(b"npy")
    open(os.path.join(out, "dsp_golden.npz"), "wb").write(b"golden")
    json.dump({"tier": a.tier, "sr_trained": 48000, "frame_hz": 250}, open(os.path.join(out, "export_info.json"), "w"))
    files = {fn: hashlib.sha256(open(os.path.join(out, fn), "rb").read()).hexdigest() for fn in sorted(os.listdir(out))}
    json.dump({"schema": 1, "models": [{"tier": a.tier, "version": a.version, "folder": folder, "opset": 17,
                                        "frame_rate_hz": 250, "files": files}]},
              open(os.path.join(a.out, "manifest.json"), "w"))
    print(f"wrote {os.path.join(a.out, 'manifest.json')} entry for {folder}")
