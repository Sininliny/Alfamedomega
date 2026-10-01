"""Independent learner CLI: python -m training.aom train/infer."""
import argparse
import json
from pathlib import Path
from .pipeline import train_manifest, infer_wav, import_wav, train_runtime_manifest


def main():
    parser = argparse.ArgumentParser(description="AOM dual-supervised WAV tensor learner")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("manifest", type=Path)
    training.add_argument("model", type=Path)
    runtime = commands.add_parser("train-runtime")
    runtime.add_argument("manifest", type=Path)
    runtime.add_argument("model", type=Path)
    inference = commands.add_parser("infer")
    inference.add_argument("model", type=Path)
    inference.add_argument("wav", type=Path)
    inference.add_argument("output", type=Path)
    inference.add_argument("--mask-threshold", type=float, required=True)
    importing = commands.add_parser("import-wav")
    importing.add_argument("model", type=Path)
    importing.add_argument("wav", type=Path)
    importing.add_argument("template", type=Path)
    importing.add_argument("bounds", type=Path)
    importing.add_argument("output", type=Path)
    importing.add_argument("--mask-threshold", type=float, required=True)
    importing.add_argument("--sample-rate", type=float, required=True)
    importing.add_argument("--head", choices=("primary_canonical", "secondary_rendered_instance"), required=True)
    importing.add_argument("--dll", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        metadata = train_manifest(args.manifest, args.model)
        print(json.dumps({"model": str(args.model), "initial": metadata["losses"][0], "final": metadata["losses"][-1]}))
    elif args.command == "train-runtime":
        print(json.dumps(train_runtime_manifest(args.manifest, args.model)))
    elif args.command == "infer":
        result = infer_wav(args.model, args.wav, mask_threshold=args.mask_threshold)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(str(args.output))
    else:
        result = import_wav(args.model, args.wav, json.loads(args.template.read_text(encoding="utf-8")),
                            json.loads(args.bounds.read_text(encoding="utf-8")), head=args.head,
                            mask_threshold=args.mask_threshold, sample_rate=args.sample_rate, dll_path=args.dll)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(str(args.output))


if __name__ == "__main__":
    main()
