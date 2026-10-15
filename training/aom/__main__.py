"""Independent learner CLI: python -m training.aom train/infer."""
import argparse
import json
import signal
import sys
from pathlib import Path
from .pipeline import train_manifest, infer_wav, import_wav, train_runtime_manifest
from .session import TrainingSession, TrainingCancelled, atomic_json, inference_metadata
from .workflow import train_local, preflight


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="AMO dual-supervised WAV tensor learner")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("manifest", type=Path)
    training.add_argument("model", type=Path)
    runtime = commands.add_parser("train-runtime")
    runtime.add_argument("manifest", type=Path)
    runtime.add_argument("model", type=Path)
    training_commands = [training, runtime]
    for action in ("train-preference", "train-importance"):
        local = commands.add_parser(action)
        local.add_argument("manifest", type=Path)
        local.add_argument("model", type=Path)
        training_commands.append(local)
    for command in training_commands:
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--checkpoint-every", type=int, default=100)
        command.add_argument("--cancel-file", type=Path)
    checking = commands.add_parser("preflight")
    checking.add_argument("manifest", type=Path)
    checking.add_argument("--category", choices=("wav_preset_learning", "runtime", "user_preference", "context_importance"), required=True)
    producing = commands.add_parser("build-dataset")
    producing.add_argument("plan", type=Path)
    producing.add_argument("output", type=Path)
    producing.add_argument("--dll", type=Path)
    evaluation = commands.add_parser("evaluate-wav")
    evaluation.add_argument("model", type=Path)
    evaluation.add_argument("manifest", type=Path)
    evaluation.add_argument("output", type=Path)
    evaluation.add_argument("--dll", type=Path)
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
    ranking = commands.add_parser("rank-controls")
    ranking.add_argument("sound", type=Path)
    ranking.add_argument("model", type=Path)
    ranking.add_argument("features", type=Path)
    ranking.add_argument("output", type=Path)
    ranking.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    if args.command in ("infer", "import-wav", "evaluate-wav"):
        sources = [getattr(args, key) for key in ("model", "wav", "manifest", "template", "bounds") if getattr(args, key, None)]
        if args.output.resolve() in {path.resolve() for path in sources}:
            parser.error("Output must be separate from model and input files")
    if args.command.startswith("train") and (args.model.resolve() == args.manifest.resolve() or
            args.checkpoint and args.checkpoint.resolve() in (args.model.resolve(), args.manifest.resolve())):
        parser.error("Model, manifest and checkpoint must use separate paths")
    if args.command.startswith("train"):
        cancelled = [False]
        def stop(*_):
            cancelled[0] = True
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        session = TrainingSession(args.command, checkpoint=args.checkpoint, resume=args.resume,
            every=args.checkpoint_every,
            cancelled=lambda: cancelled[0] or bool(args.cancel_file and args.cancel_file.exists()),
            progress=lambda step, total, metrics: print(f"progress {step}/{total} " + json.dumps(metrics), file=sys.stderr, flush=True))
        if args.command == "train":
            report = train_manifest(args.manifest, args.model, session=session)
        elif args.command == "train-runtime":
            report = train_runtime_manifest(args.manifest, args.model, session=session)
        else:
            report = train_local(args.manifest, args.model,
                "user_preference" if args.command == "train-preference" else "context_importance", session=session)
        print("saved " + str(args.model.resolve()), file=sys.stderr, flush=True)
        print(json.dumps({"model": str(args.model), "initial": report["losses"][0], "final": report["losses"][-1],
                          "dataset": inference_metadata(report)["dataset"], "quality_gates_passed": report["quality_gates_passed"]}, allow_nan=False))
    elif args.command == "rank-controls":
        from .ranking import rank_controls
        result = rank_controls(args.sound, args.model, args.features, args.output, layer=args.layer)
        print("saved " + str(args.output.resolve()), file=sys.stderr)
        print(json.dumps(result, allow_nan=False))
    elif args.command == "preflight":
        print(json.dumps(preflight(args.manifest, args.category), allow_nan=False))
    elif args.command == "build-dataset":
        from .producer import build_dataset
        result = build_dataset(args.plan, args.output, dll_path=args.dll,
            progress=lambda step, total, _: print(f"progress {step}/{total}", file=sys.stderr, flush=True))
        print("saved " + result["manifest"], file=sys.stderr)
        print(json.dumps(result))
    elif args.command == "evaluate-wav":
        from .producer import evaluate_wav
        result = evaluate_wav(args.model, args.manifest, dll_path=args.dll)
        atomic_json(args.output, result)
        print("saved " + str(args.output.resolve()), file=sys.stderr)
        print(json.dumps(result, allow_nan=False))
    elif args.command == "infer":
        result = infer_wav(args.model, args.wav, mask_threshold=args.mask_threshold)
        atomic_json(args.output, result)
        print("saved " + str(args.output.resolve()), file=sys.stderr)
        print(str(args.output))
    else:
        result = import_wav(args.model, args.wav, json.loads(args.template.read_text(encoding="utf-8")),
                            json.loads(args.bounds.read_text(encoding="utf-8")), head=args.head,
                            mask_threshold=args.mask_threshold, sample_rate=args.sample_rate, dll_path=args.dll)
        atomic_json(args.output, result)
        print("saved " + str(args.output.resolve()), file=sys.stderr)
        print(str(args.output))


if __name__ == "__main__":
    try:
        main()
    except TrainingCancelled as error:
        print(str(error), file=sys.stderr)
        sys.exit(130)
