"""Command line: python -m alfapref <command> (run from the preference/ folder).

    init      --profile DIR                    create an empty profile (behaviour learning off)
    scan      --profile DIR FOLDER             add or refresh a library folder
    forget    --profile DIR FOLDER             remove a library folder's entries
    fit       --profile DIR [--events FILE] [--behaviour on|off]
    words     --profile DIR [WORD]             list learned words, or one word's related words
    generate  --profile DIR [--text "warm pad"] [--context bass|mid|high[,sustained]] [-k 8]
    simulate  [--exposures 300]                simulated user, prints learning curve

--renderer MODULE:CALLABLE picks the preview renderer (default: the sketch renderer). A callable
must take (patch, pitch=60, velocity=0.8, seconds=1.0) and return 48 kHz mono audio.
"""
import argparse
import importlib
import json
import sys

from .profile import Profile, context_from_name
from .renderers import SketchRenderer


def _renderer(spec):
    if not spec:
        return SketchRenderer()
    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="alfapref", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "scan", "forget", "fit", "words", "generate", "simulate"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--profile")
    ap.add_argument("--renderer")
    ap.add_argument("--events")
    ap.add_argument("--behaviour", choices=["on", "off"])
    ap.add_argument("--text")
    ap.add_argument("--context", default="")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("--beta", type=float)
    ap.add_argument("--cem", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exposures", type=int, default=300)
    a = ap.parse_intermixed_args(argv)

    if a.command == "simulate":
        from . import simulate
        simulate.run(a.exposures, seed=a.seed, renderer=_renderer(a.renderer) if a.renderer else None)
        return
    if not a.profile:
        ap.error("--profile is required")
    if a.command == "init":
        Profile.create(a.profile)
        print("created", a.profile)
        return
    prof = Profile.load(a.profile)
    if a.command == "scan":
        print(json.dumps(prof.scan(a.arg)))
        prof.save()
    elif a.command == "forget":
        print("removed", prof.forget(a.arg), "entries")
        prof.save()
    elif a.command == "fit":
        if a.behaviour:
            prof.settings["learn_from_behaviour"] = a.behaviour == "on"
        print(json.dumps(prof.fit(_renderer(a.renderer), a.events)))
        prof.save()
    elif a.command == "words":
        if prof.words is None:
            sys.exit("no words yet: scan a library folder and fit first")
        if a.arg:
            if a.arg not in prof.words.words:
                sys.exit("unknown word: %s" % a.arg)
            for w, c in prof.words.related(a.arg):
                print("%-20s %.2f" % (w, c))
        else:
            for w, v in sorted(prof.words.words.items(), key=lambda kv: -kv[1]["n"]):
                print("%-20s %5.1f" % (w, v["n"]))
    elif a.command == "generate":
        if prof.model is None:
            sys.exit("fit the profile first")
        if a.text and prof.words:
            unknown = prof.words.query(a.text)[1]
            if unknown:
                print("unknown words (ignored):", ", ".join(unknown), file=sys.stderr)
        out = prof.generate(_renderer(a.renderer), context_from_name(a.context), a.text, k=a.k, n=a.n,
                            beta=a.beta, seed=a.seed, cem_iters=a.cem)
        for o in out:
            o["patch"]["knobs"] = [round(v, 4) for v in o["patch"]["knobs"]]
            o["score"] = round(o["score"], 3)
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
