"""Command line: python -m alfapref <command> (run from the preference/ folder).

    init      --profile DIR                    create an empty profile (behaviour learning off)
    scan      --profile DIR FOLDER             add or refresh a library folder
    forget    --profile DIR FOLDER             remove a library folder's entries
    label     --profile DIR FILE WORD...       teach your own words with one audio example
    forget-word --profile DIR WORD             stop a word meaning anything
    fit       --profile DIR [--events DIR|FILE] [--consent plugin.json] [--behaviour on|off]
    explain   --profile DIR                    what is learned, from what, and what is stored
    reset     --profile DIR [--all]            forget the learned model (--all: examples too)
    words     --profile DIR [WORD]             list learned words, or one word's related words
    generate  --profile DIR [--text "warm pad"] [--context bass|mid|high[,sustained]] [-k 8]
    nearby    --profile DIR --patch JSON|FILE [--mode my_taste|model|community|wild]
              [--community DIR] [-k 8]         "randomize nearby" around a patch
    simulate  [--exposures 300]                simulated user, prints learning curve
    batch     --index FILE [--events DIR|FILE] [--consent FILE] [--assimilation DIR]
              [--labels FILE] [--presets FILE] --out DIR
              control-center batch (docs/control-center/preference-batch.md): reads its library
              index, the plugin-assimilation features, hand labels and the user's own presets if
              given, learns from the behaviour log only if the user switched learning on, writes
              a fresh profile to --out and prints "progress N/M" lines

Behaviour learning is off unless consent says otherwise. --consent names the plugin.json with the
switches (alfapref/consent.py); without it the learner looks next to the event folder
(<home>/preference/plugin.json). `fit --behaviour on` turns every category on for that profile.

--renderer MODULE:CALLABLE picks the preview renderer (default: the sketch renderer). A callable
must take (patch, pitch=60, velocity=0.8, seconds=1.0) and return 48 kHz mono audio.
"""
import argparse
import importlib
import json
import os
import sys

from . import consent
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
    ap.add_argument("command", choices=["init", "scan", "forget", "label", "forget-word", "fit", "explain", "reset",
                                        "words", "generate", "nearby", "simulate", "batch"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--profile")
    ap.add_argument("--renderer")
    ap.add_argument("--events")
    ap.add_argument("--index")
    ap.add_argument("--assimilation", help="control center's <home>/assimilation folder")
    ap.add_argument("--background", type=int, default=512, help="prior renders for the background set")
    ap.add_argument("--out")
    ap.add_argument("--behaviour", choices=["on", "off"])
    ap.add_argument("--consent", help="plugin.json holding the learning switches")
    ap.add_argument("--labels", help="JSONL of hand labels: {\"path\": ..., \"labels\": [...]}")
    ap.add_argument("--presets", help="JSONL of the user's presets: {\"name\", \"category\", \"tags\", \"patch\"}")
    ap.add_argument("--patch")
    ap.add_argument("--mode", default="my_taste")
    ap.add_argument("--community", help="a community profile folder, for --mode community")
    ap.add_argument("--all", action="store_true")
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
    if a.command == "batch":
        if not a.index or not a.out:
            ap.error("batch needs --index and --out")
        return batch(a.index, a.events, a.out, _renderer(a.renderer), a.seed, a.background, a.assimilation,
                     a.consent, a.labels, a.presets)
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
    elif a.command == "label":
        if not a.arg or not a.rest:
            ap.error("label needs a file and at least one word")
        prof.label(a.arg, a.rest)
        print("labelled", os.path.basename(a.arg), "as", ", ".join(a.rest), "(fit to use it)")
        prof.save()
    elif a.command == "forget-word":
        prof.forget_word(a.arg)
        print("forgot", a.arg)
        prof.save()
    elif a.command == "fit":
        cons = None
        if a.behaviour:
            cons = {"learn": a.behaviour == "on"}
        elif a.consent or consent.find(a.events):
            cons = consent.read(a.consent or consent.find(a.events))
        print(json.dumps(prof.fit(_renderer(a.renderer), a.events, consent=cons)))
        prof.save()
    elif a.command == "explain":
        print("\n".join(prof.explain()))
    elif a.command == "reset":
        prof.reset(keep_library=not a.all)
        prof.save()
        print("forgot the learned model" + (" and all examples" if a.all else ""))
    elif a.command == "nearby":
        if not a.patch:
            ap.error("nearby needs --patch")
        patch = json.load(open(a.patch)) if os.path.exists(a.patch) else json.loads(a.patch)
        comm = Profile.load(a.community) if a.community else None
        out, used = prof.randomize_nearby(patch, _renderer(a.renderer), a.mode, context_from_name(a.context),
                                          k=a.k, seed=a.seed, community=comm)
        if used != a.mode:
            print("%s is not available here, explored the model space instead" % a.mode, file=sys.stderr)
        for o in out:
            o["patch"]["knobs"] = [round(v, 4) for v in o["patch"]["knobs"]]
            o["score"] = round(o["score"], 3)
        print(json.dumps({"mode": used, "proposals": out}, indent=1))
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


def batch(index_path, events_path, out, renderer, seed=0, n_background=512, assimilation=None,
          consent_path=None, labels_path=None, presets_path=None):
    """Fresh profile from the control center's index, labels and presets, and the behaviour log
    as far as the user's consent allows (off when no consent file is found)."""
    from . import library
    with open(index_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    presets = []
    if presets_path and os.path.exists(presets_path):
        with open(presets_path) as f:
            presets = [json.loads(line) for line in f if line.strip()]
    n_labels = 0
    if labels_path and os.path.exists(labels_path):
        with open(labels_path) as f:
            n_labels = sum(1 for line in f if line.strip())
    prof = Profile.create(out)
    prof.settings["n_background"] = n_background
    n_bg = prof.settings["n_background"]
    n_assim = len(library.assimilation_lines(assimilation))
    total = len(records) + n_assim + n_labels + len(presets) + n_bg + 1
    done = [0]

    def tick(_=None):
        done[0] += 1
        print("progress %d/%d" % (done[0], total), flush=True)

    prof.index, report = library.from_records(records, prof.salt, tick)
    assim, report["assimilation"] = library.from_assimilation(assimilation, prof.salt, tick)
    prof.index += assim
    labelled, report["labels"] = library.labelled_examples(labels_path, prof.salt, tick)
    prof.index += labelled
    own, report["presets"] = library.from_presets(presets, prof.salt, renderer, tick)
    prof.index += own
    prof.ensure_background(renderer, seed, tick)
    has_events = bool(events_path) and os.path.exists(events_path)
    cons = consent.read(consent_path or consent.find(events_path) or "")
    stats = prof.fit(renderer, events_path if has_events else None, consent=cons)
    prof.save()
    words = {w: round(v["n"], 2) for w, v in prof.words.words.items()} if prof.words else {}
    summary = {"descriptor_id": prof.index[0]["descriptor_id"] if prof.index else None,
               "files": report, "fit": stats, "words": words, "learned": prof.learned,
               "explain": prof.explain()}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    tick()
    print(json.dumps({"files": report, "fit": stats}), flush=True)


if __name__ == "__main__":
    main()
