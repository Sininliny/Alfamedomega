"""What the user lets the learner learn from (design.md s7.1).

Learning from behaviour is off until the user turns it on, and each kind of data has its own
switch. The switches live in `<home>/preference/plugin.json`, the file the plugin already reads
for "Learn from how I play":

    {"learn": true,
     "categories": {"performance": true, "parameters": true, "suggestions": true, "workflow": true},
     "personalization": 1.0}

`learn` is the master switch; without it nothing from the event log is used. A missing
`categories` object means every category is on once `learn` is on, so the plugin's current
`{"learn": true}` keeps its meaning. `personalization` (0..1) scales how strongly the learned
taste steers generation and "randomize nearby" (0 = the plain model).

The learner applies these switches itself, even if the log holds more, so turning a category off
also stops old data of that kind from counting at the next fit.
"""
import json
import os

CATEGORIES = {
    "performance": "notes you play: pitch range, note lengths, velocity and expression",
    "parameters": "knob and parameter movements after a sound is loaded",
    "suggestions": "which generated or suggested sounds you keep, like or reject",
    "workflow": "how you use sounds: loads, play time, presets saved, sounds kept in a project, "
                "reuse across projects and quick replacements",
}
PERFORMANCE_FIELDS = ("notes", "pitch_mean", "len_mean", "velocity_mean", "expression")


def default():
    return {"learn": False, "categories": {c: False for c in CATEGORIES}, "personalization": 1.0}


def normalise(raw):
    """Fill in defaults. Anything not explicitly true is off."""
    raw = raw if isinstance(raw, dict) else {}
    learn = raw.get("learn") is True
    cats = raw.get("categories")
    if isinstance(cats, dict):
        cats = {c: learn and cats.get(c) is True for c in CATEGORIES}
    else:
        cats = {c: learn for c in CATEGORIES}
    try:
        pers = min(1.0, max(0.0, float(raw.get("personalization", 1.0))))
    except (TypeError, ValueError):
        pers = 1.0
    return {"learn": learn, "categories": cats, "personalization": pers}


def read(path):
    """Consent from a plugin.json file; a missing or unreadable file means learning is off."""
    try:
        with open(path) as f:
            return normalise(json.load(f))
    except (OSError, ValueError):
        return normalise(None)


def find(events_path):
    """The plugin.json next to an event folder (<home>/preference/events -> <home>/preference)."""
    if not events_path:
        return None
    base = events_path if os.path.isdir(events_path) else os.path.dirname(events_path)
    for d in (base, os.path.dirname(os.path.abspath(base))):
        p = os.path.join(d, "plugin.json")
        if os.path.exists(p):
            return p
    return None


def filter_events(events, consent):
    """Drop or strip what the user has not agreed to. Load events always stay, because they mark
    where one sound ends and the next starts, but they carry no evidence by themselves:
    - workflow off: no play time, saves, project keeps, reuse or quick-replacement penalties
    - suggestions off: rate events dropped and sounds loaded from Generate are not labelled
    - parameters off: edit events dropped
    - performance off: note counts, pitch, length, velocity and expression stripped from play events
    """
    c = normalise(consent)
    if not c["learn"]:
        return []
    cat = c["categories"]
    out = []
    for ev in events:
        typ = ev.get("type")
        if typ == "load":
            ev = dict(ev)
            if not cat["workflow"]:
                ev["project"], ev["no_workflow"] = "", True
            if not cat["suggestions"] and ev.get("source") == "generate":
                ev["unlabelled"] = True
        elif typ == "play":
            ev = dict(ev)
            if not cat["workflow"]:
                ev["seconds"] = 0.0
            if not cat["performance"]:
                for k in PERFORMANCE_FIELDS:
                    ev.pop(k, None)
        elif typ in ("save_preset", "project_save") and not cat["workflow"]:
            continue
        elif typ == "rate" and not cat["suggestions"]:
            continue
        elif typ == "edit" and not cat["parameters"]:
            continue
        out.append(ev)
    return out


def describe(consent):
    """Plain-language list of what is and is not being learned, for the UI and the CLI."""
    c = normalise(consent)
    lines = ["Learning from how you play: %s" % ("on" if c["learn"] else "off (nothing is learned from behaviour)")]
    for k, text in CATEGORIES.items():
        lines.append("  [%s] %s: %s" % ("x" if c["categories"][k] else " ", k, text))
    lines.append("Personalization strength: %.0f%%" % (100 * c["personalization"]))
    lines.append("Everything stays on this computer.")
    return lines
