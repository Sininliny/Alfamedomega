"""Behaviour events -> exposures -> weak labels and pairs (design.md s3.1).

Event log: one JSON object per line, in time order. Every event has "t" (unix seconds), "type"
and "inst" (salted hash of the plugin instance). Types and extra fields:

    load          patch, source ("browser" | "project" | "generate" | "init" | "user"), project (hash)
    play          seconds (held-note seconds since the last play event), notes, pitch_mean, len_mean
    edit          patch (the current patch after a debounced edit)
    save_preset   patch
    project_save
    rate          value (+1 or -1)
    close

A patch is {"method": str, "pair": [str, str] | null, "knobs": [32 floats in 0..1]}.

consent.filter_events() runs first and may add two flags to a load: "no_workflow" (no quick-replace
penalty) and "unlabelled" (the exposure gives no labels). A quick replacement of the "init"
default patch is not a judgement of it, so it carries no penalty either.
"""
import hashlib
import json
import math
import os
from dataclasses import dataclass, field

HALF_LIFE_DAYS = 180.0
QUICK_DWELL_S = 20.0
QUICK_PLAY_S = 5.0
SMALL_EDIT = 0.05
MIN_ABS_REWARD = 0.15
BROWSE_SKIP_WEIGHT = 0.3
PAIR_MIN_POS = 0.5


def fingerprint(patch):
    key = [patch["method"], patch.get("pair"), [round(float(v), 3) for v in patch["knobs"]]]
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:16]


def knob_distance(a, b):
    ka, kb = a["knobs"], b["knobs"]
    if a["method"] != b["method"] or a.get("pair") != b.get("pair"):
        return 1.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(ka, kb)) / len(ka))


@dataclass
class Exposure:
    inst: str
    start: float
    patch: dict
    source: str = "user"
    project: str = ""
    final: dict = None
    end: float = None
    play_s: float = 0.0
    notes: int = 0
    pitch_sum: float = 0.0
    len_sum: float = 0.0
    saved: bool = False
    kept: bool = False
    rating: int = 0
    replaced: bool = False
    reuses: int = 0
    no_workflow: bool = False
    unlabelled: bool = False
    next: "Exposure" = field(default=None, repr=False)

    @property
    def dwell(self):
        return (self.end if self.end is not None else self.start) - self.start

    def context(self):
        """[bass, mid, high, sustained] from what was played (design.md s2)."""
        if self.notes == 0:
            return [0.0, 0.0, 0.0, 0.0]
        pm, lm = self.pitch_sum / self.notes, self.len_sum / self.notes
        return [float(pm < 48), float(48 <= pm <= 72), float(pm > 72), float(lm > 1.0)]

    def reward(self):
        """(pos, neg) evidence for this exposure."""
        play = min(1.0, math.log1p(self.play_s) / math.log1p(120.0))
        pos = 0.5 * play + 1.0 * self.saved + 0.6 * self.kept + 0.5 * min(self.reuses, 2) \
            + 1.5 * max(self.rating, 0)
        quick = self.replaced and self.dwell < QUICK_DWELL_S and self.play_s < QUICK_PLAY_S \
            and not self.saved and not self.kept
        skip_weight = 0.0 if (self.no_workflow or self.source == "init") else \
            BROWSE_SKIP_WEIGHT if self.source == "browser" else 1.0
        neg = quick * skip_weight \
            + 1.5 * max(-self.rating, 0)
        return pos, neg


def read_log(path):
    """One JSONL file, or a folder of them (e.g. <home>/preference/events/<date>.jsonl).
    Lines that do not parse are skipped, since the plugin may be mid-write. Sorted by time."""
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl")) \
        if os.path.isdir(path) else [path]
    out = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if isinstance(ev, dict) and "type" in ev and "t" in ev:
                    out.append(ev)
    return sorted(out, key=lambda e: float(e["t"]))


def exposures(events):
    """Group events into exposures, in load order."""
    current, out = {}, []
    projects_by_fp = {}
    for ev in events:
        inst, typ, t = ev.get("inst", ""), ev["type"], float(ev["t"])
        cur = current.get(inst)
        if typ == "load":
            nxt = Exposure(inst, t, ev["patch"], ev.get("source", "user"), ev.get("project", ""),
                           no_workflow=bool(ev.get("no_workflow")), unlabelled=bool(ev.get("unlabelled")))
            if cur is not None:
                cur.end, cur.replaced, cur.next = t, True, nxt
            current[inst] = nxt
            out.append(nxt)
            continue
        if cur is None:
            continue
        cur.end = t
        if typ == "play":
            n = int(ev.get("notes", 0))
            cur.play_s += float(ev.get("seconds", 0.0))
            cur.notes += n
            cur.pitch_sum += float(ev.get("pitch_mean", 60.0)) * n
            cur.len_sum += float(ev.get("len_mean", 0.5)) * n
        elif typ == "edit":
            cur.final = ev["patch"]
        elif typ == "save_preset":
            cur.saved, cur.final = True, ev.get("patch", cur.final)
        elif typ == "project_save":
            cur.kept = True
        elif typ == "rate":
            cur.rating = 1 if ev.get("value", 0) > 0 else -1
        elif typ == "close":
            del current[inst]
    for e in out:                         # a patch loaded again under another project is reused
        fp = fingerprint(e.final or e.patch)
        seen = projects_by_fp.setdefault(fp, [])
        if e.project and e.project not in seen:
            seen.append(e.project)
    for e in out:
        fp = fingerprint(e.final or e.patch)
        e.reuses = max(0, len(projects_by_fp.get(fp, [])) - 1)
    return out


def labels(exps, now=None):
    """Weak labels. Returns (points, pairs):
    points: list of (patch, context, y in {-1, +1}, weight)
    pairs:  list of (better patch, worse patch, context, weight)"""
    if now is None:
        now = max((e.end or e.start) for e in exps) if exps else 0.0
    points, pairs = [], []
    score = {}
    for e in exps:
        pos, neg = e.reward()
        if e.unlabelled:
            pos = neg = 0.0
        r = pos - neg
        decay = 0.5 ** (max(0.0, now - e.start) / 86400.0 / HALF_LIFE_DAYS)
        score[id(e)] = (pos, neg, r, decay)
        ctx = e.context()
        edited = e.final is not None and knob_distance(e.patch, e.final) > SMALL_EDIT
        if e.unlabelled:
            continue
        if abs(r) >= MIN_ABS_REWARD:
            y = 1 if r > 0 else -1
            # after a large edit the praise belongs to the edited patch, not the loaded one
            points.append((e.final if (edited and y > 0) else e.patch, ctx, y, min(1.0, abs(r)) * decay))
        if edited and pos > PAIR_MIN_POS:
            pairs.append((e.final, e.patch, ctx, min(1.0, pos) * decay))
    for e in exps:
        n = e.next
        if n is None:
            continue
        pa, na, ra, da = score[id(e)]
        pb, nb, rb, db = score[id(n)]
        if ra <= -MIN_ABS_REWARD and rb >= MIN_ABS_REWARD:
            pairs.append((n.final or n.patch, e.patch, n.context(), min(abs(ra), abs(rb), 1.0) * min(da, db)))
    return points, pairs
