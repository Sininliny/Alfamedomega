"""A user's local preference profile: ties the evidence, basis, scorer, word map and sampler
together (design.md s4 to s7). A profile is one folder:

    profile.json    settings, salt, background descriptors, fitted basis, scorer and word map
    library.jsonl   one entry per library file: salted key, size, mtime, descriptor, words
    events.jsonl    behaviour log written by the plugin (read only here)
"""
import json
import os
import secrets

import numpy as np

from . import descriptor as desc_mod
from . import events as ev_mod
from . import generate as gen
from . import library
from .model import Basis, PreferenceModel, WordModel, features

VERSION = 1
PITCH_FOR_CTX = {0: 36, 1: 60, 2: 84}
NO_CTX = [0.0, 0.0, 0.0, 0.0]


def preview_pitch(ctx):
    for i in (0, 2, 1):
        if ctx[i] > 0.5:
            return PITCH_FOR_CTX[i]
    return 60


def context_from_name(name):
    names = ["bass", "mid", "high", "sustained"]
    c = [0.0] * 4
    for part in (name or "").split(","):
        if part.strip():
            c[names.index(part.strip())] = 1.0
    return c


class Profile:
    def __init__(self, path, data=None):
        self.path = path
        d = data or {}
        self.salt = d.get("salt") or secrets.token_hex(16)
        self.settings = d.get("settings", {"n_background": 512, "d": 24, "lam": 1.0, "beta": 1.0,
                                           "gamma": 1.0, "learn_from_behaviour": False})
        self.background = [np.asarray(b) for b in d.get("background", [])]
        self.background_methods = d.get("background_methods", [])
        self.basis = Basis.from_dict(d["basis"]) if d.get("basis") else None
        self.model = PreferenceModel.from_dict(d["model"]) if d.get("model") else None
        self.words = WordModel.from_dict(d["words"]) if d.get("words") else None
        self.liked = d.get("liked", [])
        self.index = []
        self._desc_cache = {}

    # -- storage
    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "profile.json")) as f:
            p = cls(path, json.load(f))
        lib = os.path.join(path, "library.jsonl")
        if os.path.exists(lib):
            with open(lib) as f:
                p.index = [json.loads(line) for line in f if line.strip()]
        return p

    @classmethod
    def create(cls, path):
        os.makedirs(path, exist_ok=True)
        p = cls(path)
        p.save()
        return p

    def save(self):
        d = {"version": VERSION, "descriptor_id": desc_mod.DESCRIPTOR_ID, "salt": self.salt,
             "settings": self.settings,
             "background": [[round(float(v), 5) for v in b] for b in self.background],
             "background_methods": self.background_methods,
             "basis": self.basis.to_dict() if self.basis else None,
             "model": self.model.to_dict() if self.model else None,
             "words": self.words.to_dict() if self.words else None,
             "liked": self.liked}
        tmp = os.path.join(self.path, "profile.json.tmp")
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, os.path.join(self.path, "profile.json"))
        tmp = os.path.join(self.path, "library.jsonl.tmp")
        with open(tmp, "w") as f:
            for e in self.index:
                f.write(json.dumps(e) + "\n")
        os.replace(tmp, os.path.join(self.path, "library.jsonl"))

    # -- evidence
    def scan(self, folder, max_files=None):
        self.index, report = library.scan(folder, self.salt, self.index, max_files)
        return report

    def forget(self, folder):
        before = len(self.index)
        self.index = library.forget(self.index, self.salt, folder)
        return before - len(self.index)

    def describe_patch(self, patch, ctx, renderer):
        pitch = preview_pitch(ctx)
        key = (ev_mod.fingerprint(patch), pitch)
        if key not in self._desc_cache:
            self._desc_cache[key] = desc_mod.descriptor(renderer(patch, pitch=pitch), desc_mod.SR)
        return self._desc_cache[key]

    def ensure_background(self, renderer, seed=0, progress=None):
        n = self.settings["n_background"]
        if len(self.background) == n:
            return
        rng = np.random.default_rng(seed)
        pats = gen.sample_prior(rng, n)
        self.background = []
        for i, p in enumerate(pats):
            if progress:
                progress(i)
            self.background.append(self.describe_patch(p, NO_CTX, renderer))
        self.background_methods = [p["method"] for p in pats]

    # -- fitting
    def fit(self, renderer, event_log=None, now=None):
        """Refit basis, scorer and word map from the library index and (if enabled) the log."""
        self.ensure_background(renderer)
        points, pairs = [], []
        if event_log is not None and self.settings.get("learn_from_behaviour"):
            evs = ev_mod.read_log(event_log) if isinstance(event_log, str) else event_log
            points, pairs = ev_mod.labels(ev_mod.exposures(evs), now)
        lib_desc = [np.asarray(e["desc"]) for e in self.index]
        pt_desc = [self.describe_patch(p, c, renderer) for p, c, _, _ in points]
        pr_desc = [(self.describe_patch(a, c, renderer), self.describe_patch(b, c, renderer)) for a, b, c, _ in pairs]
        extra = lib_desc + pt_desc
        self.basis = Basis.fit(self.background, extra if extra else None, self.settings["d"])
        B = self.basis
        rows = []
        for dsc, e in zip(lib_desc, self.index):
            rows.append((features(B(dsc)[0], None, NO_CTX), 1.0, float(e.get("weight", 1.0))))
        for dsc, (p, c, y, w) in zip(pt_desc, points):
            rows.append((features(B(dsc)[0], p["method"], c), float(y), w))
        for (da, db), (a, b, c, w) in zip(pr_desc, pairs):
            rows.append((features(B(da)[0], a["method"], c) - features(B(db)[0], b["method"], c), 1.0, w))
        w_pos = sum(r[2] for r in rows if r[1] > 0)
        if w_pos > 0:
            wb = w_pos / len(self.background)
            for dsc in self.background:
                rows.append((features(B(dsc)[0], None, NO_CTX), -1.0, wb))
        self.model = PreferenceModel(B.dim, self.settings["lam"]).fit(rows)
        if self.index:
            self.words = WordModel().fit(B(np.array(lib_desc)), [e["words"] for e in self.index])
        else:
            self.words = None
        liked = sorted([(w, p) for p, _, y, w in points if y > 0], key=lambda t: -t[0])
        self.liked = [p for _, p in liked[:16]]
        return {"library": len(lib_desc), "points": len(points), "pairs": len(pairs),
                "background": len(self.background)}

    # -- scoring and generation
    def z(self, patch, ctx, renderer):
        return self.basis(self.describe_patch(patch, ctx, renderer))[0]

    def score_fn(self, renderer, ctx=NO_CTX, text=None, theta=None, gamma=None):
        gamma = self.settings["gamma"] if gamma is None else gamma
        parsed = self.words.query(text)[0] if (text and self.words) else []

        def fn(patches):
            out = []
            for p in patches:
                z = self.z(p, ctx, renderer)
                s = self.model.score(z, p["method"], ctx, theta)
                if parsed:
                    s += gamma * self.words.query_score(z, parsed)
                out.append(s)
            return out
        return fn

    def generate(self, renderer, ctx=NO_CTX, text=None, k=8, n=256, beta=None, seed=0, cem_iters=0,
                 thompson=True, methods=None):
        rng = np.random.default_rng(seed)
        beta = self.settings["beta"] if beta is None else beta
        theta = self.model.sample_theta(rng) if thompson else None
        fn = self.score_fn(renderer, ctx, text, theta)
        props = gen.propose(fn, rng, n=n, k=k, beta=beta, seeds=self.liked, cem_iters=cem_iters,
                            methods=methods)
        out = []
        for patch, s, origin in props:
            why = self.words.top_words(self.z(patch, ctx, renderer)) if self.words else []
            out.append({"patch": patch, "score": s, "origin": origin, "words": [w for w, _ in why]})
        return out
