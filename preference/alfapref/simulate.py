"""Simulated user (design.md s9): a hidden taste drives behaviour; check that the profile learns it.

The user's hidden utility is a random direction in the first principal components of the
descriptor space. Each exposure the user either keeps the sound (plays it, saves the project,
sometimes saves a preset) with probability sigmoid(2u - 1), or skips it after a few seconds.
Half of the exposures come from the browser (prior patches) and, once a model exists, half from
the profile's own proposals.
"""
import numpy as np

from . import consent
from . import generate as gen
from .model import Basis
from .profile import Profile
from .renderers import SketchRenderer

MID = [0.0, 1.0, 0.0, 0.0]


def auc(scores, truth):
    """Probability that a random pair is ordered the same way by scores and by truth."""
    s, t = np.asarray(scores), np.asarray(truth)
    i, j = np.triu_indices(len(s), 1)
    keep = t[i] != t[j]
    agree = np.sign(s[i] - s[j]) == np.sign(t[i] - t[j])
    ties = s[i] == s[j]
    return float(((agree & ~ties) + 0.5 * ties)[keep].mean())


def run(n_exposures=300, eval_every=50, seed=0, renderer=None, n_background=256, n_eval=150,
        n_candidates=128, log=print):
    rng = np.random.default_rng(seed)
    R = renderer or SketchRenderer()
    prof = Profile(None)
    prof.settings.update(n_background=n_background)
    prof.consent = consent.normalise({"learn": True})       # the simulated user opted in
    prof.ensure_background(R, seed=seed)
    b0 = Basis.fit(prof.background, None, 24)
    u = rng.standard_normal(b0.dim)
    u[6:] = 0
    u /= np.linalg.norm(u)
    raw = lambda p: float(u @ b0(prof.describe_patch(p, MID, R))[0])
    eval_set = gen.sample_prior(rng, n_eval)
    ue = np.array([raw(p) for p in eval_set])
    mu, sd = ue.mean(), ue.std() + 1e-9
    util = lambda p: (raw(p) - mu) / sd
    ue = (ue - mu) / sd

    events, t, history, pending = [], 0.0, [], []
    for i in range(n_exposures):
        if pending and i % 2:
            patch, source = pending.pop(), "generate"
        else:
            patch, source = gen.sample_prior(rng, 1)[0], "browser"
        v = util(patch)
        events.append({"t": t, "type": "load", "inst": "i0", "patch": patch, "source": source,
                       "project": "p%d" % (i // 25)})
        if rng.random() < 1 / (1 + np.exp(-(2 * v - 1))):
            events.append({"t": t + 30, "type": "play", "inst": "i0", "seconds": 40 + 40 * rng.random(),
                           "notes": 30, "pitch_mean": 60, "len_mean": 0.5})
            events.append({"t": t + 290, "type": "project_save", "inst": "i0"})
            if v > 1.0 and rng.random() < 0.5:
                events.append({"t": t + 295, "type": "save_preset", "inst": "i0", "patch": patch})
            t += 300
        else:
            events.append({"t": t + 2, "type": "play", "inst": "i0", "seconds": 1.0, "notes": 2,
                           "pitch_mean": 60, "len_mean": 0.5})
            t += 5
        if (i + 1) % eval_every == 0:
            stats = prof.fit(R, events, now=t)
            s = [prof.model.score(prof.z(p, MID, R), p["method"], MID) for p in eval_set]
            props = prof.generate(R, MID, k=8, n=n_candidates, seed=seed + i, thompson=True)
            pending = [p["patch"] for p in props if p["origin"] != "explore"]
            gen_u = float(np.mean([util(p["patch"]) for p in props if p["origin"] != "explore"]))
            row = {"exposures": i + 1, "points": stats["points"], "pairs": stats["pairs"],
                   "auc": auc(s, ue), "proposal_utility": gen_u, "prior_utility": 0.0}
            history.append(row)
            if log:
                log("exposures %4d  labels %4d  pairs %3d  AUC %.3f  proposal utility %+.2f sd (prior 0)"
                    % (row["exposures"], row["points"], row["pairs"], row["auc"], gen_u))
    return history
