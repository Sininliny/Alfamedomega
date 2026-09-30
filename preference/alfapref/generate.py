"""Patch prior and preference-guided generation (design.md s6).

The prior mirrors training/alfamedomega_train.py: knobs uniform in [0, 1], method drawn with
`method_probs("pretrain")` (capacity share ** 0.5, hybrid 5% of the rest), hybrid pairs drawn
from HYBRID_PARTS except additive with physical. If METHOD.md changes the prior, change it here too.
"""
import numpy as np

METHODS = ["subtractive", "wavetable", "additive", "spectral", "fm", "physical", "granular",
           "sampling", "hybrid"]
HYBRID_PARTS = ["subtractive", "wavetable", "additive", "fm", "physical"]
HYBRID_EXCLUDE = {frozenset(("additive", "physical"))}           # both use PARTIALS
N_KNOBS = 32

PREVALENCE = {"sampling": 30, "wavetable": 22, "subtractive": 20, "fm": 8, "hybrid": 7,
              "granular": 4, "physical": 4, "additive": 3, "spectral": 2}
DIFFICULTY = {"sampling": 1.5, "physical": 1.4, "fm": 1.3, "wavetable": 1.2, "spectral": 1.2,
              "subtractive": 1.0, "granular": 0.8, "hybrid": 0.6, "additive": 0.5}
ALPHA, FLOOR = 0.6, 0.04


def expert_shares(p=PREVALENCE, d=DIFFICULTY, alpha=ALPHA, floor=FLOOR):
    floored = set()
    while True:
        s = sum(p[k] ** alpha * d[k] for k in p if k not in floored)
        c = {k: floor if k in floored else (1 - len(floored) * floor) * p[k] ** alpha * d[k] / s for k in p}
        new = floored | {k for k in p if c[k] < floor}
        if new == floored:
            return c
        floored = new


def method_probs():
    sh = expert_shares()
    q = {m: sh[m] ** 0.5 for m in METHODS}
    q["hybrid"] = sum(q.values()) * 0.05
    s = sum(q.values())
    return {m: v / s for m, v in q.items()}


def sample_prior(rng, n, methods=None):
    """n patches from the training prior. `methods` restricts the method set (e.g. to exclude
    sampling when no sample is loaded)."""
    probs = method_probs()
    names = [m for m in METHODS if methods is None or m in methods]
    p = np.array([probs[m] for m in names])
    p = p / p.sum()
    out = []
    for m in rng.choice(names, size=n, p=p):
        pair = None
        while m == "hybrid" and (pair is None or frozenset(pair) in HYBRID_EXCLUDE):
            pair = [str(v) for v in rng.choice(HYBRID_PARTS, size=2, replace=False)]
        out.append({"method": str(m), "pair": pair, "knobs": rng.random(N_KNOBS).tolist()})
    return out


def perturb(patch, rng, sigma=0.08):
    """'Randomize nearby': Gaussian step on the knobs, reflected into [0, 1]."""
    k = np.asarray(patch["knobs"]) + sigma * rng.standard_normal(N_KNOBS)
    k = np.abs(k)
    k = 1 - np.abs(1 - k)
    return {"method": patch["method"], "pair": patch.get("pair"), "knobs": np.clip(k, 0, 1).tolist()}


def gumbel_topk(scores, k, beta, rng):
    """k indices without replacement, with probability proportional to exp(beta * score)."""
    s = beta * np.asarray(scores, dtype=np.float64)
    g = s - np.log(-np.log(rng.random(len(s))))
    return list(np.argsort(-g)[:k])


def propose(score_fn, rng, n=256, k=8, beta=1.0, seeds=(), seed_frac=0.25, explore_frac=0.2,
            cem_iters=0, elite_frac=0.1, methods=None):
    """Preference-guided proposals.

    score_fn(patches) -> scores (it renders previews, embeds and scores; see profile.Profile).
    seeds: liked patches to perturb. Returns a list of (patch, score, origin) of length k, where
    origin is "prior", "nearby" or "explore"; the explore slots are pure prior draws.
    """
    n_seed = int(round(n * seed_frac)) if seeds else 0
    cands = sample_prior(rng, n - n_seed, methods)
    origin = ["prior"] * len(cands)
    for i in range(n_seed):
        cands.append(perturb(seeds[i % len(seeds)], rng))
        origin.append("nearby")
    scores = np.asarray(score_fn(cands), dtype=np.float64)

    for _ in range(cem_iters):                       # cross-entropy refinement per method
        n_el = max(2, int(len(cands) * elite_frac))
        elite = [cands[i] for i in np.argsort(-scores)[:n_el]]
        new = []
        for m in sorted({e["method"] for e in elite}):
            ks = np.array([e["knobs"] for e in elite if e["method"] == m])
            mu, sd = ks.mean(0), ks.std(0) + 0.05
            pairs = [e.get("pair") for e in elite if e["method"] == m]
            share = max(1, int(round(n * len(ks) / n_el)))
            for j in range(share):
                kn = np.clip(mu + sd * rng.standard_normal(N_KNOBS), 0, 1)
                new.append({"method": m, "pair": pairs[j % len(pairs)], "knobs": kn.tolist()})
        new_scores = np.asarray(score_fn(new), dtype=np.float64)
        cands, scores = cands + new, np.concatenate([scores, new_scores])
        origin += ["refined"] * len(new)

    n_explore = max(1, int(round(k * explore_frac))) if explore_frac > 0 else 0
    picks = gumbel_topk(scores, k - n_explore, beta, rng)
    out = [(cands[i], float(scores[i]), origin[i]) for i in picks]
    if n_explore:
        ex = sample_prior(rng, n_explore, methods)
        for p, s in zip(ex, score_fn(ex)):
            out.append((p, float(s), "explore"))
    return out


# "Randomize nearby" modes (design.md s6.5). sigma is the knob step, beta how strongly the
# scorer picks among the neighbours (0 = uniformly), pull the share of neighbours that also move
# part of the way toward the user's liked sounds of the same method.
NEARBY_MODES = {
    "my_taste": {"sigma": 0.08, "beta": 1.0, "pull": 0.25, "scorer": "personal"},
    "model": {"sigma": 0.08, "beta": 0.0, "pull": 0.0, "scorer": None},
    "community": {"sigma": 0.08, "beta": 1.0, "pull": 0.0, "scorer": "community"},
    "wild": {"sigma": 0.25, "beta": 0.15, "pull": 0.0, "scorer": "personal"},
}
DEFAULT_NEARBY = "my_taste"


def toward(patch, target, rng, sigma):
    """A step from `patch` part of the way (up to half) toward `target`, plus a little noise."""
    a, b = np.asarray(patch["knobs"]), np.asarray(target["knobs"])
    k = a + rng.uniform(0.0, 0.5) * (b - a) + 0.5 * sigma * rng.standard_normal(N_KNOBS)
    return {"method": patch["method"], "pair": patch.get("pair"), "knobs": np.clip(k, 0, 1).tolist()}


def nearby(patch, score_fn, rng, k=8, n=64, sigma=0.08, beta=1.0, pull=0.0, liked=()):
    """Neighbours of the current patch, keeping its method. score_fn may be None (no preference).
    Returns [(patch, score, origin)] with origin "nearby" or "toward-liked"."""
    same = [p for p in liked if p["method"] == patch["method"] and p.get("pair") == patch.get("pair")]
    n_pull = int(round(n * pull)) if same else 0
    cands = [perturb(patch, rng, sigma) for _ in range(n - n_pull)]
    origin = ["nearby"] * len(cands)
    for i in range(n_pull):
        cands.append(toward(patch, same[i % len(same)], rng, sigma))
        origin.append("toward-liked")
    scores = np.zeros(len(cands)) if score_fn is None else np.asarray(score_fn(cands), dtype=np.float64)
    picks = gumbel_topk(scores, k, beta, rng)
    return [(cands[i], float(scores[i]), origin[i]) for i in picks]
