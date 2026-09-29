"""Basis, preference scorer and word map (design.md s2, s4, s5). NumPy only."""
import numpy as np

from . import tokens
from .generate import METHODS

N_CTX = 4


def _sig(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def _softplus_neg(u):                        # l(u) = log(1 + exp(-u)), stable
    return np.logaddexp(0.0, -u)


class Basis:
    """Standardise descriptors with the background statistics, then project onto top PCs."""

    def __init__(self, mean, scale, comps):
        self.mean, self.scale, self.comps = (np.asarray(v, dtype=np.float64) for v in (mean, scale, comps))

    @classmethod
    def fit(cls, background, extra=None, d=24):
        B = np.asarray(background, dtype=np.float64)
        mean = B.mean(0)
        scale = B.std(0) + 1e-3
        X = (B - mean) / scale
        if extra is not None and len(extra):
            X = np.vstack([X, (np.asarray(extra, dtype=np.float64) - mean) / scale])
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return cls(mean, scale, Vt[: min(d, Vt.shape[0])])

    @property
    def dim(self):
        return self.comps.shape[0]

    def __call__(self, desc):
        D = np.atleast_2d(np.asarray(desc, dtype=np.float64))
        return ((D - self.mean) / self.scale) @ self.comps.T

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "comps": self.comps.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["scale"], d["comps"])


def method_onehot(method):
    v = np.zeros(len(METHODS))
    if method is not None:
        v[METHODS.index(method)] = 1.0
    return v


def features(z, method, ctx):
    """f = vec([z; onehot(m)] (x) [1; c]). Library files have no patch, so method is None."""
    a = np.concatenate([np.asarray(z, dtype=np.float64).ravel(), method_onehot(method)])
    b = np.concatenate([[1.0], np.asarray(ctx, dtype=np.float64).ravel()])
    return np.outer(a, b).ravel()


class PreferenceModel:
    """s(x, c) = theta . f(x, c), fitted by Newton's method on a weighted logistic loss with a
    Gaussian prior; the Hessian gives a Laplace posterior for Thompson sampling."""

    def __init__(self, d, lam=1.0, ctx_lam_mult=4.0):
        self.d = d
        self.n_a = d + len(METHODS)
        self.F = self.n_a * (1 + N_CTX)
        self.lam = lam
        prior = np.full((self.n_a, 1 + N_CTX), lam * ctx_lam_mult)
        prior[:, 0] = lam
        self.prior = prior.ravel()
        self.theta = np.zeros(self.F)
        self.chol = None                             # Cholesky factor of the posterior precision
        self.n_rows = 0

    def score(self, z, method, ctx, theta=None):
        th = self.theta if theta is None else theta
        return float(features(z, method, ctx) @ th)

    def fit(self, rows, iters=50, tol=1e-8):
        """rows: list of (feature vector, sign y, weight). A pair (i better than j) is the row
        (f_i - f_j, +1, w); background rows are (f_b, -1, w)."""
        if not rows:
            self.theta, self.chol, self.n_rows = np.zeros(self.F), None, 0
            return self
        X = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows], dtype=np.float64)
        w = np.array([r[2] for r in rows], dtype=np.float64)
        Xy = X * y[:, None]
        th = self.theta.copy()

        def loss(t):
            return float((w * _softplus_neg(Xy @ t)).sum() + 0.5 * (self.prior * t * t).sum())

        cur = loss(th)
        for _ in range(iters):
            m = Xy @ th
            g = -(Xy * (w * _sig(-m))[:, None]).sum(0) + self.prior * th
            h = w * _sig(m) * _sig(-m)
            H = (X * h[:, None]).T @ X + np.diag(self.prior)
            step = np.linalg.solve(H, g)
            a = 1.0
            while a > 1e-4:
                cand = th - a * step
                new = loss(cand)
                if new <= cur - 1e-4 * a * float(g @ step):
                    break
                a *= 0.5
            th, done = cand, cur - new < tol
            cur = new
            if done:
                break
        m = Xy @ th
        h = w * _sig(m) * _sig(-m)
        H = (X * h[:, None]).T @ X + np.diag(self.prior)
        self.theta, self.chol, self.n_rows = th, np.linalg.cholesky(H), len(rows)
        return self

    def sample_theta(self, rng):
        """One draw from N(theta, H^-1)."""
        if self.chol is None:
            return self.theta + rng.standard_normal(self.F) / np.sqrt(self.prior)
        eps = rng.standard_normal(self.F)
        return self.theta + np.linalg.solve(self.chol.T, eps)

    def to_dict(self):
        return {"d": self.d, "lam": self.lam, "prior": self.prior.tolist(), "theta": self.theta.tolist(),
                "chol": None if self.chol is None else self.chol.tolist(), "n_rows": self.n_rows}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["d"], d["lam"])
        m.prior, m.theta = np.array(d["prior"]), np.array(d["theta"])
        m.chol = None if d["chol"] is None else np.array(d["chol"])
        m.n_rows = d["n_rows"]
        return m


class WordModel:
    """One Gaussian per word with a shared diagonal covariance and means shrunk toward the
    library mean; LLR_t(z) = d_t . (z - (mu_t + mu_0) / 2)."""

    def __init__(self, mu0=None, var=None, words=None, kappa=3.0):
        self.mu0 = None if mu0 is None else np.asarray(mu0, dtype=np.float64)
        self.var = None if var is None else np.asarray(var, dtype=np.float64)
        self.words = {} if words is None else {k: {"mu": np.asarray(v["mu"]), "n": v["n"]} for k, v in words.items()}
        self.kappa = kappa

    def fit(self, Z, tags):
        """Z: [N, d] library items; tags: list of {word: weight}."""
        Z = np.asarray(Z, dtype=np.float64)
        self.mu0 = Z.mean(0)
        self.var = Z.var(0) + 1e-2
        acc = {}
        for z, tg in zip(Z, tags):
            for word, wt in tg.items():
                s, n = acc.get(word, (np.zeros_like(self.mu0), 0.0))
                acc[word] = (s + wt * z, n + wt)
        self.words = {k: {"mu": (s + self.kappa * self.mu0) / (n + self.kappa), "n": n} for k, (s, n) in acc.items()}
        return self

    def direction(self, word):
        return (self.words[word]["mu"] - self.mu0) / self.var

    def llr(self, z, word):
        mu = self.words[word]["mu"]
        return float(self.direction(word) @ (np.asarray(z).ravel() - 0.5 * (mu + self.mu0)))

    def query(self, text):
        """Parse a query; returns ([(word, sign)] known, [unknown words])."""
        known, unknown = [], []
        for w, s in tokens.parse_query(text):
            (known if w in self.words else unknown).append((w, s) if w in self.words else w)
        return known, unknown

    def query_score(self, z, parsed):
        return sum(s * self.llr(z, w) for w, s in parsed)

    def related(self, word, min_cos=0.8, min_n=2.0):
        d = self.direction(word)
        out = []
        for other, v in self.words.items():
            if other == word or v["n"] < min_n:
                continue
            e = self.direction(other)
            c = float(d @ e / (np.linalg.norm(d) * np.linalg.norm(e) + 1e-12))
            if c >= min_cos:
                out.append((other, c))
        return sorted(out, key=lambda t: -t[1])

    def top_words(self, z, k=3, min_n=2.0):
        """Words that best describe z, for the 'why this sound' readout."""
        s = [(w, self.llr(z, w)) for w, v in self.words.items() if v["n"] >= min_n]
        return sorted(s, key=lambda t: -t[1])[:k]

    def to_dict(self):
        return {"mu0": None if self.mu0 is None else self.mu0.tolist(),
                "var": None if self.var is None else self.var.tolist(), "kappa": self.kappa,
                "words": {k: {"mu": v["mu"].tolist(), "n": v["n"]} for k, v in self.words.items()}}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mu0"], d["var"], d["words"], d["kappa"])
