"""Run from preference/: python -m unittest discover tests   (ALFA_SLOW=1 adds the simulation)."""
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alfapref import consent as C                         # noqa: E402
from alfapref import descriptor as D                      # noqa: E402
from alfapref import events as E                          # noqa: E402
from alfapref import generate as G                        # noqa: E402
from alfapref import library as L                         # noqa: E402
from alfapref import tokens as T                          # noqa: E402
from alfapref.model import PreferenceModel, WordModel, features   # noqa: E402
from alfapref.profile import Profile                      # noqa: E402
from alfapref.renderers import SketchRenderer             # noqa: E402
from alfapref.simulate import auc                         # noqa: E402

SR = 48000


def tone(f0=220.0, bright=1.0, sr=SR, secs=0.5):
    t = np.arange(int(sr * secs)) / sr
    x = sum(np.sin(2 * np.pi * f0 * h * t) * h ** -(2.5 - 2 * bright) for h in range(1, 30) if f0 * h < sr / 2)
    x = x * np.exp(-t * 3)
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


def write_wav(path, x, sr=SR, bits=16, float_=False):
    x = np.asarray(x, dtype=np.float64)
    if float_:
        data, tag = x.astype("<f4").tobytes(), 3
    elif bits == 16:
        data, tag = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes(), 1
    else:
        v = (np.clip(x, -1, 1) * (2 ** 23 - 1)).astype(np.int32)
        data, tag = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], 1).astype(np.uint8).tobytes(), 1
    fmt = struct.pack("<HHIIHH", tag, 1, sr, sr * bits // 8, bits // 8, bits)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data)


class Tokens(unittest.TestCase):
    def test_path_words(self):
        w = T.path_words("Pads/Warm Stuff/warm_pad_C#m_120bpm_v2_FINAL.wav")
        self.assertEqual(w["warm"], 1.0)
        self.assertEqual(w["pad"], 1.0)
        self.assertEqual(w["stuff"], 0.5)
        for gone in ("c#m", "120", "bpm", "final", "v2", "wav"):
            self.assertNotIn(gone, w)

    def test_camel_and_fm(self):
        self.assertEqual(T.words("FMBell"), ["fm", "bell"])
        self.assertEqual(T.words("glassBells02"), ["glass", "bell"])
        self.assertEqual(T.words("Fm"), [])

    def test_notes_with_octave_and_plural_rule(self):
        self.assertEqual(T.words("Pad_F#1_Chorus"), ["pad", "chorus"])
        self.assertEqual(T.words("C-1 Eb3 bass"), ["bass"])
        self.assertEqual(T.words("Pads Leads Keys"), ["pad", "lead", "key"])
        self.assertNotIn("key", T.words("Pluck_Key_Am"))
        self.assertEqual(T.words("Pads Bells Glass Analysis"), ["pad", "bell", "glass", "analysis"])

    def test_query(self):
        self.assertEqual(T.parse_query("warm airy not dark -harsh"),
                         [("warm", 1.0), ("airy", 1.0), ("dark", -1.0), ("harsh", -1.0)])


class Descriptor(unittest.TestCase):
    def test_shape_and_loudness_invariance(self):
        x = tone()
        a, b = D.descriptor(x), D.descriptor(0.1 * x)
        self.assertEqual(a.shape, (D.DIM,))
        self.assertLess(np.abs(a[:128] - b[:128]).mean(), 0.05)

    def test_brightness_raises_centroid(self):
        dark, bright = D.descriptor(tone(bright=0.1)), D.descriptor(tone(bright=1.0))
        self.assertGreater(bright[128], dark[128] + 0.05)

    def test_resample(self):
        a = D.descriptor(tone(sr=48000), 48000)
        b = D.descriptor(tone(sr=44100), 44100)
        self.assertLess(np.abs(a[:64] - b[:64]).mean(), 0.1)

    def test_silence_and_short(self):
        for x in (np.zeros(100), np.zeros(48000), tone(secs=0.01)):
            self.assertTrue(np.isfinite(D.descriptor(x)).all())


class Library(unittest.TestCase):
    def test_wav_formats(self):
        x = tone()
        with tempfile.TemporaryDirectory() as d:
            for bits, fl in ((16, False), (24, False), (32, True)):
                p = os.path.join(d, "t.wav")
                write_wav(p, x, bits=bits, float_=fl)
                y, sr = L.read_wav(p)
                self.assertEqual(sr, SR)
                np.testing.assert_allclose(y, x, atol=2e-4)

    def test_scan_stores_no_paths(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "SecretClient", "Pads"))
            write_wav(os.path.join(d, "SecretClient", "Pads", "warm_pad.wav"), tone(bright=0.2))
            write_wav(os.path.join(d, "glass_bell.wav"), tone(880, 1.0))
            open(os.path.join(d, "notes.txt"), "w").close()
            idx, rep = L.scan(d, "salt")
            self.assertEqual(rep["added"], 2)
            dump = json.dumps(idx)
            for secret in (d, "warm_pad", "glass_bell", "SecretClient", ".wav"):
                self.assertNotIn(secret, dump)
            idx2, rep2 = L.scan(d, "salt", idx)
            self.assertEqual((rep2["added"], rep2["unchanged"]), (0, 2))
            self.assertEqual(L.forget(idx2, "salt", d), [])
            other, _ = L.scan(d, "another-salt")
            self.assertNotEqual({e["key"] for e in other}, {e["key"] for e in idx})


def _patch(v=0.5, method="subtractive"):
    return {"method": method, "pair": None, "knobs": [v] * 32}


class Events(unittest.TestCase):
    def log(self):
        a, b, c = _patch(0.1), _patch(0.5), _patch(0.9)
        c_edit = dict(c, knobs=[0.7] * 32)
        return [
            {"t": 0, "type": "load", "inst": "i", "patch": a, "source": "generate", "project": "p1"},
            {"t": 3, "type": "play", "inst": "i", "seconds": 1, "notes": 2, "pitch_mean": 40, "len_mean": 0.3},
            {"t": 5, "type": "load", "inst": "i", "patch": b, "source": "browser", "project": "p1"},
            {"t": 60, "type": "play", "inst": "i", "seconds": 50, "notes": 40, "pitch_mean": 40, "len_mean": 0.3},
            {"t": 400, "type": "project_save", "inst": "i"},
            {"t": 500, "type": "load", "inst": "j", "patch": c, "source": "browser", "project": "p2"},
            {"t": 520, "type": "edit", "inst": "j", "patch": c_edit},
            {"t": 600, "type": "play", "inst": "j", "seconds": 90, "notes": 30, "pitch_mean": 80, "len_mean": 2},
            {"t": 700, "type": "save_preset", "inst": "j", "patch": c_edit},
            {"t": 800, "type": "load", "inst": "k", "patch": b, "source": "project", "project": "p3"},
        ]

    def test_labels(self):
        ex = E.exposures(self.log())
        self.assertEqual(len(ex), 4)
        self.assertEqual(ex[1].context(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(ex[1].reuses, 1)                 # b loaded again under another project
        points, pairs = E.labels(ex)
        ys = {E.fingerprint(p): y for p, _, y, _ in points}
        self.assertEqual(ys[E.fingerprint(_patch(0.1))], -1)
        self.assertEqual(ys[E.fingerprint(_patch(0.5))], 1)
        self.assertEqual(ys[E.fingerprint(_patch(0.7))], 1)   # praise goes to the edited patch
        kinds = {(E.fingerprint(a), E.fingerprint(b)) for a, b, _, _ in pairs}
        self.assertIn((E.fingerprint(_patch(0.5)), E.fingerprint(_patch(0.1))), kinds)   # replacement
        self.assertIn((E.fingerprint(_patch(0.7)), E.fingerprint(_patch(0.9))), kinds)   # edit

    def test_browser_skip_is_weaker(self):
        log = [{"t": 0, "type": "load", "inst": "i", "patch": _patch(0.1), "source": "browser"},
               {"t": 3, "type": "load", "inst": "i", "patch": _patch(0.2), "source": "browser"},
               {"t": 6, "type": "load", "inst": "i", "patch": _patch(0.3), "source": "generate"},
               {"t": 9, "type": "load", "inst": "i", "patch": _patch(0.4), "source": "browser"}]
        ex = E.exposures(log)
        self.assertLess(ex[0].reward()[1], ex[2].reward()[1])


    def test_init_replacement_is_not_a_judgement(self):
        log = [{"t": 0, "type": "load", "inst": "i", "patch": _patch(0.1), "source": "init"},
               {"t": 3, "type": "load", "inst": "i", "patch": _patch(0.2), "source": "user"}]
        self.assertEqual(E.exposures(log)[0].reward()[1], 0.0)


class Consent(unittest.TestCase):
    def test_default_is_off(self):
        self.assertFalse(C.normalise(None)["learn"])
        self.assertFalse(C.read("/does/not/exist")["learn"])
        self.assertEqual(C.filter_events(Events().log(), None), [])
        self.assertFalse(any(C.normalise({"categories": {"workflow": True}})["categories"].values()))

    def test_plugin_switch_means_every_category(self):
        c = C.normalise({"learn": True})
        self.assertTrue(all(c["categories"].values()))

    def test_categories_filter_what_counts(self):
        log = Events().log()
        full = E.labels(E.exposures(C.filter_events(log, {"learn": True})))
        self.assertTrue(full[0])
        c = {"learn": True, "categories": {"performance": False, "parameters": False,
                                           "suggestions": False, "workflow": True}}
        evs = C.filter_events(log, c)
        self.assertFalse(any(e["type"] == "edit" for e in evs))
        self.assertFalse(any("pitch_mean" in e for e in evs))
        ex = E.exposures(evs)
        self.assertTrue(ex[0].unlabelled)                  # a is from Generate
        self.assertEqual(ex[1].context(), [0.0] * 4)
        points, pairs = E.labels(ex)
        self.assertNotIn(E.fingerprint(_patch(0.1)), {E.fingerprint(p) for p, _, _, _ in points})
        # workflow off: play time, keeps, saves, reuse and skips give nothing
        c = {"learn": True, "categories": {"performance": True, "parameters": True,
                                           "suggestions": True, "workflow": False}}
        ex = E.exposures(C.filter_events(log, c))
        self.assertTrue(all(e.reward() == (0.0, 0.0) for e in ex))

    def test_profile_respects_consent_and_resets(self):
        R = SketchRenderer()
        with tempfile.TemporaryDirectory() as d:
            prof = Profile.create(os.path.join(d, "p"))
            prof.settings["n_background"] = 16
            st = prof.fit(R, Events().log())
            self.assertEqual(st["points"], 0)
            st = prof.fit(R, Events().log(), consent={"learn": True})
            self.assertGreater(st["points"], 0)
            self.assertTrue(prof.liked)
            prof.save()
            prof = Profile.load(prof.path)
            self.assertTrue(prof.consent["learn"])
            self.assertIn("From behaviour", "\n".join(prof.explain()))
            prof.reset()
            self.assertIsNone(prof.model)
            self.assertEqual(prof.liked, [])


class OwnWords(unittest.TestCase):
    def test_record_words_weights(self):
        w = T.record_words("Pads/Big_Pad.wav", tags=["Cinematic"], description="a lonely, cold texture",
                           labels=["warm"])
        self.assertEqual(w["pad"], 1.0)
        self.assertEqual(w["cinematic"], T.TAG_WEIGHT)
        self.assertEqual(w["lonely"], T.DESCRIPTION_WEIGHT)
        self.assertEqual(w["warm"], T.LABEL_WEIGHT)

    def test_labels_teach_words_and_unknown_words_are_ignored(self):
        R = SketchRenderer()
        with tempfile.TemporaryDirectory() as d:
            prof = Profile.create(os.path.join(d, "p"))
            prof.settings["n_background"] = 16
            for i in range(4):
                path = os.path.join(d, "x%d.wav" % i)
                write_wav(path, tone(220, 0.9 if i < 2 else 0.1))
                prof.label(path, ["lonely"] if i < 2 else ["cold"])
            prof.fit(R)
            self.assertEqual(set(prof.words.words), {"lonely", "cold"})
            known, unknown = prof.parse_text("lonely shimmer")
            self.assertEqual(known, [("lonely", 1.0)])
            self.assertEqual(unknown, [("shimmer", 1.0)])
            p = G.sample_prior(np.random.default_rng(0), 1)[0]
            base = prof.score_fn(R)([p])[0]
            self.assertAlmostEqual(prof.score_fn(R, text="shimmer")([p])[0], base)

            class Guess:
                def score(self, word, descriptor):
                    return 10.0 if word == "shimmer" else -10.0
            prof.general_text = Guess()
            self.assertGreater(prof.score_fn(R, text="shimmer")([p])[0], base)
            # a word the user has used is never overridden by the general model
            self.assertAlmostEqual(prof.score_fn(R, text="lonely")([p])[0],
                                   base + prof.words.query_score(prof.z(p, [0.0] * 4, R), [("lonely", 1.0)]))
            prof.forget_word("cold")
            self.assertNotIn("cold", prof.words.words)


class Nearby(unittest.TestCase):
    def test_modes(self):
        R = SketchRenderer()
        with tempfile.TemporaryDirectory() as d:
            prof = Profile.create(os.path.join(d, "p"))
            prof.settings["n_background"] = 16
            start = {"method": "subtractive", "pair": None, "knobs": [0.5] * 32}
            out, used = prof.randomize_nearby(start, R)
            self.assertEqual(used, "model")                # nothing learned yet
            prof.fit(R, Events().log(), consent={"learn": True})
            out, used = prof.randomize_nearby(start, R, k=4, n=24)
            self.assertEqual(used, "my_taste")
            self.assertEqual(len(out), 4)
            self.assertTrue(all(o["patch"]["method"] == "subtractive" for o in out))
            dist = lambda o: np.abs(np.array(o["patch"]["knobs"]) - 0.5).mean()
            wild, _ = prof.randomize_nearby(start, R, "wild", k=4, n=24)
            self.assertGreater(np.mean([dist(o) for o in wild]), np.mean([dist(o) for o in out]))
            _, used = prof.randomize_nearby(start, R, "community")
            self.assertEqual(used, "model")                # no community profile
            _, used = prof.randomize_nearby(start, R, "community", community=prof)
            self.assertEqual(used, "community")
            with self.assertRaises(ValueError):
                prof.randomize_nearby(start, R, "loud")
            prof.consent["personalization"] = 0.0
            self.assertEqual(prof.randomize_nearby(start, R)[1], "model")

    def test_taste_pulls_toward_liked(self):
        rng = np.random.default_rng(0)
        start = {"method": "fm", "pair": None, "knobs": [0.2] * 32}
        liked = [{"method": "fm", "pair": None, "knobs": [0.9] * 32}]
        fn = lambda ps: [float(np.mean(p["knobs"])) * 20 for p in ps]
        taste = G.nearby(start, fn, rng, k=8, n=64, beta=1.0, pull=0.25, liked=liked)
        plain = G.nearby(start, None, rng, k=8, n=64, beta=0.0)
        self.assertGreater(np.mean([np.mean(p["knobs"]) for p, _, _ in taste]),
                           np.mean([np.mean(p["knobs"]) for p, _, _ in plain]))


class Models(unittest.TestCase):
    def test_scorer_recovers_taste(self):
        rng = np.random.default_rng(0)
        d = 8
        u = rng.standard_normal(d)
        Z = rng.standard_normal((300, d))
        util = Z @ u
        rows = [(features(z, None, [0] * 4), 1.0 if s + 0.5 * rng.standard_normal() > 0 else -1.0, 1.0)
                for z, s in zip(Z[:200], util[:200])]
        m = PreferenceModel(d).fit(rows)
        s = [m.score(z, None, [0] * 4) for z in Z[200:]]
        self.assertGreater(auc(s, util[200:]), 0.9)
        th = m.sample_theta(rng)
        self.assertEqual(th.shape, m.theta.shape)

    def test_pairs_only_and_context(self):
        rng = np.random.default_rng(1)
        d = 6
        Z = rng.standard_normal((400, d))
        bass, lead = [1, 0, 0, 0], [0, 0, 1, 0]
        util = lambda z, c: (z[0] if c is lead else -z[0])        # bright good in leads, bad in bass
        rows = []
        for i in range(0, 400, 2):
            c = bass if i % 4 else lead
            a, b = Z[i], Z[i + 1]
            if util(a, c) < util(b, c):
                a, b = b, a
            rows.append((features(a, None, c) - features(b, None, c), 1.0, 1.0))
        m = PreferenceModel(d).fit(rows)
        hi, lo = np.eye(d)[0] * 2, -np.eye(d)[0] * 2
        self.assertGreater(m.score(hi, None, lead), m.score(lo, None, lead))
        self.assertLess(m.score(hi, None, bass), m.score(lo, None, bass))

    def test_empty_model_is_neutral(self):
        m = PreferenceModel(4).fit([])
        self.assertEqual(m.score(np.ones(4), "fm", [0, 1, 0, 0]), 0.0)

    def test_words(self):
        rng = np.random.default_rng(2)
        Z, tags = [], []
        for i in range(60):
            z = rng.standard_normal(4) * 0.3
            if i % 3 == 0:
                z[0] -= 2; tags.append({"dark": 1.0, "bass": 1.0})
            elif i % 3 == 1:
                z[0] += 2; tags.append({"bright": 1.0, "lead": 0.5})
            else:
                tags.append({"pad": 1.0})
            Z.append(z)
        wm = WordModel().fit(np.array(Z), tags)
        lo, hi = np.array([-2.0, 0, 0, 0]), np.array([2.0, 0, 0, 0])
        self.assertGreater(wm.llr(lo, "dark"), wm.llr(hi, "dark"))
        known, unknown = wm.query("bright not dark shimmery")
        self.assertEqual(known, [("bright", 1.0), ("dark", -1.0)])
        self.assertEqual(unknown, ["shimmery"])
        self.assertGreater(wm.query_score(hi, known), wm.query_score(lo, known))
        self.assertIn("bass", [w for w, _ in wm.related("dark")])
        self.assertEqual(wm.top_words(hi, 1)[0][0] in ("bright", "lead"), True)


class Generation(unittest.TestCase):
    def test_prior_matches_training_probs(self):
        p = G.method_probs()
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertGreater(p["sampling"], p["spectral"])
        pats = G.sample_prior(np.random.default_rng(0), 2000)
        self.assertFalse(any(x["pair"] and set(x["pair"]) == {"additive", "physical"} for x in pats))
        self.assertTrue(all(len(x["knobs"]) == 32 and 0 <= min(x["knobs"]) <= max(x["knobs"]) <= 1 for x in pats))
        self.assertTrue(all((x["pair"] is None) == (x["method"] != "hybrid") for x in pats))

    def test_propose_prefers_high_scores_and_explores(self):
        rng = np.random.default_rng(0)
        fn = lambda ps: [p["knobs"][0] * 10 for p in ps]
        out = G.propose(fn, rng, n=200, k=10, beta=2.0, cem_iters=1)
        self.assertEqual(len(out), 10)
        self.assertEqual(sum(o == "explore" for _, _, o in out), 2)
        chosen = [p["knobs"][0] for p, _, o in out if o != "explore"]
        self.assertGreater(np.mean(chosen), 0.7)


class ProfileEndToEnd(unittest.TestCase):
    def test_library_words_steer_generation(self):
        R = SketchRenderer()
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as d:
            lib, prof_dir = os.path.join(d, "lib"), os.path.join(d, "profile")
            os.makedirs(lib)
            for i in range(12):
                k = rng.random(32)
                bright = i % 2 == 0
                k[0] = 0.9 + 0.1 * k[0] if bright else 0.1 * k[0]
                k[6] = 0.0
                name = ("bright_lead_%d.wav" if bright else "dark_bass_%d.wav") % i
                write_wav(os.path.join(lib, name), R({"method": "additive", "pair": None, "knobs": k.tolist()}))
            prof = Profile.create(prof_dir)
            prof.settings["n_background"] = 48
            self.assertEqual(prof.scan(lib)["added"], 12)
            prof.fit(R)
            prof.save()
            prof2 = Profile.load(prof_dir)
            self.assertEqual(sorted(prof2.words.words), ["bass", "bright", "dark", "lead"])
            p = G.sample_prior(rng, 1)[0]
            ctx = [0.0] * 4
            self.assertAlmostEqual(prof.score_fn(R, ctx)([p])[0], prof2.score_fn(R, ctx)([p])[0], places=6)
            bright = prof2.generate(R, text="bright", k=6, n=48, seed=1, thompson=False)
            dark = prof2.generate(R, text="dark", k=6, n=48, seed=1, thompson=False)
            kb = np.mean([o["patch"]["knobs"][0] for o in bright if o["origin"] != "explore"])
            kd = np.mean([o["patch"]["knobs"][0] for o in dark if o["origin"] != "explore"])
            self.assertGreater(kb, kd)


class Batch(unittest.TestCase):
    def test_control_center_batch(self):
        import contextlib
        import io
        from alfapref import cli
        R = SketchRenderer()
        with tempfile.TemporaryDirectory() as d:
            home = os.path.join(d, "home", "preference")
            lib = os.path.join(d, "Private Samples", "Pads")
            os.makedirs(lib)
            os.makedirs(os.path.join(home, "events"))
            recs = []
            for i, name in enumerate(["warm_pad_03.wav", "warm_pad_04.wav", "glass_bell.wav", "x.flac"]):
                path = os.path.join(lib, name)
                if name.endswith(".wav"):
                    write_wav(path, tone(220 * (i + 1), 0.2 * i))
                recs.append({"path": path, "library": "Private Samples", "rel": "Pads/" + name,
                             "name_labels": [], "folder_labels": []})
            with open(os.path.join(home, "library_index.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)
            evs = Events().log()
            for day, chunk in (("2026-09-28", evs[:5]), ("2026-09-29", evs[5:])):
                with open(os.path.join(home, "events", day + ".jsonl"), "w") as f:
                    f.writelines(json.dumps(e) + "\n" for e in chunk)
                    f.write('{"t": 9, "type": "pla')              # torn last line is skipped
            args = ["batch", "--index", os.path.join(home, "library_index.jsonl"),
                    "--events", os.path.join(home, "events"), "--background", "16"]
            off = os.path.join(home, "profiles", "off")
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(args + ["--out", off])
            summ = json.loads(Path(off, "summary.json").read_text())
            self.assertEqual(summ["fit"]["points"], 0)          # no plugin.json: learning is off
            self.assertFalse(summ["learned"]["consent"]["learn"])
            with open(os.path.join(home, "plugin.json"), "w") as f:
                json.dump({"learn": True}, f)
            out = os.path.join(home, "profiles", "20260929T2000")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli.main(args + ["--out", out])
            lines = [l for l in buf.getvalue().splitlines() if l.startswith("progress")]
            self.assertEqual(lines[-1], "progress %d/%d" % (4 + 16 + 1, 4 + 16 + 1))
            summ = json.loads(Path(out, "summary.json").read_text())
            self.assertEqual(summ["files"]["added"], 3)
            self.assertEqual(summ["files"]["skipped_format"], 1)
            self.assertGreater(summ["fit"]["points"], 0)
            self.assertGreater(summ["learned"]["behaviour"]["sounds_heard"], 0)
            self.assertIn("warm", summ["words"])
            for fn in os.listdir(out):
                text = Path(out, fn).read_text()
                for secret in ("Private Samples", "warm_pad_03", d):
                    self.assertNotIn(secret, text)
            prof = Profile.load(out)
            self.assertEqual(len(prof.generate(R, k=3, n=12)), 3)


class Assimilation(unittest.TestCase):
    def test_batch_reads_plugin_features(self):
        import contextlib
        import io
        from alfapref import cli
        with tempfile.TemporaryDirectory() as d:
            home = os.path.join(d, "home")
            pdir = os.path.join(home, "assimilation", "Vendor-SuperSynth-vst3")
            os.makedirs(pdir)
            os.makedirs(os.path.join(home, "preference"))
            presets = [{"key": "p0000", "kind": "program", "name": "Dark Bass Growl", "category": ""},
                       {"key": "f00000", "kind": "vstpreset", "name": "Glass Bells", "category": "Mallets/Bells"},
                       {"key": "default", "kind": "default", "name": "default", "category": ""}]
            with open(os.path.join(pdir, "plugin.json"), "w") as f:
                json.dump({"name": "SuperSynth", "vendor": "Vendor", "presets": presets}, f)
            with open(os.path.join(pdir, "presets.json"), "w") as f:
                json.dump({p["key"]: {"name": p["name"], "status": "ranged"} for p in presets}, f)
            with open(os.path.join(pdir, "features.jsonl"), "w") as f:
                for key, f0, bright, notes in (("p0000", 55, 0.1, 6), ("f00000", 880, 1.0, 2), ("default", 220, 0.5, 1)):
                    for n in range(notes):
                        f.write(json.dumps({"preset": key, "note": "x", "midi": 40 + n, "velocity": 100,
                                            "dur_s": 1.0, "descriptor_id": D.DESCRIPTOR_ID,
                                            "descriptor": D.descriptor(tone(f0 * (1 + n / 12), bright)).tolist()}) + "\n")
                f.write(json.dumps({"preset": "p0000", "midi": 99, "descriptor_id": "other-v9",
                                    "descriptor": [0.0] * 5}) + "\n")
                f.write('{"preset": "p00')
            idx = os.path.join(home, "preference", "library_index.jsonl")
            open(idx, "w").close()
            out = os.path.join(home, "preference", "profiles", "t")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli.main(["batch", "--index", idx, "--assimilation", os.path.join(home, "assimilation"),
                          "--out", out, "--background", "16"])
            progress = [l for l in buf.getvalue().splitlines() if l.startswith("progress")]
            self.assertEqual(progress[-1], "progress %d/%d" % (9 + 16 + 1, 9 + 16 + 1))
            summ = json.loads(Path(out, "summary.json").read_text())
            self.assertEqual(summ["files"]["assimilation"], {"notes": 9, "presets": 3, "plugins": 1})
            self.assertEqual(summ["words"]["dark"], 0.5)       # six notes of one preset count once
            self.assertEqual(summ["words"]["bell"], 0.5)
            self.assertEqual(summ["words"]["mallet"], 0.25)       # category folder words weigh half
            self.assertNotIn("default", summ["words"])
            for fn in os.listdir(out):
                self.assertNotIn("SuperSynth", Path(out, fn).read_text())
            prof = Profile.load(out)
            self.assertAlmostEqual(sum(e["weight"] for e in prof.index), 1.5)

    def test_missing_folder_is_empty(self):
        self.assertEqual(L.from_assimilation(None, "s"), ([], {"notes": 0, "presets": 0, "plugins": 0}))


@unittest.skipUnless(os.environ.get("ALFA_SLOW"), "set ALFA_SLOW=1 to run the simulated user")
class Simulation(unittest.TestCase):
    def test_simulated_user(self):
        from alfapref import simulate
        h = simulate.run(40, eval_every=40, n_background=96, n_eval=80, n_candidates=48, log=None)
        self.assertGreater(h[-1]["auc"], 0.65)
        self.assertGreater(h[-1]["proposal_utility"], 0.3)


if __name__ == "__main__":
    unittest.main()
