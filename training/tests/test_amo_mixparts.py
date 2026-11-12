"""Part encoder samples (MIX-DESIGN.md §13): features, labels and note inputs line up with the scene."""

import tempfile
import unittest

import numpy as np

from training.amo.corpus import mixparts, mixscene, mixtargets

try:
    from training.amo.native import _library_path

    _library_path(str(mixscene.prepare.LIB) if mixscene.prepare.LIB.is_file() else None)
    NATIVE = True
except (FileNotFoundError, OSError):
    NATIVE = False


class InputTests(unittest.TestCase):
    def test_roll_marks_notes_at_hop_1024(self):
        rate = mixscene.SAMPLE_RATE / mixparts.HOP
        roll = mixparts.roll([[1.0, 0.5, 60, 0.7], [0.0, 0.2, 200, 1.0]])
        p = 60 - mixtargets.PITCH_LOW
        start = int(round(rate))
        self.assertEqual(roll.shape, (mixparts.FRAMES, mixtargets.PITCHES, mixparts.ROLL))
        self.assertEqual(roll[start, p, 0], 1.0)
        self.assertEqual(roll[:, :, 0].sum(), 1.0)  # out-of-range pitch dropped
        self.assertAlmostEqual(float(roll[start + 3, p, 2]), 0.7, places=6)

    def test_jitter_is_seeded_and_small(self):
        notes = [[0.1 * i, 0.2, 60 + i, 0.8] for i in range(20)]
        a = mixparts.jitter_notes(notes, np.random.default_rng(3))
        self.assertEqual(a, mixparts.jitter_notes(notes, np.random.default_rng(3)))
        self.assertLessEqual(abs(len(a) - len(notes)), 1)
        self.assertEqual(a, sorted(a))

    def test_features_are_relative_to_the_part(self):
        t = np.arange(mixscene.SAMPLE_RATE) / mixscene.SAMPLE_RATE
        tone = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32)
        a, b = mixparts.part_features(tone), mixparts.part_features(0.01 * tone)
        self.assertEqual(a.shape, (mixparts.FRAMES, mixparts.FEATURES))
        loud = a > -4  # the silent Side channel sits at the float noise floor
        self.assertTrue(loud.any())
        np.testing.assert_allclose(a[loud], b[loud], atol=1e-3)
        self.assertIsNone(mixparts.part_features(np.zeros((1000, 2), np.float32)))

    def test_label_decodes_back_to_the_insert_chain(self):
        scene = mixscene.sample_scene(mixscene.SEED, 7)
        for unit in mixtargets.level_units(scene, "L2"):
            target = mixparts.label(unit)
            self.assertEqual(mixtargets.FAMILIES[target["family"]], unit["layers"][0]["family"])
            chain = mixparts.decode_chain(target["fx_present"], target["fx_values"])
            order = mixtargets.INSERT_FX.index
            self.assertEqual([f["type"] for f in chain], sorted([f["type"] for f in unit["insert_fx"]], key=order))
            for value, kind in enumerate(mixparts.VALUE_FX):
                if target["fx_values"][value]:
                    self.assertEqual(target["fx_present"][kind], 1.0)


@unittest.skipUnless(NATIVE, "native runtime library not built")
class BuildTests(unittest.TestCase):
    def test_every_l2_atom_becomes_one_sample(self):
        with tempfile.TemporaryDirectory() as root:
            mixparts.build(root, 200000, 4, 2, 1, mixscene.SEED)
            corpus = mixparts.load_corpus(root)
            rows = corpus["rows"]
            self.assertEqual(corpus["features"].shape[1:], (mixparts.FRAMES, mixparts.FEATURES))
            self.assertEqual(len(corpus["features"]), len(rows))
            for index in range(200000, 200004):
                scene = mixscene.sample_scene(mixscene.SEED, index)
                mine = [r for r in rows if r["scene"] == index]
                self.assertLessEqual(len(mine), len(mixscene.atoms_of(scene)))
                for row in mine:
                    self.assertEqual(row["split"], mixscene.split_of(index))
                    self.assertIn(row["source"], ("clean", "masked"))
            item = mixparts.Dataset(corpus, [0], 0)[0]
            self.assertEqual(item["x"].shape, (mixparts.FRAMES, mixparts.FEATURES))
            self.assertTrue(np.isfinite(item["x"]).all())


if __name__ == "__main__":
    unittest.main()
