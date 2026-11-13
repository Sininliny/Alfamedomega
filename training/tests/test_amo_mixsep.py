"""Separation network data (MIX-DESIGN.md §14): windows, notes and remix augmentation line up."""

import tempfile
import unittest

import numpy as np

from training.amo.corpus import mixsep, mixscene, mixtargets

try:
    from training.amo.native import _library_path

    _library_path(str(mixscene.prepare.LIB) if mixscene.prepare.LIB.is_file() else None)
    NATIVE = True
except (FileNotFoundError, OSError):
    NATIVE = False


class NoteTests(unittest.TestCase):
    def test_window_keeps_overlapping_notes_and_marks_carried_ones(self):
        notes = [[0.5, 1.0, 60, 0.8], [2.2, 0.3, 62, 0.7], [7.0, 0.2, 64, 0.6]]
        inside = mixsep.window_notes(notes, 1.0)
        self.assertEqual(inside, [[-0.5, 1.0, 60, 0.8], [1.2000000000000002, 0.3, 62, 0.7]])
        roll = mixsep.roll(inside)
        p60, p62 = 60 - mixtargets.PITCH_LOW, 62 - mixtargets.PITCH_LOW
        self.assertEqual(roll[:, p60, 0].sum(), 0)  # begun before the window: no onset
        self.assertEqual(roll[0, p60, 1], 1.0)  # but its sustain shows
        self.assertEqual(roll[:, p62, 0].sum(), 1.0)
        self.assertEqual(roll.shape, (mixsep.FRAMES, mixtargets.PITCHES, mixsep.ROLL))

    def test_note_f1(self):
        truth = [[0.1, 0.2, 60, 0.8], [0.5, 0.2, 62, 0.8]]
        self.assertEqual(mixsep.note_f1(truth, truth), 1.0)
        self.assertAlmostEqual(mixsep.note_f1(truth[:1], truth), 2 / 3)
        self.assertEqual(mixsep.note_f1([[0.1, 0.2, 61, 0.8]], truth), 0.0)

    def test_bands_cover_every_bin(self):
        self.assertEqual(sum(mixsep.BAND_WIDTHS), mixsep.BINS)
        self.assertEqual(len(mixsep.BAND_WIDTHS), 48)
        self.assertTrue(all(w > 0 for w in mixsep.BAND_WIDTHS))


@unittest.skipUnless(NATIVE, "native runtime library not built")
class BuildTests(unittest.TestCase):
    def test_windows_sum_to_the_mix_and_remix_is_consistent(self):
        with tempfile.TemporaryDirectory() as root:
            report = mixsep.build(root, 300000, 4, 2, 1, mixscene.SEED)
            self.assertGreater(report["kept"], 0)
            corpus = mixsep.load_corpus(root)
            for row in corpus["rows"]:
                mix, stems = mixsep.scene_audio(corpus, row)
                self.assertEqual(mix.shape, (mixsep.WINDOW,))
                self.assertEqual(len(stems), len(row["atoms"]))
                # stems pass the bus one at a time; the bus is nearly linear at these levels
                error = np.sqrt(np.mean((stems.sum(0) - mix) ** 2)) / np.sqrt(np.mean(mix ** 2))
                self.assertLess(error, 0.5)
            data = mixsep.Dataset(corpus, list(range(len(corpus["rows"]))), 0, remix=1.0)
            for i in range(len(data)):
                item = data[i]
                k = int(item["present"].sum())
                self.assertTrue(1 <= k <= mixsep.SLOTS)
                np.testing.assert_allclose(item["stems"][:k].sum(0), item["mix"], atol=1e-5)
                self.assertFalse(item["stems"][k:].any())
                self.assertEqual(item["roll"].shape, (mixsep.SLOTS, mixsep.FRAMES, mixtargets.PITCHES, mixsep.ROLL))


if __name__ == "__main__":
    unittest.main()
