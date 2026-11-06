"""Mix scenes: every ladder level is an exact, renderable explanation of the same mix."""

import unittest

import numpy as np

from training.amo.corpus import mixscene

try:
    from training.amo.native import _library_path

    _library_path(str(mixscene.prepare.LIB) if mixscene.prepare.LIB.is_file() else None)
    NATIVE = True
except (FileNotFoundError, OSError):
    NATIVE = False


class SceneSpecTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_bounded(self):
        for index in range(200):
            a = mixscene.sample_scene(mixscene.SEED, index)
            self.assertEqual(a, mixscene.sample_scene(mixscene.SEED, index))
            atoms = mixscene.atoms_of(a)
            levels = mixscene.ladder(a, atoms)
            counts = [max(levels[k]) + 1 for k in ("L0", "L1", "L2")]
            self.assertEqual(counts[0], len(a["instruments"]))
            self.assertTrue(1 <= counts[0] <= counts[1] <= counts[2] <= mixscene.MAX_INSTRUMENTS, counts)
            self.assertLessEqual(len(atoms), mixscene.MAX_INSTRUMENTS)
            for inst in a["instruments"]:
                onsets = [n[0] for n in inst["notes"]]
                self.assertEqual(onsets, sorted(onsets))
                names = [f["type"] for f in inst["insert_fx"]]
                if "echo" in names:
                    self.assertEqual(names[-1], "echo")  # time FX last

    def test_echo_taps_are_delayed_transposed_copies(self):
        notes = [[0.0, 0.2, 60, 0.8], [0.5, 0.2, 62, 0.7]]
        echo = {"taps": 2, "delay": 0.25, "transpose": 7, "decay": 0.5}
        self.assertEqual(mixscene.echo_notes(notes, echo, 2), [[0.5, 0.2, 74, 0.8], [1.0, 0.2, 76, 0.7]])

    def test_midi_tensor_retriggers_legato_notes(self):
        midi = mixscene.midi_tensor([[0.0, 0.5, 60, 0.8], [0.5, 0.5, 64, 0.6]], 24000)
        gap = int(mixscene.GATE_GAP * mixscene.SAMPLE_RATE)
        self.assertEqual(midi[12000 - gap, 2], 0.0)
        self.assertEqual(midi[12000, 2], 1.0)
        self.assertAlmostEqual(float(midi[12000, 0]), 440 * 2 ** (-5 / 12), places=2)


@unittest.skipUnless(NATIVE, "alfa_tensor_c is not built")
class SceneRenderTests(unittest.TestCase):
    def test_layered_preset_equals_sum_of_its_layers(self):
        rng = np.random.default_rng(3)
        inst = {"layers": [{"family": "saw_lowpass", "u": rng.random(64).tolist(), "transpose": 0},
                           {"family": "sine_fm", "u": rng.random(64).tolist(), "transpose": 12}],
                "insert_fx": [mixscene.sample_fx(rng, "tanh"), mixscene.sample_fx(rng, "echo")],
                "notes": [[0.0, 0.3, 48, 0.8], [0.4, 0.4, 55, 0.6]], "note_echo": None, "gain_db": 0.0}
        frames = 24000
        parts = [mixscene.render_document(mixscene.layer_document(inst, i), frames, inst["notes"])
                 for i in range(2)]
        merged = mixscene.render_document(mixscene.preset_document(inst), frames, inst["notes"])
        self.assertGreater(np.abs(merged).max(), 1e-3)
        np.testing.assert_allclose(parts[0] + parts[1], merged, atol=1e-6)

    def test_scene_renders_and_bus_identity_is_exact(self):
        scene, stems, mix = mixscene.build_scene(mixscene.SEED, 0)
        self.assertTrue(np.isfinite(mix).all())
        self.assertAlmostEqual(float(np.abs(mix).max()), mixscene.PEAK, places=5)
        dry = np.sum(stems, axis=0).astype(np.float32)
        frames = len(dry)
        identity = mixscene.render_document(mixscene.bus_document([]), frames, audio=dry)
        np.testing.assert_allclose(identity, dry, atol=1e-6)
        features, rms = mixscene.mix_features(mix)
        self.assertEqual(features.shape[1], mixscene.MID_BANDS + mixscene.SIDE_BANDS)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
