"""Mix scenes: every ladder level is an exact, renderable explanation of the same mix."""

import unittest

import numpy as np

from training.amo.corpus import mixscene, mixtargets

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


class TargetTests(unittest.TestCase):
    def test_targets_have_fixed_shapes_and_ladder_counts(self):
        scene = mixscene.sample_scene(mixscene.SEED, 5)
        counts = [max(mixscene.ladder(scene, mixscene.atoms_of(scene))[k]) + 1 for k in mixtargets.LEVELS]
        for level, count in zip(mixtargets.LEVELS, counts):
            targets = mixtargets.scene_targets(scene, level)
            self.assertEqual(targets["count"], count)
            self.assertEqual(targets["roll"].shape, (mixscene.MAX_INSTRUMENTS, mixscene.MAX_FRAMES,
                                                     mixtargets.PITCHES, len(mixtargets.ROLL_CHANNELS)))
            self.assertEqual(targets["exists"].sum(), count)

    def test_notes_round_trip_through_the_roll(self):
        notes = [[0.013, 0.2, 60, 0.8], [0.5, 0.31, 64, 0.6], [1.7, 1.2, 40, 0.9]]
        roll = mixtargets.piano_roll(notes, mixscene.MAX_FRAMES)
        decoded = mixtargets.roll_notes(roll)
        np.testing.assert_allclose(np.asarray(decoded), np.asarray(notes), atol=1e-6)

    def test_used_dims_cover_every_coordinate_a_family_reads(self):
        for family in mixtargets.FAMILIES:
            mask = mixtargets.used_dims(family)
            self.assertTrue(mask.any(), family)
            u = np.random.default_rng(0).random(64)
            v = u.copy()
            v[~mask] = 0.123
            self.assertEqual(mixscene.prepare.make_document(family, u)[0],
                             mixscene.prepare.make_document(family, v)[0], family)


@unittest.skipUnless(NATIVE, "alfa_tensor_c is not built")
class DecodeRenderTests(unittest.TestCase):
    def test_oracle_targets_decode_to_the_same_mix(self):
        from training.amo.corpus import resynth

        scene, _, mix = mixscene.build_scene(mixscene.SEED, 2)
        reference = resynth.log_spectra(mix)
        for level in mixtargets.LEVELS:
            targets = mixtargets.scene_targets(scene, level)
            units = [mixtargets.target_as_prediction(targets, k) for k in range(targets["count"])]
            decoded = mixtargets.decode_scene(units, targets["bus_present"], targets["bus_values"], scene["seconds"])
            distance = resynth.distance(resynth.log_spectra(mixscene.render_scene(decoded)[1]), reference)
            self.assertLess(distance, 0.02, level)


def split_echo_scene(index):
    """A scene with note echo, decoded at L2 (every echo tap its own instrument)."""
    scene = mixscene.sample_scene(mixscene.SEED, index)
    targets = mixtargets.scene_targets(scene, "L2")
    units = [mixtargets.target_as_prediction(targets, k) for k in range(targets["count"])]
    return scene, mixtargets.decode_scene(units, targets["bus_present"], targets["bus_values"], scene["seconds"])


class EchoMergeTests(unittest.TestCase):
    def test_merges_fold_split_echo_taps_back_into_their_source(self):
        from training.amo.corpus import mixinfer

        exact = total = 0
        for index in range(200):
            scene, decoded = split_echo_scene(index)
            final = mixinfer.merge_sequence(decoded["instruments"])[-1]
            l1 = max(mixscene.ladder(scene, mixscene.atoms_of(scene))["L1"]) + 1
            total += 1
            exact += len(final) == l1
        self.assertGreaterEqual(exact / total, 0.95)  # the rest stay as intermediate candidates

    def test_merge_recovers_delay_transpose_and_taps(self):
        from training.amo.corpus import mixinfer

        lead = {"layers": [{"family": "saw_lowpass"}], "notes": [[0.1, 0.2, 60, 0.8], [0.6, 0.2, 62, 0.8],
                                                                 [1.1, 0.2, 64, 0.8]],
                "insert_fx": [], "note_echo": None, "gain_db": -6.0}
        taps = [dict(lead, notes=mixscene.echo_notes(lead["notes"], {"delay": 0.25, "transpose": 7}, j),
                     gain_db=-6.0 + 20 * np.log10(0.5 ** j)) for j in (1, 2)]
        other = dict(lead, layers=[{"family": "sine_fm"}], notes=taps[0]["notes"])
        merged = mixinfer.merge_sequence([lead, *taps, other])[-1]
        self.assertEqual(len(merged), 2)  # the other timbre playing the same notes is kept
        echo = merged[0]["note_echo"]
        self.assertEqual((echo["taps"], echo["transpose"]), (2, 7))
        self.assertAlmostEqual(echo["delay"], 0.25, places=3)
        self.assertAlmostEqual(echo["decay"], 0.5, places=3)


@unittest.skipUnless(NATIVE, "alfa_tensor_c is not built")
class EchoMergeRenderTests(unittest.TestCase):
    def test_merged_scene_renders_the_same_mix(self):
        from training.amo.corpus import mixinfer, resynth

        scene, decoded = split_echo_scene(3)
        target = resynth.log_spectra(mixscene.render_scene(scene)[1])
        states = mixinfer.merge_sequence(decoded["instruments"])
        self.assertGreater(len(states), 1)
        split, merged = [mixinfer.render_distance(({**decoded, "instruments": s}, target)) for s in states[::len(states) - 1]]
        self.assertLess(abs(split - merged), 1e-3)
        self.assertLess(mixinfer.cost_terms({**decoded, "instruments": states[-1]})["N"],
                        mixinfer.cost_terms(decoded)["N"])


if __name__ == "__main__":
    unittest.main()
