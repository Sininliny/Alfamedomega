"""Spectral reconstruction distance, training coordinates and the offline parameter metric.

NumPy only, so these run without PyTorch.
"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from training.aom.perceptual import (Coordinates, spectral_distance, parameter_metric,
                                     save_metric, load_metric, free_coordinates)

RATE = 16000
TIME = np.arange(8000) / RATE


def tone(frequency, partials=6, phase=0.0):
    return (sum(np.sin(2 * np.pi * k * frequency * TIME + phase) / k for k in range(1, partials + 1)) * 0.2)[:, None]


def descend(target, start, partials):
    """Sign descent on log frequency with central differences, as a learner would see it."""
    reference, x = tone(target, partials), np.log(start)
    for step in range(140):
        slope = (spectral_distance(tone(np.exp(x + 1e-3), partials), reference, RATE)
                 - spectral_distance(tone(np.exp(x - 1e-3), partials), reference, RATE))
        x -= 0.04 * 0.97 ** step * np.sign(slope)
    return np.exp(x)


class SpectralDistanceTests(unittest.TestCase):
    def test_zero_on_identity_and_insensitive_to_phase(self):
        self.assertEqual(spectral_distance(tone(440), tone(440), RATE), 0.0)
        shifted = tone(440, phase=1.3)
        waveform = float(np.mean((shifted - tone(440)) ** 2) / np.mean(tone(440) ** 2))
        self.assertGreater(waveform, 0.5)
        self.assertLess(spectral_distance(shifted, tone(440), RATE), 0.05 * spectral_distance(tone(470), tone(440), RATE))

    def test_pitch_descent_converges_from_far_starts(self):
        # Sample-domain MSE converges from 4 of 32 such starts (proposal measurement).
        for target, partials in ((110, 8), (440, 1), (1000, 3)):
            for start in (target / 1.8, target * 1.8):
                self.assertLess(abs(descend(target, start, partials) / target - 1), 0.01, (target, start))

    def test_timbre_changes_are_measured(self):
        brighter = tone(440) + 0.05 * np.sin(2 * np.pi * 2640 * TIME)[:, None]
        self.assertGreater(spectral_distance(brighter, tone(440), RATE), 1e-3)

    def test_rejects_mismatched_or_nonfinite_audio(self):
        with self.assertRaises(ValueError):
            spectral_distance(tone(440)[:100], tone(440), RATE)
        with self.assertRaises(ValueError):
            spectral_distance(np.full((10, 1), np.nan), np.zeros((10, 1)), RATE)


class CoordinateTests(unittest.TestCase):
    def test_log_roundtrip_and_bounds(self):
        coordinates = Coordinates(["linear", "log"], [[-1, 1], [20, 20000]])
        p = np.array([0.25, 440.0])
        np.testing.assert_allclose(coordinates.to_p(coordinates.to_u(p)), p)
        low, high = coordinates.u_bounds()
        np.testing.assert_allclose(low, [-1, np.log(20)])
        np.testing.assert_allclose(high, [1, np.log(20000)])
        self.assertFalse(coordinates.identity)

    def test_log_requires_positive_bounds(self):
        with self.assertRaises(ValueError):
            Coordinates(["log"], [[0, 1]])
        with self.assertRaises(ValueError):
            Coordinates(["cubic"], [[0, 1]])

    def test_manifest_default_is_linear_and_unbounded(self):
        coordinates = Coordinates.from_manifest({}, 3)
        self.assertTrue(coordinates.identity)
        with self.assertRaises(ValueError):
            Coordinates.from_manifest({"parameter_coordinates": ["log"]}, 3)


class ParameterMetricTests(unittest.TestCase):
    def test_metric_ignores_inaudible_directions_and_stays_inside_bounds(self):
        coordinates = Coordinates(["log", "linear", "linear"], [[100, 1000], [0, 1], [0, 1]])
        seen = []
        def render(p):
            seen.append(p.copy())
            return tone(p[0]) * p[1]                 # p[2] is not used by the sound
        u = coordinates.to_u(np.array([440.0, 1.0, 0.5]))
        metric = parameter_metric(render, u, [True, True, True], coordinates, step=1e-3, rank=3, sample_rate=RATE)
        U, S = metric["U"], metric["S"]
        G = U @ np.diag(S ** 2) @ U.T
        self.assertEqual(U.shape, (3, 3))
        self.assertAlmostEqual(G[2, 2], 0.0, places=12)
        self.assertGreater(G[0, 0], 0)
        self.assertGreater(G[1, 1], 0)
        # Amplitude sits on its upper bound: the difference is one-sided, never above 1.
        self.assertLessEqual(max(p[1] for p in seen), 1.0)

    def test_locked_or_absent_parameters_are_not_free(self):
        document = {"tensors": {"parameters": {"values": [1, 2, 3, 4], "presence": [True, True, False, True]}},
                    "parameter_locks": [1, 0, 0, 0]}
        np.testing.assert_array_equal(free_coordinates(document, [[0, 5], [0, 5], [0, 5], [4, 4]]),
                                      [False, True, False, False])

    def test_descriptor_rows_and_file_roundtrip(self):
        coordinates = Coordinates(["linear", "linear"], [[0, 1], [0, 1]])
        render = lambda p: tone(440) * p[0] + tone(880) * p[1]
        descriptors = lambda audio: (np.array([audio[:, 0].std(), 0.0]), np.array([True, False]))
        metric = parameter_metric(render, np.array([0.5, 0.2]), [True, True], coordinates,
                                  step=1e-3, rank=1, sample_rate=RATE, descriptors=descriptors)
        self.assertEqual(metric["Jd"].shape, (2, 2))
        self.assertTrue((metric["Jd"][1] == 0).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.npz"
            save_metric(path, metric)
            loaded = load_metric(path, 2, 4)
            self.assertEqual(loaded["U"].shape, (2, 4))
            np.testing.assert_allclose(loaded["S"][:1], metric["S"])
            self.assertTrue((loaded["S"][1:] == 0).all())
            with self.assertRaises(ValueError):
                load_metric(path, 3, 4)


class NativeMetricTests(unittest.TestCase):
    def test_producer_writes_metric_factors_for_both_heads(self):
        from training.aom.native import _library_path
        from training.aom.smoke import fixture as native_fixture
        from training.aom.producer import build_dataset, INPUTS
        from training.aom.session import atomic_json
        try:
            library = _library_path()
        except FileNotFoundError:
            self.skipTest("Native compiler build required")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            document = native_fixture(.15)
            document["tensors"]["parameters"]["values"][1] = .35
            atomic_json(folder / "source.json", document)
            schemas = {k: {"axes": t["axes"], "padded_shape": t["shape"]} for k, t in document["tensors"].items()}
            bounds = {"activation": [[0., 1.], [0., 1.]], "parameters": [[200., 200.], [.01, .5], [400., 400.], [.01, .5]],
                      "routing": [], "modulation": []}
            plan = {"schema": "aom.dataset-plan/1", "ontology": "aom.math/1", "bounds": bounds, "schemas": schemas,
                    "groups": [{"id": "g", "split": "train", "canonical_json": "source.json", "variants": [
                        {"id": "e", "perturbation_id": "p", "updates": {"parameters": [{"index": 3, "value": .2}]},
                         "render": {"frames": 1600, "sample_rate": 16000, "inputs": {k: [] for k in INPUTS}}}]}],
                    "analysis": {"frames": 2, "harmonic_ratios": [1., 2.], "minimum_fundamental_hz": 100.,
                                 "maximum_fundamental_hz": 300., "recording_policy": "controlled_render"},
                    "normalization": {"policy_id": "t", "removed": ["pitch", "loudness", "recording_condition"],
                                      "preserved": ["timbre", "articulation", "temporal_behavior"]},
                    "learn": {"hidden_width": 4, "steps": 1, "learning_rate": .01, "primary_weight": 1., "secondary_weight": 1.,
                              "mask_weight": .1, "available_ram_bytes": 10_000_000, "seed": 1},
                    "require_validation": False,
                    "parameter_coordinates": ["linear", "log", "linear", "log"],
                    "perceptual_metric": {"rank": 2, "finite_difference_step": 1e-3, "weight": 1.0,
                                          "descriptors": "aom.wave-features/1"}}
            atomic_json(folder / "plan.json", plan)
            result = build_dataset(folder / "plan.json", folder / "dataset", dll_path=library)
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(manifest["parameter_coordinates"], plan["parameter_coordinates"])
            row = manifest["examples"][0]
            for head in ("canonical", "instance"):
                metric = load_metric(folder / "dataset" / row["metric"][head], 4, 2)
                np.testing.assert_array_equal(metric["free_mask"], [False, True, False, True])
                G = metric["U"] @ np.diag(metric["S"] ** 2) @ metric["U"].T
                self.assertEqual(G[0, 0], 0.0)
                self.assertEqual(G[2, 2], 0.0)
                self.assertGreater(G[3, 3], 0.0)
                self.assertEqual(metric["Jd"].shape, (8, 4))
                np.testing.assert_allclose(metric["u_ref"][[1, 3]], np.log([.35, .15 if head == "canonical" else .2]))


if __name__ == "__main__":
    unittest.main()
