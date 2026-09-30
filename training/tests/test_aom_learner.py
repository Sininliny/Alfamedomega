from dataclasses import replace
from pathlib import Path
import tempfile
import wave
import unittest
import numpy as np
import torch

from training.aom.analysis import AnalysisProfile, extract, read_wav
from training.aom.learner import LearnSpec, train, save, load, predict, masked_loss
from training.aom.pipeline import export_prediction, representation
from training.tests import test_aom_contracts


class LearnerTests(unittest.TestCase):
    def setUp(self):
        fixture = test_aom_contracts.TrainingContractTests()
        fixture.setUp()
        self.fixture = fixture

    def test_dual_model_learns_distinct_original_and_rendered_targets_and_roundtrips(self):
        torch.set_num_threads(1)
        fixture = self.fixture
        second = replace(fixture.example, example_id="render-2", rendered_instance=fixture.canonical)
        spec = LearnSpec(12, 150, 0.03, 2.0, 1.0, 0.1, 10000000, 7)
        model, metadata = train((fixture.example, second), [[1, 0], [0, 1]],
                                fixture.canonical.ontology, fixture.schemas, spec)
        self.assertLess(metadata["losses"][-1]["primary_canonical"], metadata["losses"][0]["primary_canonical"] / 100)
        self.assertLess(metadata["losses"][-1]["secondary_rendered_instance"], metadata["losses"][0]["secondary_rendered_instance"] / 100)
        before = predict(model, metadata, [1, 0], mask_threshold=0.5)
        self.assertAlmostEqual(before["primary_canonical"]["parameters"]["values"][2], 0.5, delta=0.03)
        self.assertAlmostEqual(before["secondary_rendered_instance"]["parameters"]["values"][2], 0.8, delta=0.03)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save(path, model, metadata)
            restored, meta = load(path)
            self.assertEqual(before, predict(restored, meta, [1, 0], mask_threshold=0.5))

    def test_memory_estimate_rejects_before_training(self):
        with self.assertRaises(ValueError):
            train((self.fixture.example,), [[1]], self.fixture.canonical.ontology,
                  self.fixture.schemas, LearnSpec(12, 1, 0.03, 2, 1, 0.1, 1, 7))

    def test_padding_does_not_affect_regression_loss(self):
        self.assertEqual(float(masked_loss(torch.tensor([[1., 100.]]), torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]]))), 0)

    def test_wav_decode_and_model_load_reject_insufficient_ram(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 10)
            with self.assertRaises(ValueError):
                read_wav(path, available_ram_bytes=1)
            model_path = Path(folder) / "model.pt"
            model_path.write_bytes(b"unreadable package that must be rejected before torch.load")
            with self.assertRaises(ValueError):
                load(model_path, available_ram_bytes=1)

    def test_reference_features_remove_gain_and_approximately_preserve_pitch_invariance(self):
        rate = 16000
        time = np.arange(rate) / rate
        envelope = np.where(time < 0.2, time / 0.2, np.exp(-(time - 0.2) * 2))
        def audio(frequency):
            return envelope * (np.sin(2 * np.pi * frequency * time) + 0.25 * np.sin(4 * np.pi * frequency * time))
        profile = AnalysisProfile(4, (1., 2., 3.), 100, 1000, "controlled_render")
        base = np.array(extract(audio(200), rate, profile))
        gain = np.array(extract(audio(200) * 0.1, rate, profile))
        pitch = np.array(extract(audio(400), rate, profile))
        np.testing.assert_allclose(base, gain, atol=1e-10)
        np.testing.assert_allclose(base, pitch, atol=0.03)
        changed = np.array(extract(audio(200)[::-1], rate, profile))
        self.assertGreater(float(np.max(np.abs(base[1:5] - changed[1:5]))), 0.1)
        with self.assertRaises(ValueError):
            extract(audio(200), rate, replace(profile, recording_policy="unknown"))

    def test_export_preserves_locks_and_zeros_absent_dimensions(self):
        tensor = {"axes": ["parameter"], "shape": [2], "values": [0.3, 0.7]}
        template = {"ontology": "test", "tensors": {"parameters": tensor}, "parameter_locks": [1, 0]}
        predicted = {"ontology": "test", "primary_canonical": {"parameters": {**tensor, "values": [0.9, 0.8], "mask": [False, False]}}}
        exported = export_prediction(template, predicted, head="primary_canonical", bounds={"parameters": [(0, 1), (0, 1)]})
        self.assertEqual(exported["tensors"]["parameters"]["values"], [0.3, 0.0])
        self.assertEqual(exported["tensors"]["parameters"]["presence"], [True, False])
        self.assertEqual(representation(exported).tensors[0][1].presence, (True, False))


if __name__ == "__main__":
    unittest.main()
