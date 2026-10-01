import unittest
import tempfile
from pathlib import Path
import torch
from training.aom.local_learning import ContextImportanceModel, LocalPreferenceModel, save_local_model, load_local_model


class LocalLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)

    def test_importance_changes_with_context_using_measured_labels(self):
        widths = {name: 1 for name in ("state", "preference", "context", "relationship")}
        model = ContextImportanceModel(widths, 8, 2)
        def features(context):
            return {"state": [1.], "preference": [.5], "context": [context], "relationship": [1.]}
        report = model.fit([(features(0), [1., .1]), (features(1), [.1, 1.])],
                           steps=180, learning_rate=.03, available_ram_bytes=1000000)
        self.assertLess(report["losses"][-1], report["losses"][0] / 100)
        self.assertGreater(model.predict_importance(features(0))[0], model.predict_importance(features(0))[1])
        self.assertLess(model.predict_importance(features(1))[0], model.predict_importance(features(1))[1])
        with self.assertRaises(ValueError):
            model.fit([(features(0), [1., .1])], steps=1, learning_rate=.03, available_ram_bytes=1)

    def test_local_pairwise_feedback_learns_and_can_reverse_taste(self):
        model = LocalPreferenceModel({"aom_parameters": 2, "context": 1}, 8)
        warm = {"aom_parameters": [1., 0.], "context": [0.]}
        bright = {"aom_parameters": [0., 1.], "context": [0.]}
        result = model.fit_pairs([(warm, bright, True)], steps=100, learning_rate=.03, available_ram_bytes=1000000)
        self.assertLess(result["losses"][-1], result["losses"][0] / 10)
        self.assertGreater(model.score(warm), model.score(bright))
        model.fit_pairs([(warm, bright, False)], steps=200, learning_rate=.03, available_ram_bytes=1000000)
        self.assertLess(model.score(warm), model.score(bright))
        self.assertEqual(model.preference_tensor()["shape"][0], sum(p.numel() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "local.json"
            save_local_model(path, model)
            restored = load_local_model(path, available_ram_bytes=1000000)
            self.assertEqual(model.score(warm), restored.score(warm))


if __name__ == "__main__":
    unittest.main()
