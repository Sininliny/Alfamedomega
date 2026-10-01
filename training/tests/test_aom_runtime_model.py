import unittest
import tempfile
from pathlib import Path
import torch
from training.aom import Representation, Tensor, TensorSchema
from training.aom.runtime_model import INPUT_CATEGORIES, RuntimeTensorModel, train_runtime_sequences, export_runtime_model


class RuntimeModelTests(unittest.TestCase):
    def test_separate_recurrent_runtime_compiler_learns_tensor_parameters(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        widths = {name: 0 for name in INPUT_CATEGORIES}
        widths["midi"] = 1
        schemas = {"parameters": TensorSchema(("parameter",), (1,))}
        model = RuntimeTensorModel(widths, 8, schemas)
        def event(value):
            inputs = {name: [] for name in INPUT_CATEGORIES}
            inputs["midi"] = [value]
            target = Representation("test", (("parameters", Tensor(("parameter",), (1,), (value,))),))
            return inputs, target
        sequences = [[event(.2), event(.8), event(.2)], [event(.8), event(.2)]]
        result = train_runtime_sequences(model, sequences, ontology="test", steps=120,
                                        learning_rate=.03, mask_weight=.1, available_ram_bytes=10000000)
        self.assertLess(result["losses"][-1], result["losses"][0] / 20)
        prediction, state = model.infer(event(.2)[0], torch.zeros(1, 8), mask_threshold=.5)
        self.assertAlmostEqual(prediction["parameters"]["values"][0], .2, delta=.04)
        self.assertEqual(state.shape, (1, 8))
        with self.assertRaises(ValueError):
            model.infer({"midi": [.2]}, state, mask_threshold=.5)
        with tempfile.TemporaryDirectory() as folder:
            document = export_runtime_model(Path(folder) / "runtime.json", model,
                ontology="test", period_samples=7, phase_samples=2, mask_threshold=.5,
                available_ram_bytes=1000000, bounds={"parameters": [(0., 1.)]},
                golden_inputs=[event(.2)[0], event(.8)[0]])
            self.assertEqual(document["clock"], {"period_samples": 7, "phase_samples": 2})
            self.assertEqual(len(document["weights"]["cell.weight_ih"]), 24)
            self.assertEqual(len(document["golden"]), 2)
            self.assertAlmostEqual(document["golden"][0]["values"][0], .2, delta=.04)


if __name__ == "__main__":
    unittest.main()
