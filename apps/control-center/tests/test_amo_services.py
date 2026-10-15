import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from alfacc.amo import validate_tensor_document, infer_tensor_preset
from alfacc.cli import main, build_parser


class AmoServiceTests(unittest.TestCase):
    def test_control_center_requires_tensor_document(self):
        with self.assertRaises(ValueError):
            validate_tensor_document({"method": "wavetable", "parameters": [0] * 32})
        validate_tensor_document({"schema": "amo.tensor/1", "ontology": "amo.math/1",
            "tensors": {name: {} for name in ("activation", "parameters", "routing", "modulation")}})

    def test_inference_requires_explicit_valid_presence_threshold(self):
        with self.assertRaises(ValueError):
            infer_tensor_preset(".", "missing.pt", "missing.wav", "output.json", mask_threshold=0)

    def test_tensor_cli_dispatch_skips_legacy_model_store_initialization(self):
        with patch("alfacc.cli.infer_tensor_preset", return_value={"ontology": "amo.math/1"}) as infer:
            with patch("alfacc.cli.Context", side_effect=AssertionError("legacy store must not open")):
                with redirect_stdout(StringIO()):
                    code = main(["amo", "--repository", ".", "infer", "model.pt", "audio.wav", "output.json", "--mask-threshold", ".5"])
        self.assertEqual(code, 0)
        infer.assert_called_once()

    def test_legacy_cli_and_runtime_compiler_export_remain_available(self):
        parser = build_parser()
        legacy = parser.parse_args(["train", "--smoke"])
        self.assertEqual(legacy.tier, "lite")
        runtime = parser.parse_args(["amo", "--repository", ".", "train-runtime", "events.json", "runtime.json"])
        self.assertEqual(runtime.action, "train-runtime")
        self.assertIn("legacy method-vector", parser.format_help())

    def test_tensor_service_failures_return_cli_error_code(self):
        with patch("alfacc.cli.train_tensor_model", side_effect=RuntimeError("Insufficient memory")):
            with redirect_stderr(StringIO()) as error:
                code = main(["amo", "--repository", ".", "train", "manifest.json", "model.pt"])
        self.assertEqual(code, 1)
        self.assertIn("Insufficient memory", error.getvalue())


if __name__ == "__main__":
    unittest.main()
