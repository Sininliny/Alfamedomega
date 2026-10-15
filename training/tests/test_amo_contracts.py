"""Run with: python -m unittest discover -s training/tests -p 'test_amo*.py'."""

from dataclasses import replace
import unittest

from training.amo import (
    CanonicalGroup, ModelBudget, ModelCategory, NormalizationRecord,
    Representation, Tensor, TensorSchema, WavExample, dual_targets, pad_tensor,
    validate_model_budgets,
)


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.canonical = Representation("amo-test-ontology", (
            ("parameters", Tensor(("primitive", "parameter", "time"), (1, 2, 1), (0.0, 0.5))),
            ("routing", Tensor(("source", "destination"), (1, 2), (0.1, 0.2))),
        ))
        self.instance = Representation("amo-test-ontology", (
            ("parameters", Tensor(("primitive", "parameter", "time"), (1, 2, 1), (0.2, 0.8))),
            ("routing", Tensor(("source", "destination"), (1, 2), (0.4, 0.6))),
        ))
        self.schemas = {
            "parameters": TensorSchema(("primitive", "parameter", "time"), (2, 3, 2)),
            "routing": TensorSchema(("source", "destination"), (2, 3)),
        }
        self.record = NormalizationRecord("explicit-test-policy", frozenset({
            "pitch", "loudness", "recording_condition"}), frozenset({
            "timbre", "articulation", "temporal_behavior"}))
        self.example = WavExample("render-1", "canonical-1", self.canonical,
            self.instance, Tensor(("sample", "channel"), (2, 1), (0.1, 0.2)),
            48000, self.record, "perturbation-1")

    def test_both_targets_preserve_distinct_original_and_perturbed_values(self):
        targets = dual_targets(self.example, "amo-test-ontology", self.schemas)
        primary = targets["primary_canonical"]["parameters"]
        secondary = targets["secondary_rendered_instance"]["parameters"]
        self.assertEqual(primary.tensor.values[:4], (0.0, 0.0, 0.5, 0.0))
        self.assertEqual(secondary.tensor.values[:4], (0.2, 0.0, 0.8, 0.0))
        self.assertEqual(primary.mask, secondary.mask)
        self.assertEqual(sum(primary.mask), 2)
        self.assertTrue(primary.mask[0])  # Actual zero is not padding.
        self.assertFalse(primary.mask[1])

    def test_padding_retains_multidimensional_row_major_coordinates(self):
        source = Tensor(("source", "destination"), (2, 2), (1, 2, 3, 4))
        padded = pad_tensor(source, TensorSchema(source.axes, (3, 3)))
        self.assertEqual(padded.tensor.values, (1, 2, 0, 3, 4, 0, 0, 0, 0))
        self.assertEqual(padded.mask, (True, True, False, True, True, False, False, False, False))

    def test_tiers_use_same_schema_with_distinct_active_extents(self):
        schema = TensorSchema(("primitive",), (4,))
        small = pad_tensor(Tensor(("primitive",), (1,), (0.7,)), schema)
        large = pad_tensor(Tensor(("primitive",), (3,), (0.7, 0.2, 0.1)), schema)
        self.assertEqual(small.tensor.shape, large.tensor.shape)
        self.assertEqual(sum(small.mask), 1)
        self.assertEqual(sum(large.mask), 3)

    def test_padding_rejects_truncation_and_axis_reinterpretation(self):
        source = Tensor(("primitive",), (2,), (1, 2))
        for schema in (TensorSchema(("primitive",), (1,)), TensorSchema(("method",), (2,))):
            with self.assertRaises(ValueError):
                pad_tensor(source, schema)

    def test_variants_share_canonical_but_keep_instance_targets(self):
        second = replace(self.example, example_id="render-2", perturbation_id="perturbation-2",
                         rendered_instance=self.canonical, sample_rate=44100)
        group = CanonicalGroup((self.example, second))
        self.assertEqual(group.examples[0].canonical, group.examples[1].canonical)
        self.assertNotEqual(group.examples[0].rendered_instance, group.examples[1].rendered_instance)
        with self.assertRaises(ValueError):
            CanonicalGroup((self.example, replace(second, canonical=self.instance)))

    def test_missing_normalization_or_temporal_preservation_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.record, removed=frozenset({"pitch", "loudness"}))
        with self.assertRaises(ValueError):
            replace(self.record, preserved=frozenset({"timbre", "articulation"}))
        with self.assertRaises(ValueError):
            replace(self.record, preserved=self.record.preserved | {"pitch"})

    def test_representation_version_and_components_must_match(self):
        with self.assertRaises(ValueError):
            dual_targets(self.example, "other-ontology", self.schemas)
        with self.assertRaises(ValueError):
            dual_targets(self.example, "amo-test-ontology", {"parameters": self.schemas["parameters"]})
        with self.assertRaises(ValueError):
            replace(self.example, rendered_instance=replace(self.instance, ontology="other"))

    def test_budgets_are_independent_and_excess_memory_is_rejected(self):
        budgets = {
            ModelCategory.RUNTIME: ModelBudget(10, 20, 0, 0),
            ModelCategory.WAV_PRESET_LEARNING: ModelBudget(80, 100, 40, 50),
            ModelCategory.USER_PREFERENCE: ModelBudget(5, 15, 0, 0),
        }
        validate_model_budgets(budgets)
        with self.assertRaises(ValueError):
            validate_model_budgets({ModelCategory.RUNTIME: budgets[ModelCategory.RUNTIME]})
        with self.assertRaises(ValueError):
            ModelBudget(101, 100, 0, 0)
        with self.assertRaises(ValueError):
            ModelBudget(10, 100, 51, 50)

    def test_invalid_tensor_and_audio_data_are_rejected(self):
        for values in ((float("nan"),), (float("inf"),), ()):
            with self.assertRaises(ValueError):
                Tensor(("sample",), (1,), values)
        with self.assertRaises(ValueError):
            replace(self.example, sample_rate=0)
        with self.assertRaises(ValueError):
            replace(self.example, normalized_audio=Tensor(("sample", "channel"), (0, 1), ()))


if __name__ == "__main__":
    unittest.main()
