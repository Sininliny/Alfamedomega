import copy
import unittest
from training.amo.tiers import TierPolicy, CannotFitTier, fit_importance, optimize_candidates


class TierSearchTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"ontology": "test", "tensors": {
            "parameters": {"axes": ["parameter"], "shape": [2], "values": [440., 1.]},
            "activation": {"axes": ["primitive"], "shape": [1], "values": [1.]}},
            "parameter_locks": [1, 0]}
        self.policy = TierPolicy({"active_primitives": 1}, {"pitch": 0, "timing": 0, "timbre": 0.2}, {"timbre": 1})

    def candidate(self, amplitude):
        candidate = copy.deepcopy(self.reference)
        candidate["tensors"]["parameters"]["values"][1] = amplitude
        return candidate

    def evaluate(self, original, candidate):
        return {"pitch": 0, "timing": 0, "timbre": abs(1 - candidate["tensors"]["parameters"]["values"][1])}

    def test_measured_search_preserves_locks_and_selects_smallest_actual_error(self):
        candidates = [self.candidate(.5), self.candidate(.9), self.candidate(.85)]
        candidates[2]["tensors"]["parameters"]["values"][0] = 441
        seen = []
        result = optimize_candidates(self.reference, candidates, self.policy,
                    validate_runtime=lambda doc: seen.append(doc["budget"]), evaluate=self.evaluate)
        self.assertEqual(result["candidate"], 1)
        self.assertEqual(len(result["rejected"]), 2)
        self.assertEqual(result["document"]["parameter_locks"], [1, 0])
        self.assertTrue(all(budget == self.policy.capacity for budget in seen))

    def test_unfit_configuration_is_rejected_instead_of_blind_pruning(self):
        with self.assertRaises(CannotFitTier):
            optimize_candidates(self.reference, [self.candidate(.5)], self.policy,
                                validate_runtime=lambda doc: None, evaluate=self.evaluate)
        with self.assertRaises(CannotFitTier):
            optimize_candidates(self.reference, [self.candidate(1)], self.policy,
                                validate_runtime=lambda doc: None,
                                evaluate=lambda a, b: {"pitch": 0.1, "timing": 0, "timbre": 0})

    def test_importance_is_learned_from_actual_observation_losses(self):
        model = fit_importance(self.reference, [(self.candidate(.5), .25), (self.candidate(.8), .1)], ridge=.001)
        self.assertEqual(model["observations"], 2)
        self.assertAlmostEqual(model["weights"][1], .5, delta=.01)
        self.assertEqual(model["weights"][0], 0)


if __name__ == "__main__":
    unittest.main()
