"""Reader boundaries keep construction answers out of observed model inputs."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.amo.corpus.reader import batches


class CorpusReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        shard = self.root / "shards" / "fixture"
        shard.mkdir(parents=True)
        (self.root / "shards.json").write_text(
            json.dumps([{"shard": "shards/fixture"}])
        )
        np.savez_compressed(
            shard / "prepared.npz",
            id=np.array(
                [
                    "ordinary-train",
                    "held-train",
                    "tail-train",
                    "held-validation",
                    "held-test",
                ]
            ),
            canonical_id=np.array(["a", "b", "c", "d", "e"]),
            type_id=np.array([0, 6, 1, 21, 28]),
            split=np.array([0, 0, 0, 1, 2]),
            X_audio=np.zeros((5, 2244), dtype=np.float32),
            conditions=np.zeros((5, 10), dtype=np.float32),
            fx_input_mask=np.zeros(5, dtype=bool),
            canonical_parameters=np.zeros((5, 2), dtype=np.float32),
            canonical_parameters_mask=np.tile([True, False], (5, 1)),
            instance_parameters=np.ones((5, 2), dtype=np.float32),
        )

    def test_observations_exclude_answers_and_tail_is_retained(self):
        result = list(batches(self.root, batch_size=2))
        self.assertEqual([len(b["metadata"]["id"]) for b in result], [2, 1])
        for batch in result:
            self.assertEqual(
                set(batch["inputs"]), {"X_audio", "conditions", "fx_input_mask"}
            )
            self.assertNotIn("canonical_id", batch["targets"])
            self.assertIn("instance_parameters", batch["targets"])
            # A present numerical zero and an absent padded zero stay distinct.
            np.testing.assert_array_equal(
                batch["targets"]["canonical_parameters_mask"][0], [True, False]
            )

    def test_topology_holdout_filters_every_original_split(self):
        train = list(batches(self.root, protocol="unseen_topology"))
        self.assertEqual(
            np.concatenate([b["metadata"]["id"] for b in train]).tolist(),
            ["ordinary-train", "tail-train"],
        )
        test = list(
            batches(self.root, split="topology_test", protocol="unseen_topology")
        )
        self.assertEqual(
            np.concatenate([b["metadata"]["id"] for b in test]).tolist(),
            ["held-train", "held-validation", "held-test"],
        )

    def test_invalid_protocol_and_batch_size_are_rejected(self):
        for arguments in (
            {"protocol": "unknown"},
            {"split": "topology_test"},
            {"batch_size": 0},
            {"batch_size": -1},
        ):
            with self.assertRaises(ValueError):
                next(batches(self.root, **arguments))
