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


class CorpusTrainingDataTests(unittest.TestCase):
    """NumPy side of the corpus trainer; the model itself needs PyTorch."""

    def test_load_split_concatenates_all_shards(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        shards = []
        for index, type_id in enumerate((0, 3)):
            shard = root / "shards" / f"s{index}"
            shard.mkdir(parents=True)
            shards.append({"shard": f"shards/s{index}"})
            np.savez_compressed(
                shard / "prepared.npz",
                id=np.array([f"{index}-a", f"{index}-b"]),
                canonical_id=np.array(["a", "b"]),
                type_id=np.array([type_id, type_id]),
                split=np.array([0, 1]),
                X_audio=np.zeros((2, 2244), dtype=np.float32),
                conditions=np.zeros((2, 10), dtype=np.float32),
                fx_input_mask=np.zeros(2, dtype=bool),
            )
        (root / "shards.json").write_text(json.dumps(shards))
        from training.amo.corpus.reader import load_split

        train = load_split(root, "train")
        self.assertEqual(train["id"].tolist(), ["0-a", "1-a"])
        self.assertEqual(train["type_id"].tolist(), [0, 3])
        with self.assertRaises(ValueError):
            load_split(root, "test")

    def test_target_statistics_round_trip_and_constant_coordinates(self):
        from training.amo.corpus.train import TargetStats

        type_id = np.array([0, 0, 0, 1, 1])
        values = np.array(
            [[200.0, 0.5, 0.0], [2000.0, 0.5, 0.0], [6000.0, 0.5, 0.0],
             [-0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], np.float32)
        mask = np.array([[1, 1, 0]] * 3 + [[1, 0, 0]] * 2, bool)
        stats = TargetStats.fit(type_id, [values], [mask], types=2)
        self.assertTrue(stats.log[0, 0])  # positive and spans > 8x: log space
        self.assertFalse(stats.log[1, 0])  # signed
        self.assertFalse(stats.learnable[0, 1])  # constant inside family 0
        self.assertFalse(stats.learnable[0, 2])  # absent padding
        z = stats.encode(type_id, values, mask)
        self.assertTrue(np.all(z[~mask] == 0))
        decoded = stats.decode(type_id, z)
        np.testing.assert_allclose(decoded[mask], values[mask], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(stats.weight(type_id, mask)[:, 1], 0)
