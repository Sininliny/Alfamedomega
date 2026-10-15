import unittest

from training.amo.schema import canonical_id, same_id, schema_is


class SchemaRenameCompatibility(unittest.TestCase):
    def test_pre_rename_ids_match_current_ids(self):
        self.assertTrue(schema_is("amo.training-checkpoint/1", "amo.training-checkpoint/1"))
        self.assertTrue(schema_is("aom.training-checkpoint/1", "amo.training-checkpoint/1"))  # rename-compat
        self.assertFalse(schema_is("aom.training-checkpoint/2", "amo.training-checkpoint/1"))  # rename-compat
        self.assertTrue(same_id("aom.math/1", "amo.math/1"))  # rename-compat
        self.assertEqual(canonical_id(None), None)


if __name__ == "__main__":
    unittest.main()
