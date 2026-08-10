import unittest

from iscai_stage1.artifacts.scenario_hashing import (
    combined_causal_artifact_sha256,
    manifest_record_sha256,
)


class TestScenarioHashing(unittest.TestCase):

    def test_manifest_record_hash_deterministic(self):
        line = b'{"scenario_id":"abc"}\n'

        self.assertEqual(
            manifest_record_sha256(line),
            manifest_record_sha256(line),
        )

    def test_manifest_record_hash_changes(self):
        self.assertNotEqual(
            manifest_record_sha256(
                b'{"scenario_id":"abc"}\n'
            ),
            manifest_record_sha256(
                b'{"scenario_id":"xyz"}\n'
            ),
        )

    def test_combined_hash_deterministic(self):
        kwargs = dict(
            scenario_id="abc",
            anchor_index=10,
            actor_hashes=("a", "b"),
            map_hash="m",
        )

        self.assertEqual(
            combined_causal_artifact_sha256(
                **kwargs
            ),
            combined_causal_artifact_sha256(
                **kwargs
            ),
        )

    def test_combined_hash_changes_with_causal_input(self):
        a = combined_causal_artifact_sha256(
            scenario_id="abc",
            anchor_index=10,
            actor_hashes=("a", "b"),
            map_hash="m",
        )

        b = combined_causal_artifact_sha256(
            scenario_id="abc",
            anchor_index=10,
            actor_hashes=("a", "changed"),
            map_hash="m",
        )

        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
