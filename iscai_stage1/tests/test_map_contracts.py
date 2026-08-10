import unittest

from iscai_stage1.maps.contracts import (
    CausalDynamicMapFrame,
    CausalMapArtifact,
    StaticMapFeature,
)
from iscai_stage1.maps.hashing import (
    causal_map_sha256,
)


class TestMapContracts(unittest.TestCase):

    def artifact(self):
        return CausalMapArtifact(
            scenario_id="scenario",
            anchor_index=1,
            static_features=(
                StaticMapFeature(
                    feature_id=7,
                    kind="lane",
                    points_W_m=(
                        (1.0, 2.0, 3.0),
                        (2.0, 2.0, 3.0),
                    ),
                    points_H0_m=(
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                    ),
                    type_name="TYPE_SURFACE_STREET",
                    speed_limit_mph=35.0,
                    interpolating=False,
                    entry_lane_ids=(5,),
                    exit_lane_ids=(9,),
                ),
            ),
            dynamic_frames=(
                CausalDynamicMapFrame(
                    time_index=0,
                    timestamp_s=0.0,
                    lane_states=(),
                ),
                CausalDynamicMapFrame(
                    time_index=1,
                    timestamp_s=0.1,
                    lane_states=(),
                ),
            ),
        )

    def test_semantics(self):
        artifact = self.artifact()

        self.assertEqual(
            artifact.artifact_semantics,
            "causal_womd_annotation_upstream",
        )
        self.assertFalse(
            artifact.sensor_realistic
        )

    def test_dynamic_prefix_length(self):
        artifact = self.artifact()

        self.assertEqual(
            len(artifact.dynamic_frames),
            artifact.anchor_index + 1,
        )

    def test_hash_deterministic(self):
        a = self.artifact()
        b = self.artifact()

        self.assertEqual(
            causal_map_sha256(a),
            causal_map_sha256(b),
        )

    def test_hash_changes_with_causal_map(self):
        a = self.artifact()

        b = CausalMapArtifact(
            scenario_id=a.scenario_id,
            anchor_index=a.anchor_index,
            static_features=(),
            dynamic_frames=a.dynamic_frames,
        )

        self.assertNotEqual(
            causal_map_sha256(a),
            causal_map_sha256(b),
        )


if __name__ == "__main__":
    unittest.main()
