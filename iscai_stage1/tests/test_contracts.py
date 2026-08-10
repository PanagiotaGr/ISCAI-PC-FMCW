import unittest

from iscai_stage1.contracts.stage1a import (
    ARTIFACT_SEMANTICS,
    ASSOCIATION_MODE,
    HEADLAMP_SURROGATE_MODE,
    LIDAR_ACTOR_ASSIGNMENT_MODE,
    RECEIVER_GEOMETRY_MODE,
    SENSOR_REALISTIC,
    HeadlampSurrogateConfig,
    ReceiverGeometryConfig,
    Stage1ArtifactSemantics,
    ZERO_MAT3,
    ZERO_VEC3,
)


class TestFrozenContracts(unittest.TestCase):
    def test_artifact_semantics(self) -> None:
        semantics = Stage1ArtifactSemantics()

        self.assertEqual(
            semantics.artifact_semantics,
            "causal_womd_annotation_upstream",
        )
        self.assertFalse(semantics.sensor_realistic)
        self.assertEqual(ASSOCIATION_MODE, "oracle_womd_track_id")
        self.assertEqual(
            LIDAR_ACTOR_ASSIGNMENT_MODE,
            "oracle_causal_box",
        )
        self.assertEqual(ARTIFACT_SEMANTICS, semantics.artifact_semantics)
        self.assertFalse(SENSOR_REALISTIC)

    def test_headlamp_baseline(self) -> None:
        config = HeadlampSurrogateConfig()

        self.assertEqual(config.mode, HEADLAMP_SURROGATE_MODE)
        self.assertEqual(
            config.translation_in_sdc_m(4.8),
            (2.4, 0.0, 0.0),
        )
        self.assertEqual(config.roll_rad, 0.0)
        self.assertEqual(config.pitch_rad, 0.0)
        self.assertEqual(config.yaw_rad, 0.0)

    def test_receiver_baseline(self) -> None:
        config = ReceiverGeometryConfig()

        self.assertEqual(config.mode, RECEIVER_GEOMETRY_MODE)
        self.assertEqual(config.offset_mean_body_m, ZERO_VEC3)
        self.assertEqual(config.offset_covariance_body_m2, ZERO_MAT3)


if __name__ == "__main__":
    unittest.main()