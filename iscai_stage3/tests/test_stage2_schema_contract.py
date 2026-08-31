import unittest

from dataclasses import fields

from iscai_stage2.observations.detection_set import (
    UnlabeledDetection,
    UnlabeledDetectionFrame,
)

from iscai_stage3.contracts import (
    forbidden_algorithm_fields,
    resolve_stage2_algorithm_schema,
)


class TestStage2SchemaContract(
    unittest.TestCase
):

    def test_required_detection_fields_resolve(
        self,
    ):
        schema = (
            resolve_stage2_algorithm_schema()
        )

        values = {
            schema.detection_key_field,
            schema.range_field,
            schema.radial_velocity_field,
            schema.azimuth_field,
            schema.elevation_field,
            schema.covariance_field,
        }

        self.assertEqual(
            len(values),
            6,
        )

    def test_required_frame_fields_resolve(
        self,
    ):
        schema = (
            resolve_stage2_algorithm_schema()
        )

        self.assertNotEqual(
            schema.frame_timestamp_field,
            schema.frame_detections_field,
        )

    def test_no_truth_or_identity_fields(
        self,
    ):
        self.assertEqual(
            forbidden_algorithm_fields(),
            (),
        )

    def test_algorithm_types_are_dataclasses(
        self,
    ):
        self.assertGreater(
            len(fields(UnlabeledDetection)),
            0,
        )

        self.assertGreater(
            len(
                fields(
                    UnlabeledDetectionFrame
                )
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
