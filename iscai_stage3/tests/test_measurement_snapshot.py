import math
import unittest

from iscai_stage3.observations import (
    covariance_matrix_4x4,
    snapshot_from_detection,
    wrap_angle_rad,
)

from stage2_test_factory import (
    make_detection,
)


class TestMeasurementSnapshot(
    unittest.TestCase
):

    def test_angle_wrap(self):
        self.assertAlmostEqual(
            wrap_angle_rad(
                math.pi + 0.1
            ),
            -math.pi + 0.1,
        )

    def test_stage2_snapshot(self):
        detection = make_detection(
            key="abc",
            range_m=20.0,
            vr_mps=-2.0,
            az_rad=0.2,
        )

        snapshot = (
            snapshot_from_detection(
                detection
            )
        )

        self.assertEqual(
            snapshot.detection_key,
            "abc",
        )

        self.assertAlmostEqual(
            snapshot.range_m,
            20.0,
        )

        self.assertEqual(
            len(
                snapshot.covariance_4x4
            ),
            4,
        )

    def test_bad_covariance_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            covariance_matrix_4x4(
                (
                    (1.0, 0.0),
                    (0.0, 1.0),
                )
            )


if __name__ == "__main__":
    unittest.main()
