import unittest

from iscai_stage3.baselines.mode_prediction import (
    predict_ca_measurement,
    predict_ctrv_measurement,
    predict_cv_measurement,
)


class TestModePrediction(unittest.TestCase):

    def test_cv_advances_position(self):

        z = predict_cv_measurement(
            position=(
                10.0,
                0.0,
                0.0,
            ),
            velocity=(
                1.0,
                0.0,
                0.0,
            ),
            dt=1.0,
        )

        self.assertAlmostEqual(
            z[0],
            11.0,
        )

        self.assertAlmostEqual(
            z[1],
            1.0,
        )


    def test_ca_advances_more_than_cv(self):

        cv = predict_cv_measurement(
            position=(
                10.0,
                0.0,
                0.0,
            ),
            velocity=(
                1.0,
                0.0,
                0.0,
            ),
            dt=1.0,
        )

        ca = predict_ca_measurement(
            position=(
                10.0,
                0.0,
                0.0,
            ),
            velocity=(
                1.0,
                0.0,
                0.0,
            ),
            acceleration=(
                1.0,
                0.0,
                0.0,
            ),
            dt=1.0,
        )

        self.assertGreater(
            ca[0],
            cv[0],
        )


    def test_ctrv_zero_turn_matches_straight_motion(self):

        z = predict_ctrv_measurement(
            position=(
                10.0,
                0.0,
                0.0,
            ),
            speed_mps=1.0,
            heading_rad=0.0,
            yaw_rate_radps=0.0,
            vertical_velocity_mps=0.0,
            dt=1.0,
        )

        self.assertAlmostEqual(
            z[0],
            11.0,
        )

        self.assertAlmostEqual(
            z[2],
            0.0,
        )


    def test_ctrv_turn_changes_azimuth(self):

        z = predict_ctrv_measurement(
            position=(
                10.0,
                0.0,
                0.0,
            ),
            speed_mps=5.0,
            heading_rad=0.0,
            yaw_rate_radps=0.2,
            vertical_velocity_mps=0.0,
            dt=1.0,
        )

        self.assertGreater(
            z[2],
            0.0,
        )


    def test_invalid_dt_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            predict_cv_measurement(
                position=(
                    10.0,
                    0.0,
                    0.0,
                ),
                velocity=(
                    1.0,
                    0.0,
                    0.0,
                ),
                dt=0.0,
            )


if __name__ == "__main__":
    unittest.main()
