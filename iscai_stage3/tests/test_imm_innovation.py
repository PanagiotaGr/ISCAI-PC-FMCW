import unittest


from iscai_stage3.baselines.imm_innovation import (
    measurement_vector,
    innovation_from_prediction,
    build_innovation_record,
)



class TestIMMInnovation(unittest.TestCase):


    def test_measurement_vector(self):

        z = measurement_vector(
            range_m=10.0,
            radial_velocity_mps=1.0,
            azimuth_rad=0.1,
            elevation_rad=0.2,
        )

        self.assertEqual(
            len(z),
            4,
        )


    def test_innovation_sign(self):

        innovation = (
            innovation_from_prediction(
                predicted_measurement=(
                    10.0,
                    1.0,
                    0.1,
                    0.2,
                ),
                measured=(
                    11.0,
                    2.0,
                    0.2,
                    0.3,
                ),
                mode_name="CV",
            )
        )

        self.assertAlmostEqual(
            innovation[0],
            1.0,
        )

        self.assertAlmostEqual(
            innovation[1],
            1.0,
        )

        self.assertAlmostEqual(
            innovation[2],
            0.1,
        )

        self.assertAlmostEqual(
            innovation[3],
            0.1,
        )


    def test_record_creation(self):

        record = build_innovation_record(
            mode_name="CTRV",
            predicted_measurement=(
                1.0,
                2.0,
                3.0,
                4.0,
            ),
            measured=(
                1.1,
                2.1,
                3.1,
                4.1,
            ),
            covariance_trace=2.0,
            log_likelihood=-5.0,
        )

        self.assertEqual(
            record.mode_name,
            "CTRV",
        )

        self.assertAlmostEqual(
            record.innovation[0],
            0.1,
        )


    def test_negative_covariance_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            build_innovation_record(
                mode_name="CV",
                predicted_measurement=(
                    1,1,1,1
                ),
                measured=(
                    1,1,1,1
                ),
                covariance_trace=-1.0,
                log_likelihood=-1.0,
            )


    def test_nan_likelihood_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            build_innovation_record(
                mode_name="CV",
                predicted_measurement=(
                    1,1,1,1
                ),
                measured=(
                    1,1,1,1
                ),
                covariance_trace=1.0,
                log_likelihood=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
