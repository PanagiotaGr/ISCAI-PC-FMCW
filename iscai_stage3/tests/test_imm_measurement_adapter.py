import unittest


from iscai_stage3.baselines.imm_measurement_adapter import (
    cartesian_to_measurement,
    measurement_residual,
)



class TestIMMMeasurementAdapter(
    unittest.TestCase
):


    def test_forward_axis(self):

        z = cartesian_to_measurement(
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
        )

        self.assertAlmostEqual(
            z[0],
            10.0,
        )

        self.assertAlmostEqual(
            z[1],
            1.0,
        )

        self.assertAlmostEqual(
            z[2],
            0.0,
        )



    def test_residual(self):

        r = measurement_residual(
            predicted=(
                10.0,
                1.0,
                0.0,
                0.0,
            ),
            measured=(
                11.0,
                2.0,
                0.1,
                0.1,
            ),
        )

        self.assertAlmostEqual(
            r[0],
            1.0,
        )



    def test_zero_range_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            cartesian_to_measurement(
                position=(
                    0.0,
                    0.0,
                    0.0,
                ),
                velocity=(
                    0.0,
                    0.0,
                    0.0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
