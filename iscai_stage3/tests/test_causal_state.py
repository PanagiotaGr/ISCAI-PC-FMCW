import unittest


from iscai_stage3.baselines.causal_state import (
    estimate_causal_motion_state,
)



class TestCausalState(unittest.TestCase):


    def test_velocity_estimation(self):

        state = estimate_causal_motion_state(
            positions=(
                (0.0,0.0,0.0),
                (1.0,0.0,0.0),
                (2.0,0.0,0.0),
            ),

            timestamps=(
                0.0,
                1.0,
                2.0,
            ),
        )


        self.assertAlmostEqual(
            state.velocity[0],
            1.0,
        )


        self.assertAlmostEqual(
            state.heading_rad,
            0.0,
        )



    def test_acceleration(self):

        state = estimate_causal_motion_state(
            positions=(
                (0.0,0.0,0.0),
                (1.0,0.0,0.0),
                (3.0,0.0,0.0),
            ),

            timestamps=(
                0.0,
                1.0,
                2.0,
            ),
        )


        self.assertGreater(
            state.acceleration[0],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
