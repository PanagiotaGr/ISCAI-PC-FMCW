import unittest


from iscai_stage4.predictor.stage4_input import (
    Stage4ActorInput,
)

from iscai_stage4.predictor.gmm_predictor import (
    GMMPredictor,
)



class TestGMMPredictor(
    unittest.TestCase
):


    def test_modes(self):

        actor = Stage4ActorInput(

            time_index=1,

            position=(
                0.0,
                0.0,
                0.0,
            ),

            velocity=(
                5.0,
                0.0,
                0.0,
            ),

            heading_rad=0.0,

            range_m=20.0,

            radial_velocity_mps=0.0,

            measurement_covariance=(

                (
                    0.1,
                    0.0,
                    0.0,
                ),

                (
                    0.0,
                    0.1,
                    0.0,
                ),

                (
                    0.0,
                    0.0,
                    0.1,
                ),
            ),

            length_m=4.5,
            width_m=1.8,
            height_m=1.5,

            actor_class="vehicle",
        )


        model = GMMPredictor(
            horizon=10
        )


        prediction = model.predict(
            actor
        )


        self.assertEqual(
            prediction.mode_count,
            5,
        )


        self.assertAlmostEqual(
            prediction.probabilities_sum(),
            1.0,
        )



if __name__ == "__main__":
    unittest.main()
