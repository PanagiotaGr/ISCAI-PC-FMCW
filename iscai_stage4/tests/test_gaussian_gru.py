import unittest


from iscai_stage4.predictor.stage4_input import (
    Stage4ActorInput,
)

from iscai_stage4.predictor.gaussian_gru import (
    GaussianGRUPredictor,
)



class TestGaussianGRU(
    unittest.TestCase
):


    def actor(self):

        return Stage4ActorInput(

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

            range_m=10.0,

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

            length_m=4.0,
            width_m=2.0,
            height_m=1.5,

            actor_class="vehicle",
        )


    def test_gaussian_output(self):

        model = GaussianGRUPredictor(
            horizon=10
        )


        prediction = model.predict(
            self.actor()
        )


        self.assertEqual(
            prediction.horizon,
            10,
        )


        for cov in prediction.covariance:

            self.assertGreater(
                cov[0][0],
                0,
            )


if __name__ == "__main__":
    unittest.main()
