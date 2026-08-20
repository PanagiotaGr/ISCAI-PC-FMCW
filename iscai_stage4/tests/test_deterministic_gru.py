import unittest


from iscai_stage4.predictor.stage4_input import (
    Stage4ActorInput,
)

from iscai_stage4.predictor.deterministic_gru import (
    DeterministicGRUPredictor,
)



class TestDeterministicGRU(
    unittest.TestCase
):


    def test_prediction_horizon(
        self,
    ):

        actor = Stage4ActorInput(

            time_index=5,

            position=(
                0.0,
                0.0,
                0.0,
            ),

            velocity=(
                10.0,
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


        model = DeterministicGRUPredictor(
            horizon=10,
            dt=0.1,
        )


        prediction = model.predict(
            actor
        )


        self.assertEqual(
            prediction.horizon,
            10,
        )


        self.assertAlmostEqual(
            prediction.positions[-1][0],
            10.0,
        )



if __name__ == "__main__":
    unittest.main()
