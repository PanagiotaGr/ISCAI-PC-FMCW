import unittest


from iscai_stage4.predictor.stage4_input import (
    Stage4ActorInput,
)


class TestStage4Input(unittest.TestCase):


    def test_feature_vector(self):

        x = Stage4ActorInput(

            time_index=10,

            position=(
                1.0,
                2.0,
                0.0,
            ),

            velocity=(
                5.0,
                0.0,
                0.0,
            ),

            heading_rad=0.2,

            range_m=20.0,

            radial_velocity_mps=-5.0,

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
                    0.2,
                ),
            ),

            length_m=4.5,
            width_m=1.8,
            height_m=1.6,

            actor_class="vehicle",
        )


        self.assertEqual(
            len(
                x.feature_vector()
            ),
            13,
        )


    def test_negative_range(self):

        with self.assertRaises(
            ValueError
        ):

            Stage4ActorInput(

                time_index=0,

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

                heading_rad=0.0,

                range_m=-1.0,

                radial_velocity_mps=0.0,

                measurement_covariance=(

                    (
                        1.0,
                        0.0,
                        0.0,
                    ),

                    (
                        0.0,
                        1.0,
                        0.0,
                    ),

                    (
                        0.0,
                        0.0,
                        1.0,
                    ),
                ),

                length_m=1.0,
                width_m=1.0,
                height_m=1.0,

                actor_class="vehicle",
            )


if __name__ == "__main__":
    unittest.main()
