import unittest


from iscai_stage4.controller.prediction_adapter import (
    prediction_to_controller,
)



class TestPredictionAdapter(
    unittest.TestCase
):


    def test_conversion(self):

        item = {

            "actor_id":
                0,

            "model_name":
                "GAUSSIAN_GRU",

            "trajectory":

                [
                    [
                        1.0,
                        2.0,
                        0.0,
                    ],
                    [
                        2.0,
                        3.0,
                        0.0,
                    ],
                ],

            "uncertainty":
                0.2,

        }


        result = prediction_to_controller(
            item
        )


        self.assertEqual(
            result.model_name,
            "GAUSSIAN_GRU",
        )


        self.assertAlmostEqual(
            result.confidence,
            0.8,
        )


        self.assertEqual(
            len(result.trajectory),
            2,
        )


if __name__ == "__main__":

    unittest.main()
