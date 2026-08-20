import unittest


from iscai_stage3.pipeline.smoke import (
    evaluate_target,
    select_target,
)



IDENTITY3 = (
    (0.01,0.0,0.0),
    (0.0,0.001,0.0),
    (0.0,0.0,0.001),
)



class TestStage3Pipeline(unittest.TestCase):


    def test_end_to_end_target(self):

        target = evaluate_target(
            target_id=1,
            range_m=20.0,
            azimuth_rad=0.0,
            elevation_rad=0.0,
            measurement_covariance=IDENTITY3,
            priority=2.0,
        )


        self.assertEqual(
            target.target_id,
            1,
        )


        self.assertGreater(
            target.beam_confidence,
            0.0,
        )


    def test_selection(self):

        a = evaluate_target(
            target_id=1,
            range_m=20.0,
            azimuth_rad=0.5,
            elevation_rad=0.0,
            measurement_covariance=IDENTITY3,
            priority=1.0,
        )


        b = evaluate_target(
            target_id=2,
            range_m=20.0,
            azimuth_rad=0.0,
            elevation_rad=0.0,
            measurement_covariance=IDENTITY3,
            priority=2.0,
        )


        selected = select_target(
            (a,b)
        )


        self.assertEqual(
            selected.target_id,
            2,
        )



if __name__ == "__main__":
    unittest.main()
