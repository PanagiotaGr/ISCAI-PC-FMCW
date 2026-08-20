import unittest


from iscai_stage3.control.temporal_adb import (
    uncertainty_growth,
    build_temporal_schedule,
)



class Dummy:

    def __init__(
        self,
        target_id,
        utility,
        uncertainty_penalty,
    ):
        self.target_id = target_id
        self.utility = utility
        self.uncertainty_penalty = uncertainty_penalty



class TestTemporalADB(unittest.TestCase):


    def test_uncertainty_increases_with_time(self):

        value = uncertainty_growth(
            1.0,
            2.0,
        )

        self.assertGreater(
            value,
            1.0,
        )


    def test_schedule_length(self):

        targets = (
            Dummy(
                1,
                5.0,
                0.1,
            ),
            Dummy(
                2,
                2.0,
                0.2,
            ),
        )


        result = build_temporal_schedule(
            targets,
            horizon_s=1.0,
            dt=0.5,
        )


        self.assertEqual(
            len(result.steps),
            3,
        )


    def test_best_target_selected(self):

        targets = (
            Dummy(
                1,
                5.0,
                0.1,
            ),
            Dummy(
                2,
                1.0,
                0.1,
            ),
        )


        result = build_temporal_schedule(
            targets,
            1.0,
            1.0,
        )


        self.assertEqual(
            result.steps[0].target_id,
            1,
        )


if __name__ == "__main__":
    unittest.main()
