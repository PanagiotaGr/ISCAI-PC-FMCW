import unittest


from iscai_stage3.control.closed_loop import (
    run_closed_loop,
    propagate_uncertainty,
)



class TestClosedLoopADB(unittest.TestCase):


    def test_uncertainty_evolves(self):

        value = propagate_uncertainty(
            1.0,
            1.0,
        )

        self.assertGreater(
            value,
            1.0,
        )


    def test_horizon_execution(self):

        result = run_closed_loop(
            target_id=1,
            initial_uncertainty=0.5,
            horizon_s=1.0,
            dt=0.5,
        )

        self.assertEqual(
            len(result.states),
            3,
        )


    def test_no_unnecessary_switching(self):

        result = run_closed_loop(
            target_id=1,
            initial_uncertainty=0.5,
            horizon_s=2.0,
            dt=1.0,
        )

        self.assertEqual(
            result.switches,
            0,
        )


if __name__ == "__main__":
    unittest.main()
