import unittest


from iscai_stage3.adb.controller import (
    ADBTargetScore,
    compute_adb_utility,
    choose_best_target,
)



class TestADBController(unittest.TestCase):


    def test_high_confidence_wins(self):

        utility = compute_adb_utility(
            beam_confidence=1.0,
            priority=2.0,
            uncertainty_penalty=0.1,
        )

        self.assertAlmostEqual(
            utility,
            1.9,
        )


    def test_uncertainty_penalty(self):

        low = compute_adb_utility(
            beam_confidence=1.0,
            priority=1.0,
            uncertainty_penalty=0.1,
        )


        high = compute_adb_utility(
            beam_confidence=1.0,
            priority=1.0,
            uncertainty_penalty=1.0,
        )


        self.assertGreater(
            low,
            high,
        )


    def test_best_target_selection(self):

        targets = (
            ADBTargetScore(
                target_id=1,
                beam_confidence=1.0,
                priority=1.0,
                uncertainty_penalty=0.2,
                utility=0.8,
            ),

            ADBTargetScore(
                target_id=2,
                beam_confidence=1.0,
                priority=2.0,
                uncertainty_penalty=0.1,
                utility=1.9,
            ),
        )


        result = choose_best_target(
            targets
        )


        self.assertEqual(
            result.target_id,
            2,
        )


    def test_empty_selection_rejected(self):

        with self.assertRaises(ValueError):

            choose_best_target(
                ()
            )


if __name__ == "__main__":
    unittest.main()
