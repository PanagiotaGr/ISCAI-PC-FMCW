import unittest

from pathlib import Path


ROOT = Path(
    "/home/agni/waymo/iscai_stage3/"
    "src/iscai_stage3"
)


class TestIMMBoundary(unittest.TestCase):

    def test_imm_is_not_a_model_selector(
        self,
    ):
        text = (
            ROOT
            / "filters"
            / "imm.py"
        ).read_text(
            encoding="utf-8"
        )

        required = (
            "interact_mode_states",
            "mixing_probabilities",
            "transition_matrix",
            "log_likelihood",
            "posterior_mode_probabilities",
            "combine_imm_states",
        )

        for token in required:
            self.assertIn(
                token,
                text,
            )

    def test_imm_has_three_required_modes(
        self,
    ):
        from iscai_stage3.filters import (
            MODE_NAMES,
        )

        self.assertEqual(
            MODE_NAMES,
            (
                "CV",
                "CA",
                "CTRV",
            ),
        )

    def test_imm_has_no_truth_dependency(
        self,
    ):
        text = (
            ROOT
            / "filters"
            / "imm.py"
        ).read_text(
            encoding="utf-8"
        )

        forbidden = (
            "DetectionTruthSidecar",
            "DetectionTruthEntry",
            "EvaluatorTruthSequence",
            "source_track_id",
            "scenario.tracks",
            "object_type",
            "iscai_stage3_panagiota",
        )

        for token in forbidden:
            self.assertNotIn(
                token,
                text,
            )


if __name__ == "__main__":
    unittest.main()
