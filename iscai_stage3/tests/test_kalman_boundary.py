import unittest

from pathlib import Path


ROOT = Path(
    "/home/agni/waymo/iscai_stage3/"
    "src/iscai_stage3"
)


class TestKalmanBoundary(
    unittest.TestCase
):

    def test_only_real_kalman_predictor_path(
        self,
    ):
        kalman_files = sorted(
            ROOT.rglob("*kalman*.py")
        )

        names = {
            str(path.relative_to(ROOT))
            for path in kalman_files
        }

        self.assertIn(
            "filters/kalman_ekf.py",
            names,
        )

        self.assertIn(
            "baselines/kalman.py",
            names,
        )

        forbidden_names = (
            "kalman_cv_fake",
            "simple_kalman_cv",
            "placeholder_kalman",
        )

        for name in names:
            self.assertFalse(
                any(
                    token in name
                    for token in forbidden_names
                )
            )

    def test_filter_has_no_truth_dependency(
        self,
    ):
        text = (
            ROOT
            / "filters"
            / "kalman_ekf.py"
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
        )

        for token in forbidden:
            self.assertNotIn(
                token,
                text,
            )


if __name__ == "__main__":
    unittest.main()
