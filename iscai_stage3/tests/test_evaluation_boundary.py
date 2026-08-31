import unittest

from pathlib import Path


ROOT = Path(
    "/home/agni/waymo/iscai_stage3/"
    "src/iscai_stage3"
)


class TestEvaluationBoundary(
    unittest.TestCase
):

    def test_algorithm_packages_do_not_import_evaluator(
        self,
    ):
        algorithm_roots = (
            ROOT / "association",
            ROOT / "baselines",
            ROOT / "filters",
            ROOT / "hough",
            ROOT / "state",
        )

        offending = []

        for base in algorithm_roots:
            for path in base.rglob(
                "*.py"
            ):
                text = path.read_text(
                    encoding="utf-8"
                )

                if (
                    "iscai_stage3.evaluation"
                    in text
                ):
                    offending.append(
                        str(path)
                    )

        self.assertEqual(
            offending,
            [],
        )

    def test_assignment_source_never_reads_future_points(
        self,
    ):
        text = (
            ROOT
            / "evaluation"
            / "assignment.py"
        ).read_text(
            encoding="utf-8"
        )

        forbidden = (
            ".points",
            "horizon_s",
            "TruthTrajectoryPoint",
        )

        for token in forbidden:
            self.assertNotIn(
                token,
                text,
            )


if __name__ == "__main__":
    unittest.main()
