import unittest

from pathlib import Path


ROOT = Path(
    "/home/agni/waymo/iscai_stage3/"
    "src/iscai_stage3/association"
)


class TestAssociationBoundary(
    unittest.TestCase
):

    def test_association_has_no_truth_dependency(
        self,
    ):
        forbidden = (
            "DetectionTruthEntry",
            "DetectionTruthSidecar",
            "EvaluatorTruthSequence",
            "scenario.tracks",
            "actor_id",
            "source_track_id",
            "object_type",
        )

        offenders = []

        for path in ROOT.rglob("*.py"):
            text = path.read_text(
                encoding="utf-8"
            )

            if any(
                token in text
                for token in forbidden
            ):
                offenders.append(
                    path.name
                )

        self.assertEqual(
            offenders,
            [],
        )


if __name__ == "__main__":
    unittest.main()
