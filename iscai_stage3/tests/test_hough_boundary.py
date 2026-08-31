import ast
import unittest

from pathlib import Path


ROOT = Path(
    "/home/agni/waymo/iscai_stage3/"
    "src/iscai_stage3"
)

HOUGH_SOURCE = (
    ROOT
    / "hough"
    / "multidimensional.py"
)


class TestHoughBoundary(
    unittest.TestCase
):

    def test_hough_is_real_accumulator_pipeline(
        self,
    ):
        text = HOUGH_SOURCE.read_text(
            encoding="utf-8"
        )

        required = (
            "build_sparse_accumulator",
            "extract_hough_peaks",
            "x_anchor_bin",
            "y_anchor_bin",
            "vx_bin",
            "vy_bin",
            "_select_peak_support",
            "_support_is_valid",
        )

        for token in required:
            self.assertIn(
                token,
                text,
            )

    def test_hough_does_not_use_estimated_association(
        self,
    ):
        text = HOUGH_SOURCE.read_text(
            encoding="utf-8"
        )

        forbidden = (
            "associate_estimated_gnn",
            "AssociatedTrack",
            "AssociatedDetection",
            "detection_key(",
        )

        for token in forbidden:
            self.assertNotIn(
                token,
                text,
            )

    def test_hough_has_no_truth_or_class_dependency(
        self,
    ):
        text = HOUGH_SOURCE.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            text,
            filename=str(
                HOUGH_SOURCE
            ),
        )

        # Exact forbidden identifiers/classes/modules.
        forbidden_text_tokens = (
            "DetectionTruthSidecar",
            "DetectionTruthEntry",
            "EvaluatorTruthSequence",
            "source_track_id",
            "scenario.tracks",
            "iscai_stage3_panagiota",
            "Multiple Hypothesis Tracking",
        )

        for token in forbidden_text_tokens:
            self.assertNotIn(
                token,
                text,
            )

        # Exact attribute accesses only.
        #
        # This deliberately does NOT reject
        # audit flags such as:
        #
        #     actor_class_used
        #
        # because those flags explicitly record
        # that class information was NOT used.
        forbidden_attributes = {
            "actor_class",
            "object_type",
            "source_track_id",
            "truth",
        }

        seen_forbidden_attributes = sorted(
            {
                node.attr
                for node in ast.walk(tree)
                if (
                    isinstance(
                        node,
                        ast.Attribute,
                    )
                    and
                    node.attr
                    in
                    forbidden_attributes
                )
            }
        )

        self.assertEqual(
            seen_forbidden_attributes,
            [],
            msg=(
                "Forbidden algorithm-facing "
                "attributes found: "
                f"{seen_forbidden_attributes}"
            ),
        )

        # Also reject exact variable names.
        forbidden_names = {
            "actor_class",
            "object_type",
            "source_track_id",
            "truth",
        }

        seen_forbidden_names = sorted(
            {
                node.id
                for node in ast.walk(tree)
                if (
                    isinstance(
                        node,
                        ast.Name,
                    )
                    and
                    node.id
                    in
                    forbidden_names
                )
            }
        )

        self.assertEqual(
            seen_forbidden_names,
            [],
            msg=(
                "Forbidden algorithm-facing "
                "names found: "
                f"{seen_forbidden_names}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
