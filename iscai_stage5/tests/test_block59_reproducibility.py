from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(
    "/home/agni/waymo/iscai_stage5"
)

REPORT = (
    ROOT
    /
    "reports/block59_reproducibility.json"
)

A = (
    ROOT
    /
    "artifacts/block59/fresh_process_A.json"
)

B = (
    ROOT
    /
    "artifacts/block59/fresh_process_B.json"
)


class Block59ReproducibilityTest(
    unittest.TestCase
):
    def test_block59_candidate_report(
        self,
    ):
        self.assertTrue(
            REPORT.is_file()
        )

        report = json.loads(
            REPORT.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            report[
                "stage"
            ],
            5,
        )

        self.assertEqual(
            report[
                "block"
            ],
            "5.9",
        )

        self.assertEqual(
            report[
                "status"
            ],
            "PASS_FROZEN",
        )

        self.assertTrue(
            report[
                "exact_match_to_frozen_block58"
            ]
        )

        self.assertTrue(
            report[
                "fresh_process_repeat_exact"
            ]
        )

        self.assertEqual(
            report[
                "predetermined_ranks"
            ],
            [
                1,
                60,
                120,
            ],
        )

    def test_fresh_process_outputs(
        self,
    ):
        self.assertTrue(
            A.is_file()
        )

        self.assertTrue(
            B.is_file()
        )

        a = json.loads(
            A.read_text(
                encoding="utf-8"
            )
        )

        b = json.loads(
            B.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            a[
                "status"
            ],
            "PASS",
        )

        self.assertEqual(
            b[
                "status"
            ],
            "PASS",
        )

        self.assertTrue(
            a[
                "all_fresh_semantics_equal_frozen"
            ]
        )

        self.assertTrue(
            b[
                "all_fresh_semantics_equal_frozen"
            ]
        )

        self.assertEqual(
            a[
                "semantic_signature_sha256"
            ],
            b[
                "semantic_signature_sha256"
            ],
        )

        self.assertEqual(
            a[
                "trace_signature_sha256"
            ],
            b[
                "trace_signature_sha256"
            ],
        )

        for category in (
            "receiver_posterior",
            "beam_probability",
            "topk_selection",
            "optical_metrics",
        ):
            self.assertGreater(
                a[
                    "trace_category_counts"
                ][
                    category
                ],
                0,
            )

            self.assertEqual(
                a[
                    "trace_category_counts"
                ][
                    category
                ],
                b[
                    "trace_category_counts"
                ][
                    category
                ],
            )

        self.assertGreater(
            a[
                "receiver_posterior_embedded_array_hash_count"
            ],
            0,
        )

        self.assertEqual(
            a[
                "receiver_posterior_embedded_array_hash_count"
            ],
            b[
                "receiver_posterior_embedded_array_hash_count"
            ],
        )

    def test_scene_level_exact_repeat(
        self,
    ):
        a = json.loads(
            A.read_text(
                encoding="utf-8"
            )
        )

        b = json.loads(
            B.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            len(
                a[
                    "scenes"
                ]
            ),
            3,
        )

        self.assertEqual(
            len(
                b[
                    "scenes"
                ]
            ),
            3,
        )

        for scene_a, scene_b in zip(
            a[
                "scenes"
            ],
            b[
                "scenes"
            ],
        ):
            self.assertEqual(
                scene_a[
                    "rank"
                ],
                scene_b[
                    "rank"
                ],
            )

            self.assertTrue(
                scene_a[
                    "exact_match_to_frozen"
                ]
            )

            self.assertTrue(
                scene_b[
                    "exact_match_to_frozen"
                ]
            )

            self.assertEqual(
                scene_a[
                    "fresh_semantic_sha256"
                ],
                scene_b[
                    "fresh_semantic_sha256"
                ],
            )

            self.assertEqual(
                scene_a[
                    "trace_sha256"
                ],
                scene_b[
                    "trace_sha256"
                ],
            )


if __name__ == "__main__":
    unittest.main()
