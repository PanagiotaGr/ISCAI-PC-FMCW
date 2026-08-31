from pathlib import Path
import tempfile
import unittest

import numpy as np

from iscai_stage4.ml.reproducibility_probe_runtime import (
    aggregate_probe_results,
    choose_probe_plan,
    probe_plan_sha256,
)


def write_shard(
    path,
    *,
    scenario_id,
    sample_count,
    prediction_sha,
):
    np.savez(
        path,

        scenario_id=np.asarray(
            scenario_id
        ),

        all_supervised_prediction_count=np.asarray(
            sample_count,
            dtype=np.int64,
        ),

        all_prediction_sha256=np.asarray(
            prediction_sha
        ),

        truth_index_source=np.asarray(
            "sample.truth_track_index"
        ),
    )


class TestBlock49FreshProbe(
    unittest.TestCase
):

    def test_plan_uses_first_nonempty_scenes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(
                root
            )

            rows = [
                {
                    "scenario_id":
                        "a"
                },
                {
                    "scenario_id":
                        "b"
                },
                {
                    "scenario_id":
                        "c"
                },
            ]

            write_shard(
                root / "001_a.npz",
                scenario_id="a",
                sample_count=0,
                prediction_sha="a" * 64,
            )

            write_shard(
                root / "002_b.npz",
                scenario_id="b",
                sample_count=3,
                prediction_sha="b" * 64,
            )

            write_shard(
                root / "003_c.npz",
                scenario_id="c",
                sample_count=4,
                prediction_sha="c" * 64,
            )

            plan = choose_probe_plan(
                rows,
                root,
                count=2,
            )

            self.assertEqual(
                [
                    item[
                        "rank"
                    ]
                    for item in plan
                ],
                [
                    2,
                    3,
                ],
            )

    def test_plan_hash_exact_repeat(self):
        plan = [
            {
                "rank": 1,
                "scenario_id": "x",
                "expected_prediction_sha256":
                    "a" * 64,
                "expected_supervised_samples": 5,
                "truth_index_source":
                    "sample.truth_track_index",
            }
        ]

        self.assertEqual(
            probe_plan_sha256(
                plan
            ),
            probe_plan_sha256(
                list(
                    plan
                )
            ),
        )

    def test_aggregate_exact_match(self):
        plan = [
            {
                "rank": 1,
                "scenario_id": "x",
                "expected_prediction_sha256":
                    "a" * 64,
                "expected_supervised_samples": 5,
                "truth_index_source":
                    "sample.truth_track_index",
            }
        ]

        result = [
            {
                "status":
                    "PASS",

                "rank":
                    1,

                "scenario_id":
                    "x",

                "expected_prediction_sha256":
                    "a" * 64,

                "actual_prediction_sha256":
                    "a" * 64,

                "supervised_samples":
                    5,

                "strict_state_load":
                    True,

                "exact_prediction_match":
                    True,

                "tracks_to_predict_accessed":
                    False,
            }
        ]

        summary = aggregate_probe_results(
            plan,
            result,
        )

        self.assertTrue(
            summary[
                "all_exact"
            ]
        )

    def test_aggregate_rejects_changed_prediction(self):
        plan = [
            {
                "rank": 1,
                "scenario_id": "x",
                "expected_prediction_sha256":
                    "a" * 64,
                "expected_supervised_samples": 5,
                "truth_index_source":
                    "sample.truth_track_index",
            }
        ]

        result = [
            {
                "status":
                    "PASS",

                "rank":
                    1,

                "scenario_id":
                    "x",

                "expected_prediction_sha256":
                    "a" * 64,

                "actual_prediction_sha256":
                    "b" * 64,

                "supervised_samples":
                    5,

                "strict_state_load":
                    True,

                "exact_prediction_match":
                    False,

                "tracks_to_predict_accessed":
                    False,
            }
        ]

        with self.assertRaises(
            RuntimeError
        ):
            aggregate_probe_results(
                plan,
                result,
            )


if __name__ == "__main__":
    unittest.main()
