from dataclasses import dataclass
import unittest

import numpy as np
import torch

from iscai_stage4.ml.formal_runtime import (
    extract_deterministic_prediction,
    gmm_formal_metrics,
    resolve_sample_truth_track_index,
    trajectory_point_metrics,
)


@dataclass
class Sample:
    truth_track_index: int


class TestBlock48FormalRuntime(
    unittest.TestCase
):

    def test_truth_index_resolution(self):
        result = (
            resolve_sample_truth_track_index(
                Sample(
                    truth_track_index=17
                )
            )
        )

        self.assertEqual(
            result[
                "index"
            ],
            17,
        )

    def test_prediction_id_is_not_truth_identity(self):
        class OnlyPrediction:
            prediction_id = "abc"

        with self.assertRaises(
            RuntimeError
        ):
            resolve_sample_truth_track_index(
                OnlyPrediction()
            )

    def test_deterministic_tensor_extraction(self):
        value = torch.zeros(
            3,
            4,
            3,
        )

        result = (
            extract_deterministic_prediction(
                value
            )
        )

        self.assertIs(
            value,
            result,
        )

    def test_point_metrics_exact_zero(self):
        mean = np.zeros(
            (
                2,
                4,
                3,
            )
        )

        truth = np.zeros_like(
            mean
        )

        validity = np.ones(
            (
                2,
                4,
            ),
            dtype=bool,
        )

        classes = np.asarray(
            [
                1,
                2,
            ]
        )

        result = trajectory_point_metrics(
            mean,
            truth,
            validity,
            classes,
            class_id_to_name={
                1: "A",
                2: "B",
            },
        )

        self.assertEqual(
            result[
                "ade_m"
            ],
            0.0,
        )

        self.assertEqual(
            result[
                "fde_1.0s_m"
            ],
            0.0,
        )

    def test_gmm_identical_components_zero_minade(self):
        logits = np.zeros(
            (
                2,
                3,
            )
        )

        means = np.zeros(
            (
                2,
                3,
                4,
                3,
            )
        )

        truth = np.zeros(
            (
                2,
                4,
                3,
            )
        )

        validity = np.ones(
            (
                2,
                4,
            ),
            dtype=bool,
        )

        result = gmm_formal_metrics(
            logits,
            means,
            truth,
            validity,
        )

        self.assertEqual(
            result[
                "minADE_m"
            ],
            0.0,
        )

        self.assertAlmostEqual(
            result[
                "effective_mode_count"
            ],
            3.0,
        )

    def test_gmm_map_component_selection(self):
        logits = np.asarray(
            [
                [
                    0.0,
                    1.0,
                    5.0,
                ]
            ]
        )

        means = np.zeros(
            (
                1,
                3,
                4,
                3,
            )
        )

        means[
            0,
            2,
            :,
            0
        ] = 9.0

        truth = np.zeros(
            (
                1,
                4,
                3,
            )
        )

        validity = np.ones(
            (
                1,
                4,
            ),
            dtype=bool,
        )

        result = gmm_formal_metrics(
            logits,
            means,
            truth,
            validity,
        )

        self.assertTrue(
            np.array_equal(
                result[
                    "MAP_mean"
                ][
                    0
                ],
                means[
                    0,
                    2
                ],
            )
        )


if __name__ == "__main__":
    unittest.main()
