import unittest

import numpy as np
import torch

from iscai_stage4.ml.gmm_runtime import (
    denormalize_gmm,
    prediction_sha256,
)


class TestBlock46Runtime(
    unittest.TestCase
):

    def test_denormalized_mean_transform(self):
        means = torch.zeros(
            2,
            3,
            4,
            3,
        )

        scale = (
            torch.eye(3)
            .reshape(
                1,
                1,
                1,
                3,
                3,
            )
            .repeat(
                2,
                3,
                4,
                1,
                1,
            )
        )

        label_mean = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        )

        label_std = torch.ones(
            4,
            3,
        )

        metric_mean, _ = denormalize_gmm(
            means,
            scale,
            label_mean,
            label_std,
        )

        self.assertTrue(
            torch.equal(
                metric_mean[
                    0,
                    0
                ],
                label_mean,
            )
        )

    def test_denormalized_covariance_scale(self):
        means = torch.zeros(
            1,
            3,
            4,
            3,
        )

        scale = (
            torch.eye(3)
            .reshape(
                1,
                1,
                1,
                3,
                3,
            )
            .repeat(
                1,
                3,
                4,
                1,
                1,
            )
        )

        label_mean = torch.zeros(
            4,
            3,
        )

        label_std = torch.tensor(
            [
                [2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0],
            ]
        )

        _, metric_scale = denormalize_gmm(
            means,
            scale,
            label_mean,
            label_std,
        )

        covariance = (
            metric_scale
            @
            metric_scale.transpose(
                -1,
                -2,
            )
        )

        expected = torch.diag(
            torch.tensor(
                [
                    4.0,
                    9.0,
                    16.0,
                ]
            )
        )

        self.assertTrue(
            torch.equal(
                covariance[
                    0,
                    0,
                    0
                ],
                expected,
            )
        )

    def test_prediction_hash_exact_repeat(self):
        torch.manual_seed(
            401
        )

        logits = torch.randn(
            5,
            3,
        )

        means = torch.randn(
            5,
            3,
            4,
            3,
        )

        scale = torch.randn(
            5,
            3,
            4,
            3,
            3,
        )

        first = prediction_sha256(
            logits,
            means,
            scale,
        )

        second = prediction_sha256(
            logits.clone(),
            means.clone(),
            scale.clone(),
        )

        self.assertEqual(
            first,
            second,
        )

    def test_prediction_hash_changes(self):
        logits = torch.zeros(
            2,
            3,
        )

        means = torch.zeros(
            2,
            3,
            4,
            3,
        )

        scale = torch.zeros(
            2,
            3,
            4,
            3,
            3,
        )

        first = prediction_sha256(
            logits,
            means,
            scale,
        )

        changed = means.clone()

        changed[
            0,
            0,
            0,
            0
        ] = 1.0

        second = prediction_sha256(
            logits,
            changed,
            scale,
        )

        self.assertNotEqual(
            first,
            second,
        )


if __name__ == "__main__":
    unittest.main()
