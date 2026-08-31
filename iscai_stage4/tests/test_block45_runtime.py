import unittest

import numpy as np
import torch

from iscai_stage4.ml.gaussian_math import (
    gaussian_nll_per_horizon,
)

from iscai_stage4.ml.calibration_runtime import (
    calibrated_nll_per_horizon,
    masked_nll_metrics,
    masked_planar_ade,
)


class TestBlock45Runtime(
    unittest.TestCase
):

    def test_identity_alpha_keeps_nll(self):
        raw = np.arange(
            40,
            dtype=np.float64,
        ).reshape(
            10,
            4,
        )

        q = np.ones_like(
            raw
        )

        calibrated = (
            calibrated_nll_per_horizon(
                raw,
                q,
                (
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
            )
        )

        self.assertTrue(
            np.array_equal(
                raw,
                calibrated,
            )
        )

    def test_nll_adjustment_matches_direct_gaussian(self):
        torch.manual_seed(
            201
        )

        mean = torch.zeros(
            8,
            4,
            3,
            dtype=torch.float64,
        )

        raw_scale = (
            torch.eye(
                3,
                dtype=torch.float64,
            )
            .reshape(
                1,
                1,
                3,
                3,
            )
            .repeat(
                8,
                4,
                1,
                1,
            )
        )

        target = torch.randn(
            8,
            4,
            3,
            dtype=torch.float64,
        )

        raw_nll = (
            gaussian_nll_per_horizon(
                mean,
                raw_scale,
                target,
            )
        )

        q = target.square().sum(
            dim=-1
        )

        alpha = np.asarray(
            (
                2.0,
                0.8,
                1.5,
                3.0,
            ),
            dtype=np.float64,
        )

        adjusted = (
            calibrated_nll_per_horizon(
                raw_nll.numpy(),
                q.numpy(),
                alpha,
            )
        )

        alpha_torch = torch.tensor(
            alpha,
            dtype=torch.float64,
        )

        calibrated_scale = (
            raw_scale
            *
            torch.sqrt(
                alpha_torch
            )[
                None,
                :,
                None,
                None
            ]
        )

        direct = (
            gaussian_nll_per_horizon(
                mean,
                calibrated_scale,
                target,
            )
            .numpy()
        )

        self.assertTrue(
            np.allclose(
                adjusted,
                direct,
                atol=1e-12,
                rtol=1e-12,
            )
        )

    def test_masked_nll_ignores_invalid(self):
        nll_a = np.ones(
            (
                5,
                4,
            ),
            dtype=np.float64,
        )

        nll_b = (
            nll_a.copy()
        )

        mask = np.ones_like(
            nll_a
        )

        mask[
            0,
            2
        ] = 0.0

        nll_b[
            0,
            2
        ] = 1e12

        a = masked_nll_metrics(
            nll_a,
            mask,
        )

        b = masked_nll_metrics(
            nll_b,
            mask,
        )

        self.assertEqual(
            a,
            b,
        )

    def test_planar_ade_metric(self):
        error = np.asarray(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
            ]
        )

        mask = np.ones_like(
            error
        )

        result = (
            masked_planar_ade(
                error,
                mask,
            )
        )

        self.assertEqual(
            result[
                "planar_ADE_m"
            ],
            2.5,
        )

        self.assertEqual(
            result[
                "horizon_planar_error_m"
            ][
                "1.0"
            ],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
