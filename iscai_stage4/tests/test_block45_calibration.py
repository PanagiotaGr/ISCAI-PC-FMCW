import unittest

import numpy as np
import torch

from iscai_stage4.ml import (
    CovarianceScaleCalibration,
    apply_variance_scale,
    covariance_from_scale_tril,
    fit_per_horizon_covariance_scale,
    reliability_metrics,
    scaled_mahalanobis_squared,
)


class TestBlock45Calibration(
    unittest.TestCase
):

    def test_identity_scale_keeps_cholesky(self):
        torch.manual_seed(101)

        scale = torch.randn(
            3,
            4,
            3,
            3,
        )

        calibrated = (
            apply_variance_scale(
                scale,
                (
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
            )
        )

        self.assertTrue(
            torch.equal(
                scale,
                calibrated,
            )
        )

    def test_variance_scale_changes_covariance_by_alpha(self):
        scale = (
            torch.eye(3)
            .reshape(
                1,
                1,
                3,
                3,
            )
            .repeat(
                2,
                4,
                1,
                1,
            )
        )

        alpha = (
            4.0,
            4.0,
            4.0,
            4.0,
        )

        calibrated = (
            apply_variance_scale(
                scale,
                alpha,
            )
        )

        covariance = (
            covariance_from_scale_tril(
                calibrated
            )
        )

        expected = (
            4.0
            *
            torch.eye(3)
        )

        self.assertTrue(
            torch.equal(
                covariance[
                    0,
                    0
                ],
                expected,
            )
        )

    def test_invalid_scale_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            CovarianceScaleCalibration(
                variance_scale=(
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                )
            )

    def test_scaled_mahalanobis_divides_by_alpha(self):
        q = np.full(
            (
                5,
                4,
            ),
            8.0,
        )

        scaled = (
            scaled_mahalanobis_squared(
                q,
                (
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ),
            )
        )

        self.assertTrue(
            np.array_equal(
                scaled,
                np.full(
                    (
                        5,
                        4,
                    ),
                    4.0,
                ),
            )
        )

    def test_reliability_has_required_levels(self):
        q = np.ones(
            (
                20,
                4,
            )
        )

        mask = np.ones_like(
            q
        )

        metrics = reliability_metrics(
            q,
            mask,
        )

        self.assertEqual(
            tuple(
                metrics[
                    "global_levels"
                ].keys()
            ),
            (
                "0.50",
                "0.80",
                "0.90",
                "0.95",
                "0.99",
            ),
        )

    def test_reliability_ece_finite(self):
        q = np.linspace(
            0.0,
            15.0,
            400,
        ).reshape(
            100,
            4,
        )

        mask = np.ones_like(
            q
        )

        metrics = reliability_metrics(
            q,
            mask,
        )

        self.assertTrue(
            np.isfinite(
                metrics[
                    "coverage_ECE_macro"
                ]
            )
        )

    def test_brier_finite_and_nonnegative(self):
        q = np.linspace(
            0.0,
            12.0,
            800,
        ).reshape(
            200,
            4,
        )

        mask = np.ones_like(
            q
        )

        metrics = reliability_metrics(
            q,
            mask,
        )

        score = metrics[
            "coverage_event_Brier"
        ]

        self.assertTrue(
            np.isfinite(
                score
            )
        )

        self.assertGreaterEqual(
            score,
            0.0,
        )

    def test_masked_values_do_not_change_metrics(self):
        q1 = np.ones(
            (
                20,
                4,
            )
        )

        q2 = q1.copy()

        mask = np.ones_like(
            q1
        )

        mask[
            :5,
            2
        ] = 0.0

        q2[
            :5,
            2
        ] = 1e12

        a = reliability_metrics(
            q1,
            mask,
        )

        b = reliability_metrics(
            q2,
            mask,
        )

        self.assertEqual(
            a[
                "coverage_ECE_macro"
            ],
            b[
                "coverage_ECE_macro"
            ],
        )

        self.assertEqual(
            a[
                "coverage_event_Brier"
            ],
            b[
                "coverage_event_Brier"
            ],
        )

    def test_fit_returns_four_scales(self):
        torch.manual_seed(102)

        residual = torch.randn(
            5000,
            4,
            3,
        )

        q = (
            residual
            .square()
            .sum(
                dim=-1
            )
        )

        mask = torch.ones(
            5000,
            4,
        )

        result = (
            fit_per_horizon_covariance_scale(
                q,
                mask,
            )
        )

        self.assertEqual(
            len(
                result[
                    "calibrator"
                ]
                .variance_scale
            ),
            4,
        )

    def test_underdispersed_synthetic_prefers_inflation(self):
        torch.manual_seed(103)

        residual = torch.randn(
            20000,
            4,
            3,
        )

        q = (
            2.25
            *
            residual
            .square()
            .sum(
                dim=-1
            )
        )

        mask = torch.ones(
            20000,
            4,
        )

        result = (
            fit_per_horizon_covariance_scale(
                q,
                mask,
            )
        )

        self.assertTrue(
            all(
                alpha > 1.0
                for alpha in (
                    result[
                        "calibrator"
                    ]
                    .variance_scale
                )
            )
        )

    def test_fit_does_not_worsen_macro_ece(self):
        torch.manual_seed(104)

        residual = torch.randn(
            15000,
            4,
            3,
        )

        q = (
            1.8
            *
            residual
            .square()
            .sum(
                dim=-1
            )
        )

        mask = torch.ones(
            15000,
            4,
        )

        result = (
            fit_per_horizon_covariance_scale(
                q,
                mask,
            )
        )

        raw = (
            result[
                "raw_reliability"
            ][
                "coverage_ECE_macro"
            ]
        )

        calibrated = (
            result[
                "calibrated_reliability"
            ][
                "coverage_ECE_macro"
            ]
        )

        self.assertLessEqual(
            calibrated,
            raw
            +
            1e-15,
        )

    def test_calibrator_does_not_contain_mean(self):
        calibrator = (
            CovarianceScaleCalibration(
                variance_scale=(
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                )
            )
        )

        payload = (
            calibrator.to_dict()
        )

        self.assertFalse(
            payload[
                "mean_modified"
            ]
        )

        self.assertFalse(
            payload[
                "measurement_covariance_R_t_modified"
            ]
        )

        self.assertTrue(
            payload[
                "predictive_covariance_modified"
            ]
        )


if __name__ == "__main__":
    unittest.main()
