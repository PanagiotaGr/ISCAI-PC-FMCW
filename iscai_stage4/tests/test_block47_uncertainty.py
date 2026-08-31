import unittest

import numpy as np
import torch

from iscai_stage4.ml.uncertainty_analysis import (
    balanced_rank_bins,
    latest_measurement_covariance,
    latest_measurement_sigma_rms_m,
    pearson_correlation,
    scale_physical_R_in_normalized_history,
    spearman_correlation,
)


class TestBlock47Uncertainty(
    unittest.TestCase
):

    def _history(self):
        value = np.zeros(
            (
                3,
                11,
                14,
            ),
            dtype=np.float32,
        )

        value[
            :,
            10,
            12
        ] = 1.0

        return value

    def test_identity_covariance_sigma_is_one(self):
        value = self._history()

        value[
            :,
            10,
            6
        ] = 1.0

        value[
            :,
            10,
            9
        ] = 1.0

        value[
            :,
            10,
            11
        ] = 1.0

        result = (
            latest_measurement_sigma_rms_m(
                value
            )
        )

        self.assertTrue(
            np.allclose(
                result[
                    "sigma_rms_m"
                ],
                1.0,
            )
        )

    def test_covariance_reconstruction_symmetric(self):
        value = self._history()

        value[
            :,
            10,
            6:12
        ] = np.asarray(
            [
                4.0,
                0.2,
                0.3,
                5.0,
                0.4,
                6.0,
            ],
            dtype=np.float32,
        )

        covariance = (
            latest_measurement_covariance(
                value
            )
        )

        self.assertTrue(
            np.array_equal(
                covariance,
                np.swapaxes(
                    covariance,
                    -1,
                    -2,
                ),
            )
        )

    def test_off_diagonal_does_not_change_trace_sigma(self):
        first = self._history()
        second = self._history()

        for value in (
            first,
            second,
        ):
            value[
                :,
                10,
                6
            ] = 1.0

            value[
                :,
                10,
                9
            ] = 4.0

            value[
                :,
                10,
                11
            ] = 9.0

        second[
            :,
            10,
            7
        ] = 0.5

        second[
            :,
            10,
            8
        ] = 0.2

        second[
            :,
            10,
            10
        ] = -0.3

        a = (
            latest_measurement_sigma_rms_m(
                first
            )[
                "sigma_rms_m"
            ]
        )

        b = (
            latest_measurement_sigma_rms_m(
                second
            )[
                "sigma_rms_m"
            ]
        )

        self.assertTrue(
            np.array_equal(
                a,
                b,
            )
        )

    def test_physical_scale_normalized_formula(self):
        history = torch.zeros(
            2,
            11,
            14,
        )

        history[
            :,
            :,
            12
        ] = 1.0

        mean = torch.zeros(
            14
        )

        std = torch.ones(
            14
        )

        history[
            ...,
            6:12
        ] = 2.0

        result = (
            scale_physical_R_in_normalized_history(
                history,
                feature_mean=mean,
                feature_std=std,
                variance_scale=2.0,
            )
        )

        self.assertTrue(
            torch.equal(
                result[
                    ...,
                    6:12
                ],
                torch.full_like(
                    result[
                        ...,
                        6:12
                    ],
                    4.0,
                ),
            )
        )

    def test_physical_scale_keeps_missing_rows_zero(self):
        history = torch.zeros(
            1,
            11,
            14,
        )

        mean = torch.ones(
            14
        )

        std = torch.ones(
            14
        )

        result = (
            scale_physical_R_in_normalized_history(
                history,
                feature_mean=mean,
                feature_std=std,
                variance_scale=2.0,
            )
        )

        self.assertEqual(
            int(
                torch.count_nonzero(
                    result
                )
            ),
            0,
        )

    def test_pearson_identity(self):
        values = np.arange(
            100,
            dtype=np.float64,
        )

        self.assertAlmostEqual(
            pearson_correlation(
                values,
                values,
            ),
            1.0,
            places=12,
        )

    def test_spearman_reverse(self):
        values = np.arange(
            100,
            dtype=np.float64,
        )

        self.assertAlmostEqual(
            spearman_correlation(
                values,
                values[
                    ::-1
                ],
            ),
            -1.0,
            places=12,
        )

    def test_balanced_bins_cover_all_samples(self):
        values = np.linspace(
            0.0,
            1.0,
            103,
        )

        bins = balanced_rank_bins(
            values,
            bin_count=5,
        )

        self.assertEqual(
            set(
                bins.tolist()
            ),
            {
                0,
                1,
                2,
                3,
                4,
            },
        )

        counts = [
            int(
                np.sum(
                    bins == index
                )
            )
            for index in range(5)
        ]

        self.assertLessEqual(
            max(counts)
            -
            min(counts),
            1,
        )


if __name__ == "__main__":
    unittest.main()
