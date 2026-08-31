import unittest

import torch

from iscai_stage4.ml.ablation_runtime import (
    ABLATION_ACTOR_ONLY,
    ABLATION_FULL,
    ABLATION_NO_MAP,
    ABLATION_NO_R,
    apply_normalized_ablation,
    measurement_covariance_indices,
)


FEATURE_NAMES = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "range",
    "radial_velocity",
    "azimuth",
    "elevation",
    "log_variance_range",
    "log_variance_radial_velocity",
    "observed",
    "auxiliary_flag",
)


def tensors():
    return (
        torch.randn(
            2,
            11,
            14,
        ),
        torch.randn(
            2,
            8,
            11,
            14,
        ),
        torch.ones(
            2,
            8,
        ),
        torch.randn(
            2,
            10,
        ),
    )


class TestBlock47Ablation(
    unittest.TestCase
):

    def test_covariance_discovery(self):
        self.assertEqual(
            measurement_covariance_indices(
                FEATURE_NAMES
            ),
            (
                10,
                11,
            ),
        )

    def test_no_covariance_is_rejected(self):
        names = tuple(
            f"feature_{index}"
            for index in range(
                14
            )
        )

        with self.assertRaises(
            RuntimeError
        ):
            measurement_covariance_indices(
                names
            )

    def test_full_is_exact_identity(self):
        values = tensors()

        result = apply_normalized_ablation(
            *values,
            mode=ABLATION_FULL,
            covariance_indices=(
                10,
                11,
            ),
        )

        for original, returned in zip(
            values,
            result,
        ):
            self.assertIs(
                original,
                returned,
            )

    def test_no_r_zeros_target_covariance(self):
        values = tensors()

        result = apply_normalized_ablation(
            *values,
            mode=ABLATION_NO_R,
            covariance_indices=(
                10,
                11,
            ),
        )

        target = result[0]

        self.assertTrue(
            torch.equal(
                target[
                    ...,
                    10:12
                ],
                torch.zeros_like(
                    target[
                        ...,
                        10:12
                    ]
                ),
            )
        )

    def test_no_r_preserves_non_covariance(self):
        values = tensors()

        result = apply_normalized_ablation(
            *values,
            mode=ABLATION_NO_R,
            covariance_indices=(
                10,
                11,
            ),
        )

        self.assertTrue(
            torch.equal(
                result[
                    0
                ][
                    ...,
                    :10
                ],
                values[
                    0
                ][
                    ...,
                    :10
                ],
            )
        )

        self.assertTrue(
            torch.equal(
                result[
                    0
                ][
                    ...,
                    12:
                ],
                values[
                    0
                ][
                    ...,
                    12:
                ],
            )
        )

    def test_actor_only_removes_neighbours(self):
        values = tensors()

        result = apply_normalized_ablation(
            *values,
            mode=ABLATION_ACTOR_ONLY,
            covariance_indices=(
                10,
                11,
            ),
        )

        self.assertEqual(
            int(
                torch.count_nonzero(
                    result[1]
                )
            ),
            0,
        )

        self.assertEqual(
            int(
                torch.count_nonzero(
                    result[2]
                )
            ),
            0,
        )

    def test_no_map_removes_map_only(self):
        values = tensors()

        result = apply_normalized_ablation(
            *values,
            mode=ABLATION_NO_MAP,
            covariance_indices=(
                10,
                11,
            ),
        )

        self.assertEqual(
            int(
                torch.count_nonzero(
                    result[3]
                )
            ),
            0,
        )

        self.assertTrue(
            torch.equal(
                result[0],
                values[0],
            )
        )

    def test_unknown_mode_rejected(self):
        values = tensors()

        with self.assertRaises(
            ValueError
        ):
            apply_normalized_ablation(
                *values,
                mode="oracle_future",
                covariance_indices=(
                    10,
                    11,
                ),
            )


if __name__ == "__main__":
    unittest.main()
