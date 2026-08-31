import inspect
import unittest

import torch

from iscai_stage4.ml import (
    CHI_SQUARE_3_THRESHOLDS,
    CONFIDENCE_LEVELS,
    DeterministicTrajectoryGRU,
    GaussianTrajectoryGRU,
    covariance_from_scale_tril,
    denormalize_gaussian,
    empirical_coverage,
    gaussian_nll_per_horizon,
    initialize_from_deterministic,
    masked_gaussian_nll,
    scale_tril_from_raw,
)


def inputs(
    batch=4,
):
    return (
        torch.randn(
            batch,
            11,
            14,
        ),

        torch.randn(
            batch,
            8,
            11,
            14,
        ),

        torch.ones(
            batch,
            8,
        ),

        torch.randn(
            batch,
            10,
        ),
    )


class TestBlock44GaussianGRU(
    unittest.TestCase
):

    def test_output_shapes(self):
        torch.manual_seed(10)

        model = (
            GaussianTrajectoryGRU()
        )

        output = model(
            *inputs()
        )

        self.assertEqual(
            tuple(
                output.mean.shape
            ),
            (
                4,
                4,
                3,
            ),
        )

        self.assertEqual(
            tuple(
                output
                .scale_tril
                .shape
            ),
            (
                4,
                4,
                3,
                3,
            ),
        )

    def test_cholesky_diagonal_positive(self):
        raw = torch.randn(
            5,
            4,
            6,
        )

        scale = (
            scale_tril_from_raw(
                raw
            )
        )

        diagonal = torch.diagonal(
            scale,
            dim1=-2,
            dim2=-1,
        )

        self.assertTrue(
            bool(
                (
                    diagonal
                    >
                    0.0
                ).all()
            )
        )

    def test_covariance_symmetric(self):
        raw = torch.randn(
            3,
            4,
            6,
        )

        scale = (
            scale_tril_from_raw(
                raw
            )
        )

        covariance = (
            covariance_from_scale_tril(
                scale
            )
        )

        self.assertTrue(
            torch.allclose(
                covariance,
                covariance.transpose(
                    -1,
                    -2,
                ),
            )
        )

    def test_covariance_positive_definite(self):
        raw = torch.randn(
            8,
            4,
            6,
        )

        scale = (
            scale_tril_from_raw(
                raw
            )
        )

        covariance = (
            covariance_from_scale_tril(
                scale
            )
        )

        reconstructed = (
            torch.linalg.cholesky(
                covariance
            )
        )

        self.assertTrue(
            bool(
                torch.isfinite(
                    reconstructed
                ).all()
            )
        )

    def test_nll_finite(self):
        torch.manual_seed(11)

        model = (
            GaussianTrajectoryGRU()
        )

        output = model(
            *inputs()
        )

        target = torch.randn(
            4,
            4,
            3,
        )

        mask = torch.ones(
            4,
            4,
        )

        loss = masked_gaussian_nll(
            output.mean,
            output.scale_tril,
            target,
            mask,
        )

        self.assertTrue(
            bool(
                torch.isfinite(
                    loss
                )
            )
        )

    def test_mask_excludes_invalid_target(self):
        mean = torch.zeros(
            1,
            4,
            3,
        )

        raw = torch.zeros(
            1,
            4,
            6,
        )

        scale = (
            scale_tril_from_raw(
                raw
            )
        )

        target_a = torch.zeros(
            1,
            4,
            3,
        )

        target_b = (
            target_a.clone()
        )

        target_b[
            0,
            2,
            :
        ] = 1_000_000.0

        mask = torch.tensor(
            [
                [
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                ]
            ]
        )

        loss_a = masked_gaussian_nll(
            mean,
            scale,
            target_a,
            mask,
        )

        loss_b = masked_gaussian_nll(
            mean,
            scale,
            target_b,
            mask,
        )

        self.assertEqual(
            float(loss_a),
            float(loss_b),
        )

    def test_zero_mask_safe(self):
        mean = torch.zeros(
            2,
            4,
            3,
            requires_grad=True,
        )

        raw = torch.zeros(
            2,
            4,
            6,
            requires_grad=True,
        )

        scale = (
            scale_tril_from_raw(
                raw
            )
        )

        target = torch.randn(
            2,
            4,
            3,
        )

        mask = torch.zeros(
            2,
            4,
        )

        loss = masked_gaussian_nll(
            mean,
            scale,
            target,
            mask,
        )

        loss.backward()

        self.assertEqual(
            float(
                loss.detach()
            ),
            0.0,
        )

    def test_backward_gradients_finite(self):
        torch.manual_seed(12)

        model = (
            GaussianTrajectoryGRU()
        )

        output = model(
            *inputs(
                batch=3
            )
        )

        target = torch.randn(
            3,
            4,
            3,
        )

        mask = torch.ones(
            3,
            4,
        )

        loss = masked_gaussian_nll(
            output.mean,
            output.scale_tril,
            target,
            mask,
        )

        loss.backward()

        gradients = [
            parameter.grad
            for parameter
            in model.parameters()
            if parameter.grad
            is not None
        ]

        self.assertTrue(
            gradients
        )

        self.assertTrue(
            all(
                bool(
                    torch.isfinite(
                        gradient
                    ).all()
                )
                for gradient
                in gradients
            )
        )

    def test_deterministic_initialization_mean_exact(self):
        torch.manual_seed(13)

        deterministic = (
            DeterministicTrajectoryGRU()
        )

        gaussian = (
            GaussianTrajectoryGRU()
        )

        initialize_from_deterministic(
            gaussian,
            deterministic.state_dict(),
            initial_std_normalized=0.5,
        )

        args = inputs(
            batch=5
        )

        deterministic.eval()
        gaussian.eval()

        with torch.no_grad():
            deterministic_mean = (
                deterministic(
                    *args
                )
            )

            gaussian_mean = (
                gaussian(
                    *args
                ).mean
            )

        self.assertTrue(
            torch.equal(
                deterministic_mean,
                gaussian_mean,
            )
        )

    def test_initial_std_is_half(self):
        torch.manual_seed(14)

        deterministic = (
            DeterministicTrajectoryGRU()
        )

        gaussian = (
            GaussianTrajectoryGRU()
        )

        initialize_from_deterministic(
            gaussian,
            deterministic.state_dict(),
            initial_std_normalized=0.5,
        )

        output = gaussian(
            *inputs(
                batch=2
            )
        )

        diagonal = torch.diagonal(
            output.scale_tril,
            dim1=-2,
            dim2=-1,
        )

        self.assertTrue(
            torch.allclose(
                diagonal,
                torch.full_like(
                    diagonal,
                    0.5,
                ),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_nll_matches_torch_distribution(self):
        dtype = torch.float64

        mean = torch.tensor(
            [
                [
                    [0.2, -0.1, 0.5]
                ]
            ],
            dtype=dtype,
        )

        scale = torch.tensor(
            [
                [
                    [
                        [1.2, 0.0, 0.0],
                        [0.2, 0.8, 0.0],
                        [-0.1, 0.3, 1.1],
                    ]
                ]
            ],
            dtype=dtype,
        )

        target = torch.tensor(
            [
                [
                    [0.5, 0.3, -0.2]
                ]
            ],
            dtype=dtype,
        )

        ours = (
            gaussian_nll_per_horizon(
                mean,
                scale,
                target,
            )[0, 0]
        )

        distribution = (
            torch.distributions
            .MultivariateNormal(
                mean[0, 0],
                scale_tril=(
                    scale[
                        0,
                        0,
                    ]
                ),
            )
        )

        reference = (
            -distribution.log_prob(
                target[
                    0,
                    0,
                ]
            )
        )

        self.assertTrue(
            torch.allclose(
                ours,
                reference,
                atol=1e-10,
                rtol=1e-10,
            )
        )

    def test_denormalized_covariance_transform(self):
        mean = torch.zeros(
            1,
            4,
            3,
        )

        scale = (
            torch.eye(3)
            .reshape(
                1,
                1,
                3,
                3,
            )
            .repeat(
                1,
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

        _, metric_scale = (
            denormalize_gaussian(
                mean,
                scale,
                label_mean,
                label_std,
            )
        )

        covariance = (
            covariance_from_scale_tril(
                metric_scale
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
                ],
                expected,
            )
        )

    def test_confidence_levels_exact(self):
        self.assertEqual(
            CONFIDENCE_LEVELS,
            (
                0.50,
                0.80,
                0.90,
                0.95,
                0.99,
            ),
        )

        self.assertEqual(
            set(
                CHI_SQUARE_3_THRESHOLDS
            ),
            set(
                CONFIDENCE_LEVELS
            ),
        )

    def test_empirical_coverage_perfect_origin(self):
        mean = torch.zeros(
            2,
            4,
            3,
        )

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

        target = torch.zeros(
            2,
            4,
            3,
        )

        mask = torch.ones(
            2,
            4,
        )

        result = empirical_coverage(
            mean,
            scale,
            target,
            mask,
        )

        for value in (
            result.values()
        ):
            self.assertEqual(
                value[
                    "empirical"
                ],
                1.0,
            )

    def test_model_forward_has_no_truth_or_id_inputs(self):
        parameters = set(
            inspect.signature(
                GaussianTrajectoryGRU
                .forward
            ).parameters
        )

        forbidden = {
            "track_id",
            "prediction_id",
            "actor_class",
            "truth",
            "future",
            "tracks_to_predict",
            "objects_of_interest",
        }

        self.assertFalse(
            parameters
            &
            forbidden
        )

    def test_measurement_covariance_features_reach_GRU(self):
        model = (
            GaussianTrajectoryGRU()
        )

        self.assertEqual(
            model
            .target_gru
            .input_size,
            14,
        )

        self.assertEqual(
            model
            .neighbor_gru
            .input_size,
            14,
        )


if __name__ == "__main__":
    unittest.main()
