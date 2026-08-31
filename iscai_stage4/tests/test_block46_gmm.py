import inspect
import unittest

import torch

from iscai_stage4.ml.gaussian_gru import (
    GaussianTrajectoryGRU,
)

from iscai_stage4.ml.gaussian_math import (
    gaussian_nll_per_horizon,
)

from iscai_stage4.ml.gmm_gru import (
    CENTRAL_COMPONENT_INDEX,
    GMM_COMPONENTS,
    GMMTrajectoryGRU,
    initialize_gmm_from_gaussian,
)

from iscai_stage4.ml.gmm_math import (
    map_component_mean,
    masked_gmm_joint_nll,
    mixture_mean,
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


class TestBlock46GMM(
    unittest.TestCase
):

    def test_output_shapes(self):
        torch.manual_seed(301)

        model = GMMTrajectoryGRU()

        output = model(
            *inputs()
        )

        self.assertEqual(
            tuple(
                output.means.shape
            ),
            (
                4,
                3,
                4,
                3,
            ),
        )

        self.assertEqual(
            tuple(
                output.scale_tril.shape
            ),
            (
                4,
                3,
                4,
                3,
                3,
            ),
        )

        self.assertEqual(
            tuple(
                output.mixture_logits.shape
            ),
            (
                4,
                3,
            ),
        )

    def test_probabilities_sum_to_one(self):
        model = GMMTrajectoryGRU()

        output = model(
            *inputs()
        )

        total = (
            output
            .mixture_probabilities
            .sum(
                dim=-1
            )
        )

        self.assertTrue(
            torch.allclose(
                total,
                torch.ones_like(
                    total
                ),
            )
        )

    def test_component_covariances_spd(self):
        model = GMMTrajectoryGRU()

        output = model(
            *inputs(
                batch=2
            )
        )

        covariance = (
            output.covariance
        )

        chol = torch.linalg.cholesky(
            covariance
        )

        self.assertTrue(
            bool(
                torch.isfinite(
                    chol
                ).all()
            )
        )

    def test_central_component_exact_gaussian_mean(self):
        torch.manual_seed(302)

        gaussian = (
            GaussianTrajectoryGRU()
        )

        gmm = GMMTrajectoryGRU()

        initialize_gmm_from_gaussian(
            gmm,
            gaussian.state_dict(),
        )

        args = inputs(
            batch=5
        )

        gaussian.eval()
        gmm.eval()

        with torch.no_grad():
            gaussian_mean = gaussian(
                *args
            ).mean

            gmm_mean = gmm(
                *args
            ).means[
                :,
                CENTRAL_COMPONENT_INDEX,
            ]

        self.assertTrue(
            torch.equal(
                gaussian_mean,
                gmm_mean,
            )
        )

    def test_initial_symmetry_break(self):
        torch.manual_seed(303)

        gaussian = (
            GaussianTrajectoryGRU()
        )

        gmm = GMMTrajectoryGRU()

        initialize_gmm_from_gaussian(
            gmm,
            gaussian.state_dict(),
        )

        output = gmm(
            *inputs(
                batch=2
            )
        )

        negative_delta = (
            output.means[
                :,
                0,
                :,
                1
            ]
            -
            output.means[
                :,
                1,
                :,
                1
            ]
        )

        positive_delta = (
            output.means[
                :,
                2,
                :,
                1
            ]
            -
            output.means[
                :,
                1,
                :,
                1
            ]
        )

        self.assertTrue(
            torch.allclose(
                negative_delta,
                torch.full_like(
                    negative_delta,
                    -0.05,
                ),
                atol=1e-6,
                rtol=0.0,
            )
        )

        self.assertTrue(
            torch.allclose(
                positive_delta,
                torch.full_like(
                    positive_delta,
                    0.05,
                ),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_identical_components_equal_single_gaussian_joint_nll(self):
        torch.manual_seed(304)

        batch = 7

        mean = torch.randn(
            batch,
            4,
            3,
        )

        base = (
            torch.eye(3)
            .reshape(
                1,
                1,
                3,
                3,
            )
            .repeat(
                batch,
                4,
                1,
                1,
            )
        )

        target = torch.randn(
            batch,
            4,
            3,
        )

        means = (
            mean[
                :,
                None,
                :,
                :
            ]
            .repeat(
                1,
                GMM_COMPONENTS,
                1,
                1,
            )
        )

        scale = (
            base[
                :,
                None,
                :,
                :,
                :
            ]
            .repeat(
                1,
                GMM_COMPONENTS,
                1,
                1,
                1,
            )
        )

        logits = torch.zeros(
            batch,
            GMM_COMPONENTS,
        )

        mask = torch.ones(
            batch,
            4,
        )

        gmm_loss = (
            masked_gmm_joint_nll(
                logits,
                means,
                scale,
                target,
                mask,
            )
        )

        gaussian_loss = (
            gaussian_nll_per_horizon(
                mean,
                base,
                target,
            )
            .sum(
                dim=-1
            )
            .mean()
        )

        self.assertTrue(
            torch.allclose(
                gmm_loss,
                gaussian_loss,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_masked_invalid_target_does_not_matter(self):
        torch.manual_seed(305)

        batch = 3

        means = torch.zeros(
            batch,
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
                batch,
                3,
                4,
                1,
                1,
            )
        )

        logits = torch.zeros(
            batch,
            3,
        )

        target_a = torch.zeros(
            batch,
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
        ] = 1e9

        mask = torch.ones(
            batch,
            4,
        )

        mask[
            0,
            2
        ] = 0.0

        a = masked_gmm_joint_nll(
            logits,
            means,
            scale,
            target_a,
            mask,
        )

        b = masked_gmm_joint_nll(
            logits,
            means,
            scale,
            target_b,
            mask,
        )

        self.assertEqual(
            float(a),
            float(b),
        )

    def test_zero_valid_mask_safe(self):
        logits = torch.zeros(
            2,
            3,
            requires_grad=True,
        )

        means = torch.randn(
            2,
            3,
            4,
            3,
            requires_grad=True,
        )

        raw = torch.zeros(
            2,
            3,
            4,
            6,
            requires_grad=True,
        )

        from iscai_stage4.ml.gaussian_gru import (
            scale_tril_from_raw,
        )

        scale = scale_tril_from_raw(
            raw
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

        loss = masked_gmm_joint_nll(
            logits,
            means,
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

    def test_model_backward_finite(self):
        torch.manual_seed(306)

        model = GMMTrajectoryGRU()

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

        loss = masked_gmm_joint_nll(
            output.mixture_logits,
            output.means,
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

    def test_uniform_mixture_mean_is_component_average(self):
        means = torch.zeros(
            2,
            3,
            4,
            3,
        )

        means[
            :,
            0,
        ] = -1.0

        means[
            :,
            1,
        ] = 0.0

        means[
            :,
            2,
        ] = 1.0

        logits = torch.zeros(
            2,
            3,
        )

        result = mixture_mean(
            logits,
            means,
        )

        self.assertTrue(
            torch.allclose(
                result,
                torch.zeros_like(
                    result
                ),
            )
        )

    def test_map_component_mean(self):
        means = torch.zeros(
            2,
            3,
            4,
            3,
        )

        means[
            0,
            2,
        ] = 7.0

        means[
            1,
            0,
        ] = 9.0

        logits = torch.tensor(
            [
                [0.0, 1.0, 5.0],
                [8.0, 0.0, 1.0],
            ]
        )

        result = map_component_mean(
            logits,
            means,
        )

        self.assertTrue(
            torch.equal(
                result[
                    0
                ],
                means[
                    0,
                    2
                ],
            )
        )

        self.assertTrue(
            torch.equal(
                result[
                    1
                ],
                means[
                    1,
                    0
                ],
            )
        )

    def test_forward_has_no_truth_or_mode_labels(self):
        parameters = set(
            inspect.signature(
                GMMTrajectoryGRU.forward
            ).parameters
        )

        forbidden = {
            "track_id",
            "prediction_id",
            "actor_class",
            "truth",
            "future",
            "maneuver",
            "mode_label",
            "tracks_to_predict",
            "objects_of_interest",
        }

        self.assertFalse(
            parameters
            &
            forbidden
        )


if __name__ == "__main__":
    unittest.main()
