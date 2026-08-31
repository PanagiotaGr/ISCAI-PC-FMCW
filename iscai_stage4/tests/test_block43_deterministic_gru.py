import inspect
import unittest

import torch

from iscai_stage4.ml import (
    CLASS_CYCLIST,
    CLASS_PEDESTRIAN,
    CLASS_VEHICLE,
    DeterministicTrajectoryGRU,
    balanced_class_weights,
    masked_mse_loss,
)


def inputs(
    batch=4,
):
    target = torch.randn(
        batch,
        11,
        14,
    )

    neighbors = torch.randn(
        batch,
        8,
        11,
        14,
    )

    neighbor_mask = torch.ones(
        batch,
        8,
    )

    map_context = torch.randn(
        batch,
        10,
    )

    return (
        target,
        neighbors,
        neighbor_mask,
        map_context,
    )


class TestBlock43DeterministicGRU(
    unittest.TestCase
):

    def test_output_shape(self):
        torch.manual_seed(1)

        model = (
            DeterministicTrajectoryGRU()
        )

        output = model(
            *inputs()
        )

        self.assertEqual(
            tuple(output.shape),
            (
                4,
                4,
                3,
            ),
        )

    def test_forward_finite(self):
        torch.manual_seed(2)

        model = (
            DeterministicTrajectoryGRU()
        )

        output = model(
            *inputs()
        )

        self.assertTrue(
            bool(
                torch.isfinite(
                    output
                ).all()
            )
        )

    def test_masked_neighbor_contents_do_not_matter(self):
        torch.manual_seed(3)

        model = (
            DeterministicTrajectoryGRU()
        )

        model.eval()

        target, n1, _, map_context = (
            inputs(
                batch=2
            )
        )

        n2 = n1 + 1000.0

        mask = torch.zeros(
            2,
            8,
        )

        with torch.no_grad():
            a = model(
                target,
                n1,
                mask,
                map_context,
            )

            b = model(
                target,
                n2,
                mask,
                map_context,
            )

        self.assertTrue(
            torch.equal(
                a,
                b,
            )
        )

    def test_actor_only_ablation_constructs(self):
        model = (
            DeterministicTrajectoryGRU(
                use_neighbors=False,
                use_map=False,
            )
        )

        output = model(
            *inputs(
                batch=2
            )
        )

        self.assertEqual(
            tuple(output.shape),
            (
                2,
                4,
                3,
            ),
        )

    def test_masked_loss_finite(self):
        prediction = torch.randn(
            5,
            4,
            3,
            requires_grad=True,
        )

        target = torch.randn(
            5,
            4,
            3,
        )

        mask = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.float32,
        )

        loss = masked_mse_loss(
            prediction,
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

    def test_zero_valid_mask_is_safe(self):
        prediction = torch.randn(
            2,
            4,
            3,
            requires_grad=True,
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

        loss = masked_mse_loss(
            prediction,
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

    def test_balanced_weights_require_primary_classes(self):
        with self.assertRaises(
            ValueError
        ):
            balanced_class_weights(
                [
                    CLASS_VEHICLE,
                    CLASS_PEDESTRIAN,
                ]
            )

    def test_balanced_primary_class_mass(self):
        ids = (
            [CLASS_VEHICLE] * 100
            +
            [CLASS_PEDESTRIAN] * 10
            +
            [CLASS_CYCLIST] * 2
        )

        weights = (
            balanced_class_weights(
                ids
            )
        )

        ids_tensor = torch.tensor(
            ids
        )

        masses = []

        for class_id in (
            CLASS_VEHICLE,
            CLASS_PEDESTRIAN,
            CLASS_CYCLIST,
        ):
            masses.append(
                weights[
                    ids_tensor
                    ==
                    class_id
                ].sum()
            )

        self.assertTrue(
            torch.allclose(
                masses[0],
                masses[1],
            )
        )

        self.assertTrue(
            torch.allclose(
                masses[1],
                masses[2],
            )
        )

    def test_backward_gradients_are_finite(self):
        torch.manual_seed(4)

        model = (
            DeterministicTrajectoryGRU()
        )

        prediction = model(
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

        loss = masked_mse_loss(
            prediction,
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

    def test_forward_signature_has_no_truth_or_id_inputs(self):
        parameters = set(
            inspect.signature(
                DeterministicTrajectoryGRU
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
        }

        self.assertFalse(
            parameters
            &
            forbidden
        )


if __name__ == "__main__":
    unittest.main()
