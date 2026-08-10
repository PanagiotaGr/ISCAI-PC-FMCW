import unittest

from iscai_stage1.actors.roles import compute_actor_role_masks


class TestActorRoles(unittest.TestCase):
    def test_context_actor_without_anchor_is_not_target(self) -> None:
        roles = compute_actor_role_masks(
            validity=(True, True, False),
            anchor_index=2,
            object_class="TYPE_PEDESTRIAN",
            is_sdc=False,
            receiver_geometry_valid=False,
        )

        self.assertTrue(roles.is_context_actor)
        self.assertFalse(roles.is_anchor_valid)
        self.assertFalse(roles.is_forecasting_target_candidate)
        self.assertFalse(roles.is_receiver_candidate)

    def test_forecasting_candidate_does_not_require_adjacent_velocity(self) -> None:
        # Frozen edge case:
        # t=0 valid, t=1 valid, t=2 invalid, anchor t=3 valid.
        validity = (True, True, False, True)
        timestamps = (0.0, 0.1, 0.2, 0.3)
        anchor = 3

        roles = compute_actor_role_masks(
            validity=validity,
            anchor_index=anchor,
            object_class="TYPE_PEDESTRIAN",
            is_sdc=False,
            receiver_geometry_valid=False,
        )

        # Adjacent backward-difference validity contract only.
        dt = timestamps[anchor] - timestamps[anchor - 1]
        velocity_valid_anchor = (
            validity[anchor]
            and validity[anchor - 1]
            and dt > 0.0
        )

        self.assertTrue(roles.is_anchor_valid)
        self.assertTrue(roles.is_forecasting_target_candidate)
        self.assertFalse(velocity_valid_anchor)

    def test_future_validity_cannot_change_causal_roles(self) -> None:
        causal = (True, False, True)

        roles_future_invalid = compute_actor_role_masks(
            validity=causal + (False, False, False),
            anchor_index=2,
            object_class="TYPE_VEHICLE",
            is_sdc=False,
            receiver_geometry_valid=True,
        )

        roles_future_valid = compute_actor_role_masks(
            validity=causal + (True, True, True),
            anchor_index=2,
            object_class="TYPE_VEHICLE",
            is_sdc=False,
            receiver_geometry_valid=True,
        )

        self.assertEqual(roles_future_invalid, roles_future_valid)

    def test_receiver_candidate(self) -> None:
        roles = compute_actor_role_masks(
            validity=(True, True),
            anchor_index=1,
            object_class="TYPE_VEHICLE",
            is_sdc=False,
            receiver_geometry_valid=True,
        )

        self.assertTrue(roles.is_receiver_candidate)

    def test_sdc_is_never_receiver_candidate(self) -> None:
        roles = compute_actor_role_masks(
            validity=(True, True),
            anchor_index=1,
            object_class="TYPE_VEHICLE",
            is_sdc=True,
            receiver_geometry_valid=True,
        )

        self.assertFalse(roles.is_receiver_candidate)


if __name__ == "__main__":
    unittest.main()