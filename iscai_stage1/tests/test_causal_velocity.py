import unittest

from iscai_stage1.actors.velocity import (
    derive_adjacent_backward_velocity,
)
from iscai_stage1.contracts.stage1a import ZERO_VEC3


class TestAdjacentBackwardVelocity(unittest.TestCase):
    def test_adjacent_backward_difference(self) -> None:
        result = derive_adjacent_backward_velocity(
            timestamps_s=(0.0, 0.1, 0.2),
            positions_W_m=(
                (0.0, 0.0, 0.0),
                (1.0, 2.0, 0.0),
                (3.0, 6.0, 0.0),
            ),
            state_valid=(True, True, True),
        )

        self.assertEqual(
            result.velocity_valid,
            (False, True, True),
        )

        self.assertEqual(result.velocity_W_mps[0], ZERO_VEC3)

        self.assertAlmostEqual(result.velocity_W_mps[1][0], 10.0)
        self.assertAlmostEqual(result.velocity_W_mps[1][1], 20.0)

        self.assertAlmostEqual(result.velocity_W_mps[2][0], 20.0)
        self.assertAlmostEqual(result.velocity_W_mps[2][1], 40.0)

    def test_invalid_immediate_predecessor_does_not_search_back(self) -> None:
        result = derive_adjacent_backward_velocity(
            timestamps_s=(0.0, 0.1, 0.2),
            positions_W_m=(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
            ),
            state_valid=(True, False, True),
        )

        # No fallback from t=2 to t=0.
        self.assertFalse(result.velocity_valid[2])
        self.assertEqual(result.velocity_W_mps[2], ZERO_VEC3)

    def test_invalid_current_state_has_invalid_velocity(self) -> None:
        result = derive_adjacent_backward_velocity(
            timestamps_s=(0.0, 0.1),
            positions_W_m=(
                (0.0, 0.0, 0.0),
                ZERO_VEC3,
            ),
            state_valid=(True, False),
        )

        self.assertFalse(result.velocity_valid[1])
        self.assertEqual(result.velocity_W_mps[1], ZERO_VEC3)

    def test_zero_dt_is_invalid_not_divided(self) -> None:
        result = derive_adjacent_backward_velocity(
            timestamps_s=(1.0, 1.0),
            positions_W_m=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            state_valid=(True, True),
        )

        self.assertFalse(result.velocity_valid[1])
        self.assertEqual(result.velocity_W_mps[1], ZERO_VEC3)

    def test_negative_dt_is_invalid(self) -> None:
        result = derive_adjacent_backward_velocity(
            timestamps_s=(1.0, 0.9),
            positions_W_m=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            state_valid=(True, True),
        )

        self.assertFalse(result.velocity_valid[1])


if __name__ == "__main__":
    unittest.main()