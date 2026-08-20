import math
import unittest

from iscai_stage1.geometry.rigid import (
    RigidTransform,
)
from iscai_stage2.observations.ideal import (
    adjacent_headlamp_velocity_W,
    headlamp_origin_W,
    ideal_causal_observable,
)


IDENTITY = RigidTransform(
    rotation=(
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    translation=(0.0, 0.0, 0.0),
)


class TestIdealObservations(unittest.TestCase):

    def test_range(self):
        obs = ideal_causal_observable(
            time_index=1,
            timestamp_s=0.1,
            actor_position_W_m=(
                3.0, 4.0, 0.0
            ),
            actor_position_valid=True,
            actor_velocity_W_mps=(
                0.0, 0.0, 0.0
            ),
            actor_velocity_valid=True,
            headlamp_velocity_W_mps=(
                0.0, 0.0, 0.0
            ),
            T_Ht_from_W=IDENTITY,
        )

        self.assertTrue(
            obs.geometry_valid
        )

        self.assertAlmostEqual(
            obs.range_m,
            5.0,
        )

    def test_positive_azimuth_is_left(self):
        obs = ideal_causal_observable(
            time_index=1,
            timestamp_s=0.1,
            actor_position_W_m=(
                10.0, 10.0, 0.0
            ),
            actor_position_valid=True,
            actor_velocity_W_mps=(
                0.0, 0.0, 0.0
            ),
            actor_velocity_valid=False,
            headlamp_velocity_W_mps=None,
            T_Ht_from_W=IDENTITY,
        )

        self.assertAlmostEqual(
            obs.azimuth_rad,
            math.pi / 4.0,
        )

    def test_positive_elevation_is_up(self):
        obs = ideal_causal_observable(
            time_index=1,
            timestamp_s=0.1,
            actor_position_W_m=(
                1.0, 0.0, 1.0
            ),
            actor_position_valid=True,
            actor_velocity_W_mps=(
                0.0, 0.0, 0.0
            ),
            actor_velocity_valid=False,
            headlamp_velocity_W_mps=None,
            T_Ht_from_W=IDENTITY,
        )

        self.assertAlmostEqual(
            obs.elevation_rad,
            math.pi / 4.0,
        )

    def test_approaching_target_has_negative_vr(self):
        obs = ideal_causal_observable(
            time_index=1,
            timestamp_s=0.1,
            actor_position_W_m=(
                10.0, 0.0, 0.0
            ),
            actor_position_valid=True,
            actor_velocity_W_mps=(
                -2.0, 0.0, 0.0
            ),
            actor_velocity_valid=True,
            headlamp_velocity_W_mps=(
                0.0, 0.0, 0.0
            ),
            T_Ht_from_W=IDENTITY,
        )

        self.assertTrue(
            obs.radial_velocity_valid
        )

        self.assertAlmostEqual(
            obs.radial_velocity_mps,
            -2.0,
        )

    def test_equal_actor_and_headlamp_velocity(self):
        obs = ideal_causal_observable(
            time_index=1,
            timestamp_s=0.1,
            actor_position_W_m=(
                10.0, 0.0, 0.0
            ),
            actor_position_valid=True,
            actor_velocity_W_mps=(
                5.0, 1.0, 0.0
            ),
            actor_velocity_valid=True,
            headlamp_velocity_W_mps=(
                5.0, 1.0, 0.0
            ),
            T_Ht_from_W=IDENTITY,
        )

        self.assertAlmostEqual(
            obs.radial_velocity_mps,
            0.0,
        )

    def test_invalid_actor_velocity_does_not_kill_geometry(self):
        obs = ideal_causal_observable(
            time_index=1,
            timestamp_s=0.1,
            actor_position_W_m=(
                20.0, 2.0, 1.0
            ),
            actor_position_valid=True,
            actor_velocity_W_mps=(
                0.0, 0.0, 0.0
            ),
            actor_velocity_valid=False,
            headlamp_velocity_W_mps=(
                1.0, 0.0, 0.0
            ),
            T_Ht_from_W=IDENTITY,
        )

        self.assertTrue(
            obs.geometry_valid
        )

        self.assertFalse(
            obs.radial_velocity_valid
        )

        self.assertIsNone(
            obs.radial_velocity_mps
        )

    def test_headlamp_velocity_is_strict_adjacent(self):
        velocity = (
            adjacent_headlamp_velocity_W(
                previous_origin_W_m=(
                    1.0, 2.0, 3.0
                ),
                current_origin_W_m=(
                    2.0, 4.0, 3.0
                ),
                previous_timestamp_s=1.0,
                current_timestamp_s=1.5,
            )
        )

        self.assertEqual(
            velocity,
            (2.0, 4.0, 0.0),
        )

        invalid = (
            adjacent_headlamp_velocity_W(
                previous_origin_W_m=(
                    1.0, 2.0, 3.0
                ),
                current_origin_W_m=(
                    2.0, 4.0, 3.0
                ),
                previous_timestamp_s=1.0,
                current_timestamp_s=1.0,
            )
        )

        self.assertIsNone(invalid)

    def test_headlamp_origin_from_transform(self):
        T = RigidTransform(
            rotation=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            translation=(-5.0, 2.0, -1.0),
        )

        self.assertEqual(
            headlamp_origin_W(T),
            (5.0, -2.0, 1.0),
        )


if __name__ == "__main__":
    unittest.main()
