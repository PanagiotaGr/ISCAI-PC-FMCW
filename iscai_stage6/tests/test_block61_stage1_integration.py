from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.geometry import (
    Box3D,
)

from iscai_stage6.adb.womd_geometry import (
    CausalADBActorBox,
    SUPPORTED_ADB_CLASSES,
    causal_actor_boxes_sha256,
    world_heading_to_headlamp_yaw,
)


class _Transform:
    def __init__(
        self,
        rotation,
    ):
        self.rotation = rotation


class TestBlock61Stage1Integration(
    unittest.TestCase
):

    def test_required_adb_classes(self):

        self.assertEqual(
            SUPPORTED_ADB_CLASSES,
            frozenset(
                {
                    "TYPE_VEHICLE",
                    "TYPE_PEDESTRIAN",
                    "TYPE_CYCLIST",
                }
            ),
        )

    def test_identity_heading_transform(self):

        transform = _Transform(
            np.eye(3)
        )

        value = (
            world_heading_to_headlamp_yaw(
                heading_W_rad=0.4,
                T_H0_from_W=transform,
            )
        )

        self.assertAlmostEqual(
            value,
            0.4,
        )

    def test_rotated_heading_transform(self):

        angle = 0.3

        c = math.cos(-angle)
        s = math.sin(-angle)

        rotation = np.asarray(
            [
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        transform = _Transform(
            rotation
        )

        value = (
            world_heading_to_headlamp_yaw(
                heading_W_rad=angle,
                T_H0_from_W=transform,
            )
        )

        self.assertAlmostEqual(
            value,
            0.0,
            places=12,
        )

    def test_stable_actor_hash(self):

        actor = CausalADBActorBox(
            scenario_id="demo",
            track_index=2,
            track_id="abc",
            object_type="TYPE_VEHICLE",
            box=Box3D(
                center_xyz=(
                    10.0,
                    1.0,
                    0.5,
                ),
                length_m=4.0,
                width_m=2.0,
                height_m=1.5,
                yaw_rad=0.1,
            ),
            stage1_anchor_center_H0_m=(
                10.0,
                1.0,
                0.5,
            ),
            stage1_anchor_bearing_H0_rad=(
                math.atan2(
                    1.0,
                    10.0,
                )
            ),
            center_consistency_error_m=0.0,
            source_time_index=10,
        )

        first = causal_actor_boxes_sha256(
            (actor,)
        )

        second = causal_actor_boxes_sha256(
            (actor,)
        )

        self.assertEqual(
            first,
            second,
        )

    def test_provenance_flags_default_false(self):

        actor = CausalADBActorBox(
            scenario_id="demo",
            track_index=1,
            track_id="1",
            object_type="TYPE_PEDESTRIAN",
            box=Box3D(
                center_xyz=(
                    5.0,
                    0.0,
                    0.8,
                ),
                length_m=0.8,
                width_m=0.6,
                height_m=1.7,
                yaw_rad=0.0,
            ),
            stage1_anchor_center_H0_m=(
                5.0,
                0.0,
                0.8,
            ),
            stage1_anchor_bearing_H0_rad=0.0,
            center_consistency_error_m=0.0,
            source_time_index=10,
        )

        self.assertFalse(
            actor.tracks_to_predict_used
        )

        self.assertFalse(
            actor.objects_of_interest_used
        )

        self.assertFalse(
            actor.future_state_used
        )


if __name__ == "__main__":
    unittest.main()
