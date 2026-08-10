import math
import unittest

from iscai_stage1.contracts.stage1a import (
    HeadlampSurrogateConfig,
    ReceiverGeometryConfig,
    ZERO_MAT3,
    ZERO_VEC3,
)
from iscai_stage1.geometry.frames import (
    SdcStateW,
    build_anchor_frames,
    build_dynamic_headlamp_frame,
    horizontal_bearing_rad,
)
from iscai_stage1.geometry.receiver import (
    ActorAnchorStateW,
    receiver_geometry_in_H0,
)


class TestFrames(unittest.TestCase):
    def assertVecAlmostEqual(
        self,
        actual: tuple[float, float, float],
        expected: tuple[float, float, float],
        places: int = 10,
    ) -> None:
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(
                actual_value,
                expected_value,
                places=places,
            )

    def assertTransformAlmostEqual(self, left, right, places: int = 10) -> None:
        for row in range(3):
            for col in range(3):
                self.assertAlmostEqual(
                    left.rotation[row][col],
                    right.rotation[row][col],
                    places=places,
                )
        self.assertVecAlmostEqual(
            left.translation,
            right.translation,
            places=places,
        )

    def setUp(self) -> None:
        self.anchor = SdcStateW(
            center_w_m=(10.0, 20.0, 1.0),
            heading_rad=math.pi / 2.0,
            length_m=4.8,
            valid=True,
        )
        self.config = HeadlampSurrogateConfig()
        self.frames = build_anchor_frames(self.anchor, self.config)

    def test_E0_origin_is_anchor_sdc_center(self) -> None:
        point_E0 = self.frames.T_E0_from_W.apply_point(
            self.anchor.center_w_m
        )
        self.assertVecAlmostEqual(point_E0, (0.0, 0.0, 0.0))

    def test_E0_positive_x_is_forward(self) -> None:
        # heading = +90 deg, so world +y is SDC forward.
        point_one_metre_forward_W = (10.0, 21.0, 1.0)

        point_E0 = self.frames.T_E0_from_W.apply_point(
            point_one_metre_forward_W
        )

        self.assertVecAlmostEqual(point_E0, (1.0, 0.0, 0.0))

    def test_forward_inverse_round_trip(self) -> None:
        point_W = (16.0, 23.0, 2.5)

        point_E0 = self.frames.T_E0_from_W.apply_point(point_W)
        reconstructed_W = self.frames.T_W_from_E0.apply_point(point_E0)

        self.assertVecAlmostEqual(reconstructed_W, point_W)

        point_H0 = self.frames.T_H0_from_W.apply_point(point_W)
        reconstructed_W_2 = self.frames.T_W_from_H0.apply_point(point_H0)

        self.assertVecAlmostEqual(reconstructed_W_2, point_W)

    def test_headlamp_baseline_is_front_face_midpoint_surrogate(self) -> None:
        headlamp_origin_W = self.frames.T_W_from_H0.apply_point(
            (0.0, 0.0, 0.0)
        )
        headlamp_origin_E0 = self.frames.T_E0_from_W.apply_point(
            headlamp_origin_W
        )

        self.assertVecAlmostEqual(
            headlamp_origin_E0,
            (2.4, 0.0, 0.0),
        )

    def test_Ht_at_anchor_equals_H0(self) -> None:
        dynamic = build_dynamic_headlamp_frame(
            self.anchor,
            self.config,
        )

        self.assertTransformAlmostEqual(
            dynamic.T_W_from_Ht,
            self.frames.T_W_from_H0,
        )
        self.assertTransformAlmostEqual(
            dynamic.T_Ht_from_W,
            self.frames.T_H0_from_W,
        )

    def test_horizontal_bearing_sign(self) -> None:
        self.assertAlmostEqual(
            horizontal_bearing_rad((10.0, 0.0, 0.0)),
            0.0,
        )
        self.assertGreater(
            horizontal_bearing_rad((10.0, 1.0, 0.0)),
            0.0,
        )
        self.assertLess(
            horizontal_bearing_rad((10.0, -1.0, 0.0)),
            0.0,
        )

    def test_receiver_centroid_baseline(self) -> None:
        actor = ActorAnchorStateW(
            center_w_m=(10.0, 30.0, 1.0),
            heading_rad=math.pi / 2.0,
            valid=True,
        )

        result = receiver_geometry_in_H0(
            actor,
            self.frames.T_H0_from_W,
            ReceiverGeometryConfig(),
        )

        expected_actor_center_H0 = (
            self.frames.T_H0_from_W.apply_point(actor.center_w_m)
        )

        self.assertEqual(result.receiver_offset_mean_H0, ZERO_VEC3)
        self.assertEqual(
            result.receiver_offset_covariance_H0,
            ZERO_MAT3,
        )
        self.assertVecAlmostEqual(
            result.receiver_point_mean_H0,
            expected_actor_center_H0,
        )
        self.assertTrue(result.receiver_geometry_valid)
        self.assertEqual(
            result.receiver_geometry_mode,
            "centroid_baseline",
        )


if __name__ == "__main__":
    unittest.main()