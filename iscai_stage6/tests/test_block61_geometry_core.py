from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.geometry import (
    Box3D,
    FractionalBoxRegion,
    box_corners_headlamp,
    centroid_only_projection,
    project_box_to_headlamp,
    project_points_to_headlamp,
    project_region_to_headlamp,
    region_corners_headlamp,
    validate_controller_provenance,
    wrap_angle,
)

from iscai_stage6.adb.grid import (
    IlluminationGridSpec,
    rectangular_extent_cell_indices,
)


class TestBlock61GeometryCore(
    unittest.TestCase
):

    def test_headlamp_spherical_mapping(self):

        points = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        projected = (
            project_points_to_headlamp(
                points
            )
        )

        self.assertAlmostEqual(
            projected.theta_rad[0],
            0.0,
        )

        self.assertAlmostEqual(
            projected.theta_rad[1],
            math.pi / 4.0,
        )

        self.assertAlmostEqual(
            projected.phi_rad[2],
            math.pi / 4.0,
        )

        self.assertAlmostEqual(
            projected.range_3d_m[2],
            math.sqrt(2.0),
        )

    def test_box_has_exactly_eight_corners(self):

        box = Box3D(
            center_xyz=(
                20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        corners = (
            box_corners_headlamp(
                box
            )
        )

        self.assertEqual(
            corners.shape,
            (8, 3),
        )

        self.assertEqual(
            np.unique(
                corners,
                axis=0,
            ).shape[0],
            8,
        )

    def test_box_dimensions_are_preserved(self):

        box = Box3D(
            center_xyz=(
                15.0,
                2.0,
                1.5,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
            yaw_rad=0.0,
        )

        corners = (
            box_corners_headlamp(
                box
            )
        )

        extent = (
            np.max(
                corners,
                axis=0,
            )
            -
            np.min(
                corners,
                axis=0,
            )
        )

        np.testing.assert_allclose(
            extent,
            np.asarray(
                [
                    4.0,
                    2.0,
                    1.6,
                ]
            ),
            atol=1.0e-12,
        )

    def test_yaw_rotates_length_width_axes(self):

        box = Box3D(
            center_xyz=(
                20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.0,
            yaw_rad=math.pi / 2.0,
        )

        corners = (
            box_corners_headlamp(
                box
            )
        )

        extent_xy = (
            np.max(
                corners[:, :2],
                axis=0,
            )
            -
            np.min(
                corners[:, :2],
                axis=0,
            )
        )

        np.testing.assert_allclose(
            extent_xy,
            np.asarray(
                [
                    2.0,
                    4.0,
                ]
            ),
            atol=1.0e-12,
        )

    def test_full_box_has_nonzero_angular_extent(self):

        box = Box3D(
            center_xyz=(
                20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="current_causal",
                orientation_source="current_causal",
            )
        )

        self.assertTrue(
            projection.uses_full_box
        )

        self.assertGreater(
            projection.theta_span_rad,
            0.0,
        )

        self.assertLess(
            projection.ground_range_min_m,
            projection.ground_range_max_m,
        )

        self.assertLess(
            projection.phi_min_rad,
            projection.phi_max_rad,
        )

    def test_centroid_does_not_encode_box_extent(self):

        box = Box3D(
            center_xyz=(
                25.0,
                1.0,
                1.0,
            ),
            length_m=5.0,
            width_m=2.0,
            height_m=2.0,
            yaw_rad=0.2,
        )

        centroid = (
            centroid_only_projection(
                box
            )
        )

        full = (
            project_box_to_headlamp(
                box,
                state_source="current_causal",
                orientation_source="current_causal",
            )
        )

        self.assertEqual(
            centroid.theta_rad.shape,
            (1,),
        )

        self.assertGreater(
            full.theta_span_rad,
            0.0,
        )

        self.assertGreater(
            full.range_3d_max_m
            -
            full.range_3d_min_m,
            0.0,
        )

    def test_relevant_subregion_supported(self):

        box = Box3D(
            center_xyz=(
                18.0,
                -1.0,
                1.2,
            ),
            length_m=4.5,
            width_m=2.0,
            height_m=1.8,
            yaw_rad=-0.1,
        )

        region = FractionalBoxRegion(
            x_bounds=(
                0.0,
                0.5,
            ),
            y_bounds=(
                -0.4,
                0.4,
            ),
            z_bounds=(
                0.0,
                0.5,
            ),
        )

        points = (
            region_corners_headlamp(
                box,
                region,
            )
        )

        projected = (
            project_region_to_headlamp(
                box,
                region,
            )
        )

        self.assertEqual(
            points.shape,
            (8, 3),
        )

        self.assertEqual(
            projected.points.xyz.shape,
            (8, 3),
        )

        self.assertGreater(
            projected.theta_span_rad,
            0.0,
        )

    def test_no_semantic_windshield_constants_are_needed(self):

        with self.assertRaises(
            ValueError
        ):
            FractionalBoxRegion(
                x_bounds=(
                    -0.6,
                    0.2,
                ),
                y_bounds=(
                    -0.4,
                    0.4,
                ),
                z_bounds=(
                    0.0,
                    0.5,
                ),
            )

    def test_future_gt_rejected_on_controller_path(self):

        box = Box3D(
            center_xyz=(
                10.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        with self.assertRaises(
            ValueError
        ):
            project_box_to_headlamp(
                box,
                state_source="future_gt",
                orientation_source="oracle_evaluator_only",
                controller_path=True,
            )

    def test_oracle_allowed_only_off_controller_path(self):

        validate_controller_provenance(
            state_source="oracle_future",
            orientation_source="oracle_evaluator_only",
            controller_path=False,
        )

        box = Box3D(
            center_xyz=(
                10.0,
                1.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="oracle_future",
                orientation_source="oracle_evaluator_only",
                controller_path=False,
            )
        )

        self.assertFalse(
            projection.controller_path
        )

    def test_predicted_sample_is_controller_safe(self):

        box = Box3D(
            center_xyz=(
                30.0,
                2.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.1,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="predicted_sample",
                orientation_source="predicted_tangent",
                controller_path=True,
            )
        )

        self.assertEqual(
            projection.state_source,
            "predicted_sample",
        )

    def test_angle_wrap(self):

        self.assertAlmostEqual(
            wrap_angle(
                math.pi
            ),
            -math.pi,
        )

        self.assertAlmostEqual(
            wrap_angle(
                3.0 * math.pi
            ),
            -math.pi,
        )

    def test_circular_box_interval_near_pi_boundary(self):

        box = Box3D(
            center_xyz=(
                -20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.0,
            yaw_rad=0.0,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="current_causal",
                orientation_source="current_causal",
            )
        )

        self.assertLess(
            projection.theta_span_rad,
            math.pi / 2.0,
        )

    def test_invalid_box_dimensions_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            Box3D(
                center_xyz=(
                    10.0,
                    0.0,
                    1.0,
                ),
                length_m=0.0,
                width_m=2.0,
                height_m=1.0,
                yaw_rad=0.0,
            )

    def test_grid_requires_explicit_numeric_values(self):

        grid = IlluminationGridSpec(
            theta_min_rad=-0.5,
            theta_max_rad=0.5,
            range_min_m=0.0,
            range_max_m=100.0,
            n_theta=20,
            n_range=50,
        )

        self.assertEqual(
            grid.shape,
            (20, 50),
        )

        self.assertAlmostEqual(
            grid.theta_step_rad,
            0.05,
        )

        self.assertAlmostEqual(
            grid.range_step_m,
            2.0,
        )

    def test_grid_extent_uses_more_than_centroid_cell(self):

        grid = IlluminationGridSpec(
            theta_min_rad=-0.5,
            theta_max_rad=0.5,
            range_min_m=0.0,
            range_max_m=100.0,
            n_theta=100,
            n_range=100,
        )

        box = Box3D(
            center_xyz=(
                20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=4.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="current_causal",
                orientation_source="current_causal",
            )
        )

        theta_indices, range_indices = (
            rectangular_extent_cell_indices(
                grid,
                theta_center_rad=(
                    projection.theta_center_rad
                ),
                theta_span_rad=(
                    projection.theta_span_rad
                ),
                range_min_m=(
                    projection.ground_range_min_m
                ),
                range_max_m=(
                    projection.ground_range_max_m
                ),
            )
        )

        self.assertGreater(
            theta_indices.size,
            1,
        )

        self.assertGreater(
            range_indices.size,
            1,
        )

    def test_grid_is_not_communication_codebook(self):

        grid = IlluminationGridSpec(
            theta_min_rad=-0.4,
            theta_max_rad=0.4,
            range_min_m=0.0,
            range_max_m=80.0,
            n_theta=32,
            n_range=40,
        )

        self.assertEqual(
            grid.shape,
            (32, 40),
        )

        # This object has theta/range cells only.
        self.assertFalse(
            hasattr(
                grid,
                "beam_id",
            )
        )

        self.assertFalse(
            hasattr(
                grid,
                "codebook",
            )
        )


if __name__ == "__main__":
    unittest.main()
