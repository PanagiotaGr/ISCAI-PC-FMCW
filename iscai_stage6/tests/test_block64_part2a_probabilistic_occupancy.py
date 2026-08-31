from __future__ import annotations

import inspect
import unittest

import numpy as np

from iscai_stage6.adb.geometry import (
    Box3D,
)

from iscai_stage6.adb.probabilistic_full_box import (
    CalibratedGaussianFullBoxPrediction,
    build_stochastic_future_full_boxes,
)

from iscai_stage6.adb.probabilistic_occupancy import (
    OccupancyGrid,
    convex_hull_theta_range,
    estimate_actor_occupancy_probability,
    rasterize_projected_full_box,
)


def covariance(
    variance: float,
):
    matrix = np.diag(
        [
            variance,
            variance,
            variance,
        ]
    )

    return tuple(
        tuple(
            tuple(
                float(value)
                for value in row
            )
            for row in matrix
        )
        for _ in range(
            4
        )
    )


def prediction(
    *,
    variance=0.0,
):
    return CalibratedGaussianFullBoxPrediction(
        prediction_id="synthetic",
        latest_position_H0_m=(
            20.0,
            0.0,
            1.0,
        ),
        mean_displacement_H0_m=(
            (1.0, 0.0, 0.0),
            (3.0, 0.1, 0.0),
            (5.0, 0.2, 0.0),
            (10.0, 0.4, 0.0),
        ),
        calibrated_predictive_covariance_H0_m2=(
            covariance(
                float(
                    variance
                )
            )
        ),
    )


def actor_box():
    class Actor:
        pass

    actor = Actor()

    actor.box = Box3D(
        center_xyz=(
            20.0,
            0.0,
            1.0,
        ),
        length_m=4.5,
        width_m=2.0,
        height_m=1.6,
        yaw_rad=0.0,
    )

    actor.future_state_used = False
    actor.tracks_to_predict_used = False
    actor.objects_of_interest_used = False

    return actor


def grid():
    return OccupancyGrid(
        theta_centers_rad=tuple(
            np.linspace(
                -0.20,
                0.20,
                81,
                dtype=np.float64,
            )
        ),
        range_centers_m=tuple(
            np.linspace(
                15.0,
                35.0,
                81,
                dtype=np.float64,
            )
        ),
    )


class TestBlock64Part2AOccupancy(
    unittest.TestCase
):
    def test_01_grid_is_parameterized(
        self,
    ):
        item = grid()

        self.assertEqual(
            item.shape,
            (
                81,
                81,
            ),
        )

    def test_02_grid_requires_increasing_theta(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            OccupancyGrid(
                theta_centers_rad=(
                    0.0,
                    -0.1,
                ),
                range_centers_m=(
                    1.0,
                    2.0,
                ),
            )

    def test_03_grid_requires_nonnegative_range(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            OccupancyGrid(
                theta_centers_rad=(
                    -0.1,
                    0.1,
                ),
                range_centers_m=(
                    -1.0,
                    1.0,
                ),
            )

    def test_04_convex_hull_uses_polygon_not_rectangle(
        self,
    ):
        points = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )

        hull = convex_hull_theta_range(
            points
        )

        self.assertEqual(
            hull.shape,
            (
                3,
                2,
            ),
        )

    def test_05_one_box_raster_is_boolean(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(),
                actor_box(),
                sample_count=1,
                seed=11,
            )
        )

        mask = rasterize_projected_full_box(
            forecast.future[
                0
            ][
                0
            ],
            grid(),
        )

        self.assertEqual(
            mask.dtype,
            np.bool_,
        )

        self.assertEqual(
            mask.shape,
            (
                81,
                81,
            ),
        )

        self.assertGreater(
            int(
                np.count_nonzero(
                    mask
                )
            ),
            1,
        )

    def test_06_zero_uncertainty_is_binary_probability(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.0
                ),
                actor_box(),
                sample_count=8,
                seed=11,
            )
        )

        result = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
        )

        unique = set(
            np.unique(
                result.occupancy_probability
            ).tolist()
        )

        self.assertTrue(
            unique.issubset(
                {
                    0.0,
                    1.0,
                }
            )
        )

    def test_07_probability_equals_counts_over_n(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.05
                ),
                actor_box(),
                sample_count=17,
                seed=11,
            )
        )

        result = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
        )

        np.testing.assert_array_equal(
            result.occupancy_probability,
            (
                result
                .occupancy_counts
                .astype(
                    np.float64
                )
                /
                17.0
            ),
        )

    def test_08_probability_bounds(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.05
                ),
                actor_box(),
                sample_count=19,
                seed=11,
            )
        )

        probability = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
            .occupancy_probability
        )

        self.assertGreaterEqual(
            float(
                np.min(
                    probability
                )
            ),
            0.0,
        )

        self.assertLessEqual(
            float(
                np.max(
                    probability
                )
            ),
            1.0,
        )

    def test_09_same_seed_exact_occupancy_repeat(
        self,
    ):
        first_forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.05
                ),
                actor_box(),
                sample_count=23,
                seed=11,
            )
        )

        second_forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.05
                ),
                actor_box(),
                sample_count=23,
                seed=11,
            )
        )

        first = (
            estimate_actor_occupancy_probability(
                first_forecast,
                grid(),
            )
        )

        second = (
            estimate_actor_occupancy_probability(
                second_forecast,
                grid(),
            )
        )

        np.testing.assert_array_equal(
            first.occupancy_counts,
            second.occupancy_counts,
        )

        np.testing.assert_array_equal(
            first.occupancy_probability,
            second.occupancy_probability,
        )

    def test_10_uncertainty_produces_nonbinary_cells(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.08
                ),
                actor_box(),
                sample_count=64,
                seed=11,
            )
        )

        probability = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
            .occupancy_probability
        )

        fractional = (
            probability
            [
                (
                    probability
                    >
                    0.0
                )
                &
                (
                    probability
                    <
                    1.0
                )
            ]
        )

        self.assertGreater(
            fractional.size,
            0,
        )

    def test_11_output_shape_is_horizon_theta_range(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.04
                ),
                actor_box(),
                sample_count=5,
                seed=11,
            )
        )

        result = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
        )

        self.assertEqual(
            result.shape,
            (
                4,
                81,
                81,
            ),
        )

    def test_12_no_gamma_threshold_argument(
        self,
    ):
        signature = inspect.signature(
            estimate_actor_occupancy_probability
        )

        names = set(
            signature.parameters.keys()
        )

        self.assertNotIn(
            "gamma",
            names,
        )

        self.assertNotIn(
            "threshold",
            names,
        )

    def test_13_no_class_policy_argument(
        self,
    ):
        signature = inspect.signature(
            estimate_actor_occupancy_probability
        )

        names = set(
            signature.parameters.keys()
        )

        self.assertFalse(
            any(
                "class"
                in name.lower()
                or
                "margin"
                in name.lower()
                or
                "floor"
                in name.lower()
                for name in names
            )
        )

    def test_14_no_centroid_api_in_occupancy_kernel(
        self,
    ):
        source = inspect.getsource(
            estimate_actor_occupancy_probability
        )

        self.assertNotIn(
            "centroid",
            source.lower(),
        )

    def test_15_all_eight_corners_required(
        self,
    ):
        source = inspect.getsource(
            rasterize_projected_full_box
        )

        self.assertIn(
            "_projected_theta_range_points",
            source,
        )

        forecast = (
            build_stochastic_future_full_boxes(
                prediction(),
                actor_box(),
                sample_count=1,
                seed=11,
            )
        )

        self.assertEqual(
            forecast.future[
                0
            ][
                0
            ]
            .projection
            .corners
            .xyz
            .shape,
            (
                8,
                3,
            ),
        )


if __name__ == "__main__":
    unittest.main()
