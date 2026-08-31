from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from iscai_stage5.angular_monte_carlo import (
    resolve_samplewise_headings,
)

from iscai_stage6.adb.deterministic_association import (
    PredictorAnchor,
    reciprocal_unique_nearest_anchor_association,
)

from iscai_stage6.adb.deterministic_predictive import (
    HORIZONS_S,
    DeterministicSharedMeanPrediction,
    build_deterministic_future_full_boxes,
    build_deterministic_predictive_plan,
)

from iscai_stage6.adb.geometry import (
    Box3D,
)

from iscai_stage6.adb.womd_geometry import (
    CausalADBActorBox,
)


def actor_box(
    *,
    anchor,
    box_center=None,
    yaw=0.0,
    object_type="TYPE_VEHICLE",
    future_state_used=False,
    tracks_to_predict_used=False,
    objects_of_interest_used=False,
):

    if box_center is None:
        box_center = anchor

    return CausalADBActorBox(
        scenario_id="synthetic",

        track_index=0,

        track_id="womd-actor-id-not-used-for-association",

        object_type=object_type,

        box=Box3D(
            center_xyz=tuple(
                float(value)
                for value
                in box_center
            ),

            length_m=4.5,
            width_m=1.8,
            height_m=1.6,
            yaw_rad=float(
                yaw
            ),
        ),

        stage1_anchor_center_H0_m=(
            None
            if anchor is None
            else tuple(
                float(value)
                for value
                in anchor
            )
        ),

        stage1_anchor_bearing_H0_rad=None,

        center_consistency_error_m=0.0,

        source_time_index=10,

        tracks_to_predict_used=(
            tracks_to_predict_used
        ),

        objects_of_interest_used=(
            objects_of_interest_used
        ),

        future_state_used=(
            future_state_used
        ),
    )


class TestBlock63Association(
    unittest.TestCase
):

    def test_single_pair_matches_without_id_equality(self):

        predictors = (
            PredictorAnchor(
                prediction_id="s3trk-internal-17",
                latest_position_H0_m=(
                    10.0,
                    2.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    10.1,
                    2.0,
                    0.0,
                )
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            len(
                result.matches
            ),
            1,
        )

        self.assertEqual(
            result.matches[
                0
            ].prediction_id,
            "s3trk-internal-17",
        )

    def test_reciprocal_two_by_two(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
            PredictorAnchor(
                "p1",
                (
                    20.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    10.2,
                    0.0,
                    0.0,
                )
            ),
            actor_box(
                anchor=(
                    19.8,
                    0.0,
                    0.0,
                )
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            [
                (
                    match.predictor_index,
                    match.box_index,
                )
                for match
                in result.matches
            ],
            [
                (
                    0,
                    0,
                ),
                (
                    1,
                    1,
                ),
            ],
        )

    def test_exact_predictor_tie_is_unmatched(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    0.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    1.0,
                    0.0,
                    0.0,
                )
            ),
            actor_box(
                anchor=(
                    -1.0,
                    0.0,
                    0.0,
                )
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            result.matches,
            (),
        )

        self.assertEqual(
            result.unmatched_box_indices,
            (
                0,
                1,
            ),
        )

    def test_exact_box_tie_is_unmatched(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    -1.0,
                    0.0,
                    0.0,
                ),
            ),
            PredictorAnchor(
                "p1",
                (
                    1.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    0.0,
                    0.0,
                    0.0,
                )
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            result.matches,
            (),
        )

        self.assertEqual(
            result.unmatched_box_indices,
            (
                0,
            ),
        )

    def test_missing_box_anchor_becomes_fallback(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=None,
                box_center=(
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            result.matches,
            (),
        )

        self.assertEqual(
            result.unmatched_box_indices,
            (
                0,
            ),
        )

    def test_no_class_gate(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    10.0,
                    0.0,
                    0.0,
                ),
                object_type=(
                    "TYPE_PEDESTRIAN"
                ),
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            len(
                result.matches
            ),
            1,
        )

    def test_duplicate_prediction_ids_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            reciprocal_unique_nearest_anchor_association(
                (
                    PredictorAnchor(
                        "same",
                        (
                            1.0,
                            0.0,
                            0.0,
                        ),
                    ),
                    PredictorAnchor(
                        "same",
                        (
                            2.0,
                            0.0,
                            0.0,
                        ),
                    ),
                ),
                (),
            )


class TestBlock63DeterministicFutureBox(
    unittest.TestCase
):

    def prediction(
        self,
        *,
        anchor=(
            10.0,
            0.0,
            0.0,
        ),
        displacement=None,
    ):

        if displacement is None:
            displacement = (
                (
                    1.0,
                    0.0,
                    0.0,
                ),
                (
                    2.0,
                    0.0,
                    0.0,
                ),
                (
                    3.0,
                    0.0,
                    0.0,
                ),
                (
                    4.0,
                    0.0,
                    0.0,
                ),
            )

        return DeterministicSharedMeanPrediction(
            prediction_id="predictor-internal",

            latest_position_H0_m=anchor,

            mean_displacement_H0_m=(
                displacement
            ),
        )

    def test_horizons_are_frozen(self):

        self.assertEqual(
            HORIZONS_S,
            (
                0.1,
                0.3,
                0.5,
                1.0,
            ),
        )

    def test_future_center_is_predictor_anchor_plus_mean(self):

        prediction = self.prediction(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        centers = (
            prediction.future_centers_H0_m()
        )

        self.assertEqual(
            centers[
                0
            ],
            (
                11.0,
                0.0,
                0.0,
            ),
        )

        self.assertEqual(
            centers[
                3
            ],
            (
                14.0,
                0.0,
                0.0,
            ),
        )

    def test_prediction_is_not_recentered_to_matched_box(self):

        prediction = self.prediction(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        box = actor_box(
            anchor=(
                10.1,
                0.0,
                0.0,
            ),

            # Intentionally very different from the predictor
            # anchor to prove that the future state is NOT
            # recentered to annotation geometry.
            box_center=(
                100.0,
                50.0,
                0.0,
            ),
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        self.assertEqual(
            future[
                0
            ].center_H0_m,
            (
                11.0,
                0.0,
                0.0,
            ),
        )

        self.assertNotEqual(
            future[
                0
            ].center_H0_m,
            (
                101.0,
                50.0,
                0.0,
            ),
        )

    def test_current_dimensions_are_held_across_horizons(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        for item in future:

            self.assertEqual(
                item.box.length_m,
                box.box.length_m,
            )

            self.assertEqual(
                item.box.width_m,
                box.box.width_m,
            )

            self.assertEqual(
                item.box.height_m,
                box.box.height_m,
            )

    def test_every_horizon_projects_eight_physical_corners(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        self.assertEqual(
            len(
                future
            ),
            4,
        )

        for item in future:

            self.assertEqual(
                item.projection.corners.xyz.shape,
                (
                    8,
                    3,
                ),
            )

            self.assertGreater(
                item.projection.theta_span_rad,
                0.0,
            )

            self.assertGreater(
                item.projection.ground_range_max_m,
                item.projection.ground_range_min_m,
            )

    def test_heading_matches_frozen_stage5_rule(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            yaw=0.25,
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        centers = np.asarray(
            prediction.future_centers_H0_m(),
            dtype=np.float64,
        )

        expected, expected_fallback = (
            resolve_samplewise_headings(
                trajectory_samples_h0_m=(
                    centers[
                        None,
                        :,
                        :,
                    ]
                ),

                current_position_h0_m=(
                    prediction.latest_position_H0_m
                ),

                current_heading_h0_rad=(
                    box.box.yaw_rad
                ),
            )
        )

        np.testing.assert_allclose(
            np.asarray(
                [
                    item.yaw_H0_rad
                    for item
                    in future
                ]
            ),
            expected[
                0
            ],
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_array_equal(
            np.asarray(
                [
                    item.heading_fallback_used
                    for item
                    in future
                ]
            ),
            expected_fallback[
                0
            ],
        )

    def test_low_speed_heading_carries_current_causal_yaw(self):

        current_yaw = 0.73

        prediction = self.prediction(
            displacement=(
                (
                    0.0,
                    0.0,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    0.0,
                ),
            )
        )

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            yaw=current_yaw,
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        for item in future:

            self.assertTrue(
                item.heading_fallback_used
            )

            self.assertAlmostEqual(
                item.yaw_H0_rad,
                current_yaw,
            )

    def test_future_state_provenance_is_rejected(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            future_state_used=True,
        )

        with self.assertRaises(
            RuntimeError
        ):
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )

    def test_tracks_to_predict_provenance_is_rejected(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            tracks_to_predict_used=True,
        )

        with self.assertRaises(
            RuntimeError
        ):
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )

    def test_objects_of_interest_provenance_is_rejected(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            objects_of_interest_used=True,
        )

        with self.assertRaises(
            RuntimeError
        ):
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )

    def test_deterministic_prediction_has_no_covariance_field(self):

        signature = inspect.signature(
            DeterministicSharedMeanPrediction
        )

        self.assertNotIn(
            "covariance",
            " ".join(
                signature.parameters
            ).lower(),
        )

        self.assertNotIn(
            "scale_tril",
            " ".join(
                signature.parameters
            ).lower(),
        )

    def test_plan_exposes_reactive_fallback(self):

        prediction = self.prediction(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        # Exact tie -> no accepted predictive match.
        boxes = (
            actor_box(
                anchor=(
                    9.0,
                    0.0,
                    0.0,
                )
            ),
            actor_box(
                anchor=(
                    11.0,
                    0.0,
                    0.0,
                )
            ),
        )

        plan = (
            build_deterministic_predictive_plan(
                (
                    prediction,
                ),
                boxes,
            )
        )

        self.assertEqual(
            plan.matched_forecasts,
            (),
        )

        self.assertEqual(
            plan.reactive_fallback_box_indices,
            (
                0,
                1,
            ),
        )

        self.assertEqual(
            plan.unmatched_prediction_indices,
            (
                0,
            ),
        )

    def test_matched_plan_contains_four_future_boxes(self):

        prediction = self.prediction()

        boxes = (
            actor_box(
                anchor=(
                    10.1,
                    0.0,
                    0.0,
                )
            ),
        )

        plan = (
            build_deterministic_predictive_plan(
                (
                    prediction,
                ),
                boxes,
            )
        )

        self.assertEqual(
            len(
                plan.matched_forecasts
            ),
            1,
        )

        self.assertEqual(
            plan.predicted_box_count,
            4,
        )

        self.assertEqual(
            plan.reactive_fallback_box_indices,
            (),
        )


if __name__ == "__main__":
    unittest.main()
