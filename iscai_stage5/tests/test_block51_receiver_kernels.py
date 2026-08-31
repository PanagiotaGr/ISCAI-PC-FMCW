import inspect
import math
import unittest

import numpy as np

from iscai_stage5.receiver_geometry import (
    ReceiverOffsetDistribution,
    receiver_geometry_distribution_h0,
)

from iscai_stage5.receiver_selection import (
    NO_ELIGIBLE_RECEIVER,
    ReceiverCandidate,
    ReceiverSelectionConfig,
    SELECTED_RECEIVER,
    select_primary_receiver,
)


class TestBlock51ReceiverSelection(
    unittest.TestCase
):

    def test_01_nearest_vehicle_ahead_selected(self):
        candidates = [
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    20.0,
                    0.0,
                    0.0,
                ),
            ),
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    10.0,
                    1.0,
                    0.0,
                ),
            ),
        ]

        result = select_primary_receiver(
            candidates
        )

        self.assertEqual(
            result.status,
            SELECTED_RECEIVER,
        )

        self.assertEqual(
            result.selected_candidate_index,
            1,
        )

        self.assertEqual(
            result.eligible_candidate_count,
            2,
        )

    def test_02_nonvehicle_is_not_receiver(self):
        candidates = [
            ReceiverCandidate(
                semantic_class="pedestrian",
                position_h0_m=(
                    2.0,
                    0.0,
                    0.0,
                ),
            ),
            ReceiverCandidate(
                semantic_class="cyclist",
                position_h0_m=(
                    3.0,
                    0.0,
                    0.0,
                ),
            ),
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    8.0,
                    0.0,
                    0.0,
                ),
            ),
        ]

        result = select_primary_receiver(
            candidates
        )

        self.assertEqual(
            result.selected_candidate_index,
            2,
        )

        self.assertEqual(
            result.eligible_candidate_count,
            1,
        )

    def test_03_vehicle_behind_is_not_receiver(self):
        candidates = [
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    -1.0,
                    0.0,
                    0.0,
                ),
            ),
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    5.0,
                    0.0,
                    0.0,
                ),
            ),
        ]

        result = select_primary_receiver(
            candidates
        )

        self.assertEqual(
            result.selected_candidate_index,
            1,
        )

        self.assertEqual(
            result.eligible_candidate_count,
            1,
        )

    def test_04_current_unavailable_or_invalid_excluded(self):
        candidates = [
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    2.0,
                    0.0,
                    0.0,
                ),
                current_available=False,
            ),
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    3.0,
                    0.0,
                    0.0,
                ),
                association_valid=False,
            ),
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    7.0,
                    0.0,
                    0.0,
                ),
            ),
        ]

        result = select_primary_receiver(
            candidates
        )

        self.assertEqual(
            result.selected_candidate_index,
            2,
        )

    def test_05_no_receiver_is_explicit(self):
        candidates = [
            ReceiverCandidate(
                semantic_class="pedestrian",
                position_h0_m=(
                    2.0,
                    0.0,
                    0.0,
                ),
            ),
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    -2.0,
                    0.0,
                    0.0,
                ),
            ),
        ]

        result = select_primary_receiver(
            candidates
        )

        self.assertEqual(
            result.status,
            NO_ELIGIBLE_RECEIVER,
        )

        self.assertIsNone(
            result.selected_candidate_index
        )

        self.assertEqual(
            result.eligible_candidate_count,
            0,
        )

    def test_06_optional_range_gate(self):
        candidates = [
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    100.0,
                    0.0,
                    0.0,
                ),
            )
        ]

        result = select_primary_receiver(
            candidates,
            ReceiverSelectionConfig(
                max_planar_range_m=50.0,
            ),
        )

        self.assertEqual(
            result.status,
            NO_ELIGIBLE_RECEIVER,
        )

    def test_07_planar_range_not_forward_x_is_primary_rank(self):
        candidates = [
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    5.0,
                    10.0,
                    0.0,
                ),
            ),
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    8.0,
                    0.0,
                    0.0,
                ),
            ),
        ]

        result = select_primary_receiver(
            candidates
        )

        self.assertEqual(
            result.selected_candidate_index,
            1,
        )

        self.assertAlmostEqual(
            result.planar_range_m,
            8.0,
        )

    def test_08_selector_api_has_no_future_metadata(self):
        signature = str(
            inspect.signature(
                select_primary_receiver
            )
        ).lower()

        forbidden = (
            "tracks_to_predict",
            "future",
            "ground_truth",
            "oracle",
            "track_id",
        )

        for token in forbidden:
            self.assertNotIn(
                token,
                signature,
            )


class TestBlock51ReceiverGeometry(
    unittest.TestCase
):

    def setUp(self):
        self.offset = (
            ReceiverOffsetDistribution(
                mean_body_m=(
                    2.0,
                    0.0,
                    0.5,
                ),

                covariance_body_m2=(
                    (
                        0.25,
                        0.0,
                        0.0,
                    ),
                    (
                        0.0,
                        1.0,
                        0.0,
                    ),
                    (
                        0.0,
                        0.0,
                        0.04,
                    ),
                ),
            )
        )

    def test_09_centroid_baseline(self):
        result = (
            receiver_geometry_distribution_h0(
                actor_center_h0_m=(
                    10.0,
                    3.0,
                    1.0,
                ),
                heading_h0_rad=0.7,
                mode="centroid_baseline",
            )
        )

        np.testing.assert_allclose(
            result.receiver_mean_h0_m,
            (
                10.0,
                3.0,
                1.0,
            ),
            atol=1e-12,
            rtol=0.0,
        )

        np.testing.assert_allclose(
            result.placement_covariance_h0_m2,
            np.zeros(
                (
                    3,
                    3,
                )
            ),
            atol=1e-12,
            rtol=0.0,
        )

    def test_10_known_offset_yaw_zero(self):
        result = (
            receiver_geometry_distribution_h0(
                actor_center_h0_m=(
                    10.0,
                    3.0,
                    1.0,
                ),
                heading_h0_rad=0.0,
                mode="known_receiver_offset",
                receiver_offset=self.offset,
            )
        )

        np.testing.assert_allclose(
            result.receiver_mean_h0_m,
            (
                12.0,
                3.0,
                1.5,
            ),
            atol=1e-12,
            rtol=0.0,
        )

        np.testing.assert_allclose(
            result.placement_covariance_h0_m2,
            np.zeros(
                (
                    3,
                    3,
                )
            ),
            atol=1e-12,
            rtol=0.0,
        )

    def test_11_known_offset_yaw_pi_over_two(self):
        result = (
            receiver_geometry_distribution_h0(
                actor_center_h0_m=(
                    10.0,
                    3.0,
                    1.0,
                ),
                heading_h0_rad=(
                    math.pi
                    /
                    2.0
                ),
                mode="known_receiver_offset",
                receiver_offset=self.offset,
            )
        )

        np.testing.assert_allclose(
            result.receiver_mean_h0_m,
            (
                10.0,
                5.0,
                1.5,
            ),
            atol=1e-12,
            rtol=0.0,
        )

    def test_12_uncertain_offset_rotates_covariance(self):
        result = (
            receiver_geometry_distribution_h0(
                actor_center_h0_m=(
                    0.0,
                    0.0,
                    0.0,
                ),
                heading_h0_rad=(
                    math.pi
                    /
                    2.0
                ),
                mode="uncertain_receiver_offset",
                receiver_offset=self.offset,
            )
        )

        expected = np.asarray(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.25,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.04,
                ],
            ]
        )

        np.testing.assert_allclose(
            result.placement_covariance_h0_m2,
            expected,
            atol=1e-12,
            rtol=0.0,
        )

    def test_13_invalid_non_psd_covariance_rejected(self):
        offset = ReceiverOffsetDistribution(
            mean_body_m=(
                0.0,
                0.0,
                0.0,
            ),

            covariance_body_m2=(
                (
                    1.0,
                    0.0,
                    0.0,
                ),
                (
                    0.0,
                    -0.1,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    1.0,
                ),
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            receiver_geometry_distribution_h0(
                actor_center_h0_m=(
                    0.0,
                    0.0,
                    0.0,
                ),
                heading_h0_rad=0.0,
                mode=(
                    "uncertain_receiver_offset"
                ),
                receiver_offset=offset,
            )

    def test_14_asymmetric_covariance_rejected(self):
        offset = ReceiverOffsetDistribution(
            mean_body_m=(
                0.0,
                0.0,
                0.0,
            ),

            covariance_body_m2=(
                (
                    1.0,
                    0.2,
                    0.0,
                ),
                (
                    0.0,
                    1.0,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    1.0,
                ),
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            receiver_geometry_distribution_h0(
                actor_center_h0_m=(
                    0.0,
                    0.0,
                    0.0,
                ),
                heading_h0_rad=0.0,
                mode=(
                    "uncertain_receiver_offset"
                ),
                receiver_offset=offset,
            )

    def test_15_covariance_semantics_explicit(self):
        result = (
            receiver_geometry_distribution_h0(
                actor_center_h0_m=(
                    0.0,
                    0.0,
                    0.0,
                ),
                heading_h0_rad=0.0,
                mode="uncertain_receiver_offset",
                receiver_offset=self.offset,
            )
        )

        self.assertEqual(
            result.covariance_semantics,
            "receiver_placement_uncertainty_only",
        )

    def test_16_geometry_api_has_no_future_truth_argument(self):
        signature = str(
            inspect.signature(
                receiver_geometry_distribution_h0
            )
        ).lower()

        for token in (
            "future_truth",
            "ground_truth",
            "oracle",
            "tracks_to_predict",
        ):
            self.assertNotIn(
                token,
                signature,
            )


if __name__ == "__main__":
    unittest.main()
