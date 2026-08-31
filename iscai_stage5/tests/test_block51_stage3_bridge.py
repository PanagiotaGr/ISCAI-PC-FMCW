import inspect
import math
import unittest

from iscai_stage5.receiver_policy import (
    ASSOCIATION_VALID_REQUIRED,
    CONNECTIVITY_SEMANTICS,
    CURRENT_CAUSAL_AVAILABILITY_REQUIRED,
    FUTURE_TRUTH_ALLOWED,
    MAX_PLANAR_RANGE_M,
    MAX_RANGE_DEVELOPMENT_TUNING_PERFORMED,
    MAX_RANGE_FORMAL_TUNING_PERFORMED,
    MAX_RANGE_SEMANTICS,
    MINIMUM_FORWARD_H0_M,
    ORACLE_CONNECTIVITY_ALLOWED,
    PERFECT_TRACK_ID_ALLOWED,
    POSITIVE_FORWARD_H0_REQUIRED,
    PRIMARY_POLICY,
    RECEIVER_POLICY_STATUS,
    TRACKS_TO_PREDICT_ALLOWED,
    VEHICLE_CLASS_REQUIRED,
    receiver_policy_dict,
)

from iscai_stage5.stage3_receiver_bridge import (
    CurrentAssociatedActorEstimateH0,
    SOURCE_SEMANTICS,
    receiver_candidate_from_current_estimate,
    receiver_candidates_from_current_estimates,
)


class TestBlock51ReceiverPolicyFreeze(
    unittest.TestCase
):

    def test_01_policy_is_frozen(self):
        self.assertEqual(
            RECEIVER_POLICY_STATUS,
            "FROZEN",
        )

        self.assertEqual(
            PRIMARY_POLICY,
            "nearest_causal_vehicle_ahead",
        )

        self.assertEqual(
            CONNECTIVITY_SEMANTICS,
            "constructed_hypothetical_connected_vehicle",
        )

    def test_02_no_arbitrary_range_cutoff(self):
        self.assertIsNone(
            MAX_PLANAR_RANGE_M
        )

        self.assertEqual(
            MINIMUM_FORWARD_H0_M,
            0.0,
        )

        self.assertEqual(
            MAX_RANGE_SEMANTICS,
            "no_additional_receiver_eligibility_range_cutoff",
        )

        self.assertFalse(
            MAX_RANGE_DEVELOPMENT_TUNING_PERFORMED
        )

        self.assertFalse(
            MAX_RANGE_FORMAL_TUNING_PERFORMED
        )

    def test_03_required_eligibility(self):
        self.assertTrue(
            VEHICLE_CLASS_REQUIRED
        )

        self.assertTrue(
            CURRENT_CAUSAL_AVAILABILITY_REQUIRED
        )

        self.assertTrue(
            ASSOCIATION_VALID_REQUIRED
        )

        self.assertTrue(
            POSITIVE_FORWARD_H0_REQUIRED
        )

    def test_04_forbidden_metadata(self):
        self.assertFalse(
            TRACKS_TO_PREDICT_ALLOWED
        )

        self.assertFalse(
            FUTURE_TRUTH_ALLOWED
        )

        self.assertFalse(
            PERFECT_TRACK_ID_ALLOWED
        )

        self.assertFalse(
            ORACLE_CONNECTIVITY_ALLOWED
        )

    def test_05_policy_dict_preserves_no_range_cutoff(self):
        payload = receiver_policy_dict()

        self.assertIsNone(
            payload[
                "max_planar_range_m"
            ]
        )

        self.assertFalse(
            payload[
                "forbidden_inputs"
            ][
                "tracks_to_predict"
            ]
        )

        self.assertFalse(
            payload[
                "forbidden_inputs"
            ][
                "future_truth"
            ]
        )


class TestBlock51Stage3ReceiverBridge(
    unittest.TestCase
):

    def test_06_current_estimate_to_candidate(self):
        estimate = (
            CurrentAssociatedActorEstimateH0(
                semantic_class="vehicle",

                position_h0_m=(
                    12.0,
                    -1.0,
                    0.5,
                ),

                current_available=True,

                association_valid=True,
            )
        )

        candidate = (
            receiver_candidate_from_current_estimate(
                estimate
            )
        )

        self.assertEqual(
            candidate.semantic_class,
            "vehicle",
        )

        self.assertEqual(
            candidate.position_h0_m,
            (
                12.0,
                -1.0,
                0.5,
            ),
        )

        self.assertTrue(
            candidate.current_available
        )

        self.assertTrue(
            candidate.association_valid
        )

    def test_07_bridge_preserves_order(self):
        estimates = (
            CurrentAssociatedActorEstimateH0(
                semantic_class="vehicle",
                position_h0_m=(
                    20.0,
                    0.0,
                    0.0,
                ),
            ),
            CurrentAssociatedActorEstimateH0(
                semantic_class="vehicle",
                position_h0_m=(
                    10.0,
                    1.0,
                    0.0,
                ),
            ),
        )

        candidates = (
            receiver_candidates_from_current_estimates(
                estimates
            )
        )

        self.assertEqual(
            len(
                candidates
            ),
            2,
        )

        self.assertEqual(
            candidates[
                0
            ].position_h0_m[
                0
            ],
            20.0,
        )

        self.assertEqual(
            candidates[
                1
            ].position_h0_m[
                0
            ],
            10.0,
        )

    def test_08_bridge_rejects_nonfinite_position(self):
        estimate = (
            CurrentAssociatedActorEstimateH0(
                semantic_class="vehicle",

                position_h0_m=(
                    math.nan,
                    0.0,
                    0.0,
                ),
            )
        )

        with self.assertRaises(
            ValueError
        ):
            receiver_candidate_from_current_estimate(
                estimate
            )

    def test_09_bridge_rejects_empty_class(self):
        estimate = (
            CurrentAssociatedActorEstimateH0(
                semantic_class="",

                position_h0_m=(
                    1.0,
                    0.0,
                    0.0,
                ),
            )
        )

        with self.assertRaises(
            ValueError
        ):
            receiver_candidate_from_current_estimate(
                estimate
            )

    def test_10_source_semantics_fixed(self):
        estimate = (
            CurrentAssociatedActorEstimateH0(
                semantic_class="vehicle",

                position_h0_m=(
                    1.0,
                    0.0,
                    0.0,
                ),
            )
        )

        self.assertEqual(
            estimate.source_semantics,
            SOURCE_SEMANTICS,
        )

        self.assertEqual(
            SOURCE_SEMANTICS,
            "current_causal_associated_actor_estimate_H0",
        )

    def test_11_bridge_signature_has_no_forbidden_metadata(self):
        signatures = (
            str(
                inspect.signature(
                    CurrentAssociatedActorEstimateH0
                )
            ).lower(),

            str(
                inspect.signature(
                    receiver_candidate_from_current_estimate
                )
            ).lower(),

            str(
                inspect.signature(
                    receiver_candidates_from_current_estimates
                )
            ).lower(),
        )

        forbidden = (
            "tracks_to_predict",
            "objects_of_interest",
            "future_truth",
            "ground_truth",
            "oracle",
            "track_id",
        )

        for signature in signatures:

            for token in forbidden:

                self.assertNotIn(
                    token,
                    signature,
                )

    def test_12_bridge_does_not_add_identity_field(self):
        estimate = (
            CurrentAssociatedActorEstimateH0(
                semantic_class="vehicle",

                position_h0_m=(
                    5.0,
                    0.0,
                    0.0,
                ),
            )
        )

        candidate = (
            receiver_candidate_from_current_estimate(
                estimate
            )
        )

        candidate_fields = {
            field
            for field in (
                candidate.__dataclass_fields__
            )
        }

        self.assertNotIn(
            "track_id",
            candidate_fields,
        )

        self.assertNotIn(
            "scenario_id",
            candidate_fields,
        )

        self.assertNotIn(
            "tracks_to_predict",
            candidate_fields,
        )


if __name__ == "__main__":
    unittest.main()
