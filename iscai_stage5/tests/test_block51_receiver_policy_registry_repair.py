from __future__ import annotations

import inspect
import math
from pathlib import Path
import json
import unittest

from iscai_stage5.receiver_selection import (
    NO_ELIGIBLE_RECEIVER,
    SELECTED_RECEIVER,
    ReceiverCandidate,
    select_primary_receiver,
)

from iscai_stage5.receiver_policy_registry import (
    CANONICAL_POLICIES,
    FIXED_CAUSAL_TARGET,
    HIGHEST_EXPECTED_LINK_UTILITY,
    NEAREST_CAUSAL_VEHICLE_AHEAD,
    NEAREST_CONNECTED_VEHICLE_AHEAD,
    ONCOMING_CONNECTED_VEHICLE,
    PRECEDING_CONNECTED_VEHICLE,
    PRIMARY_POLICY,
    ReceiverPolicyMetadata,
    normalize_policy_name,
    policy_contract,
    receiver_policy_registry,
    select_receiver_policy,
)


class TestBlock51ReceiverPolicyRegistryRepair(
    unittest.TestCase
):

    def setUp(self):

        self.candidates = (
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    10.0,
                    1.0,
                    0.0,
                ),
            ),

            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    20.0,
                    0.2,
                    0.0,
                ),
            ),

            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    30.0,
                    -0.5,
                    0.0,
                ),
            ),

            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    40.0,
                    1.0,
                    0.0,
                ),
            ),

            # Never eligible in the existing frozen selector.
            ReceiverCandidate(
                semantic_class="pedestrian",
                position_h0_m=(
                    2.0,
                    0.0,
                    0.0,
                ),
            ),

            # Never eligible: current unavailable.
            ReceiverCandidate(
                semantic_class="vehicle",
                position_h0_m=(
                    1.0,
                    0.0,
                    0.0,
                ),
                current_available=False,
            ),
        )

        self.metadata = (
            ReceiverPolicyMetadata(
                association_key="nearest",
                causal_is_preceding=False,
                causal_is_oncoming=False,
                current_heading_h0_rad=0.0,
                expected_link_utility=0.3,
            ),

            ReceiverPolicyMetadata(
                association_key="preceding",
                causal_is_preceding=True,
                causal_is_oncoming=False,
                current_heading_h0_rad=0.0,
                expected_link_utility=0.2,
            ),

            ReceiverPolicyMetadata(
                association_key="oncoming",
                causal_is_preceding=False,
                causal_is_oncoming=True,
                current_heading_h0_rad=math.pi,
                expected_link_utility=0.4,
            ),

            ReceiverPolicyMetadata(
                association_key="best_utility",
                causal_is_preceding=False,
                causal_is_oncoming=False,
                current_heading_h0_rad=0.0,
                expected_link_utility=0.95,
            ),

            # Enormous utility must not override vehicle-only
            # eligibility from the frozen selector.
            ReceiverPolicyMetadata(
                association_key="pedestrian",
                causal_is_preceding=True,
                causal_is_oncoming=True,
                current_heading_h0_rad=math.pi,
                expected_link_utility=1000.0,
            ),

            # Enormous utility must not override current availability.
            ReceiverPolicyMetadata(
                association_key="unavailable",
                causal_is_preceding=True,
                causal_is_oncoming=True,
                current_heading_h0_rad=math.pi,
                expected_link_utility=2000.0,
            ),
        )

    def test_01_primary_policy_remains_exact_existing_selector(self):

        frozen = select_primary_receiver(
            self.candidates
        )

        repaired = select_receiver_policy(
            NEAREST_CAUSAL_VEHICLE_AHEAD,
            self.candidates,
        )

        self.assertEqual(
            PRIMARY_POLICY,
            "nearest_causal_vehicle_ahead",
        )

        self.assertEqual(
            repaired.status,
            frozen.status,
        )

        self.assertEqual(
            repaired.selected_candidate_index,
            frozen.selected_candidate_index,
        )

        self.assertEqual(
            repaired.eligible_candidate_count,
            frozen.eligible_candidate_count,
        )

        self.assertAlmostEqual(
            repaired.planar_range_m,
            frozen.planar_range_m,
        )

    def test_02_registry_has_all_five_canonical_policies(self):

        expected = {
            "preceding_connected_vehicle",
            "nearest_causal_vehicle_ahead",
            "oncoming_connected_vehicle",
            "fixed_causal_target",
            "highest_expected_link_utility",
        }

        self.assertEqual(
            set(CANONICAL_POLICIES),
            expected,
        )

        registry = (
            receiver_policy_registry()
        )

        for policy in expected:
            self.assertIn(
                policy,
                registry,
            )

            self.assertTrue(
                callable(
                    registry[policy]
                )
            )

    def test_03_nearest_alias_is_primary(self):

        self.assertEqual(
            normalize_policy_name(
                NEAREST_CONNECTED_VEHICLE_AHEAD
            ),
            NEAREST_CAUSAL_VEHICLE_AHEAD,
        )

    def test_04_preceding_policy(self):

        result = select_receiver_policy(
            PRECEDING_CONNECTED_VEHICLE,
            self.candidates,
            metadata=self.metadata,
        )

        self.assertEqual(
            result.status,
            SELECTED_RECEIVER,
        )

        self.assertEqual(
            result.selected_candidate_index,
            1,
        )

    def test_05_oncoming_policy(self):

        result = select_receiver_policy(
            ONCOMING_CONNECTED_VEHICLE,
            self.candidates,
            metadata=self.metadata,
        )

        self.assertEqual(
            result.status,
            SELECTED_RECEIVER,
        )

        self.assertEqual(
            result.selected_candidate_index,
            2,
        )

    def test_06_oncoming_current_heading_fallback(self):

        metadata = list(
            self.metadata
        )

        metadata[2] = (
            ReceiverPolicyMetadata(
                association_key="oncoming",
                causal_is_preceding=False,
                causal_is_oncoming=None,
                current_heading_h0_rad=math.pi,
                expected_link_utility=0.4,
            )
        )

        result = select_receiver_policy(
            ONCOMING_CONNECTED_VEHICLE,
            self.candidates,
            metadata=metadata,
        )

        self.assertEqual(
            result.selected_candidate_index,
            2,
        )

    def test_07_fixed_causal_target(self):

        result = select_receiver_policy(
            FIXED_CAUSAL_TARGET,
            self.candidates,
            metadata=self.metadata,
            fixed_association_key="preceding",
        )

        self.assertEqual(
            result.status,
            SELECTED_RECEIVER,
        )

        self.assertEqual(
            result.selected_candidate_index,
            1,
        )

        missing = select_receiver_policy(
            FIXED_CAUSAL_TARGET,
            self.candidates,
            metadata=self.metadata,
            fixed_association_key="not_available",
        )

        self.assertEqual(
            missing.status,
            NO_ELIGIBLE_RECEIVER,
        )

        self.assertIsNone(
            missing.selected_candidate_index,
        )

    def test_08_highest_expected_link_utility(self):

        result = select_receiver_policy(
            HIGHEST_EXPECTED_LINK_UTILITY,
            self.candidates,
            metadata=self.metadata,
        )

        self.assertEqual(
            result.status,
            SELECTED_RECEIVER,
        )

        self.assertEqual(
            result.selected_candidate_index,
            3,
        )

    def test_09_existing_eligibility_is_reused(self):

        preceding = select_receiver_policy(
            PRECEDING_CONNECTED_VEHICLE,
            self.candidates,
            metadata=self.metadata,
        )

        utility = select_receiver_policy(
            HIGHEST_EXPECTED_LINK_UTILITY,
            self.candidates,
            metadata=self.metadata,
        )

        self.assertNotIn(
            preceding.selected_candidate_index,
            (4, 5),
        )

        self.assertNotIn(
            utility.selected_candidate_index,
            (4, 5),
        )

    def test_10_policy_metadata_length_must_match(self):

        with self.assertRaises(
            ValueError
        ):
            select_receiver_policy(
                PRECEDING_CONNECTED_VEHICLE,
                self.candidates,
                metadata=self.metadata[:-1],
            )

    def test_11_nonprimary_policy_requires_metadata(self):

        with self.assertRaises(
            ValueError
        ):
            select_receiver_policy(
                PRECEDING_CONNECTED_VEHICLE,
                self.candidates,
            )

    def test_12_fixed_target_requires_key(self):

        with self.assertRaises(
            ValueError
        ):
            select_receiver_policy(
                FIXED_CAUSAL_TARGET,
                self.candidates,
                metadata=self.metadata,
            )

    def test_13_selector_api_contains_no_forbidden_truth_metadata(self):

        signature = str(
            inspect.signature(
                select_receiver_policy
            )
        ).lower()

        for forbidden in (
            "tracks_to_predict",
            "future",
            "ground_truth",
            "oracle",
            "track_id",
        ):
            self.assertNotIn(
                forbidden,
                signature,
            )

    def test_14_contract_forbids_leakage(self):

        contract = policy_contract()

        guards = (
            contract[
                "leakage_guards"
            ]
        )

        self.assertFalse(
            guards[
                "future_truth_selector"
            ]
        )

        self.assertFalse(
            guards[
                "tracks_to_predict_selector"
            ]
        )

        self.assertFalse(
            guards[
                "perfect_WOMD_track_ID_selector"
            ]
        )

        self.assertFalse(
            guards[
                "realized_future_link_utility"
            ]
        )

    def test_15_existing_candidate_type_is_not_modified(self):

        fields = set(
            ReceiverCandidate
            .__dataclass_fields__
        )

        for forbidden in (
            "track_id",
            "scenario_id",
            "tracks_to_predict",
            "association_key",
        ):
            self.assertNotIn(
                forbidden,
                fields,
            )

    def test_16_original_frozen_receiver_policy_still_primary(self):

        root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        original = json.loads(
            (
                root
                / "configs"
                / "receiver_policy.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            original["status"],
            "FROZEN",
        )

        self.assertEqual(
            original["primary_policy"],
            "nearest_causal_vehicle_ahead",
        )


if __name__ == "__main__":
    unittest.main()
