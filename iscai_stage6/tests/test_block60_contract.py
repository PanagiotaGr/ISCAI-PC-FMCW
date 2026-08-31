from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

CONTRACT = (
    S6
    / "configs/stage6_contract.json"
)

STAGE5_HANDOFF = (
    ROOT
    / "iscai_stage5/artifacts/block510/"
      "stage5_to_stage6_handoff.json"
)


class Block60ContractTest(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):

        cls.contract = json.loads(
            CONTRACT.read_text(
                encoding="utf-8"
            )
        )

        cls.handoff = json.loads(
            STAGE5_HANDOFF.read_text(
                encoding="utf-8"
            )
        )

    def test_stage_and_status(self):

        self.assertEqual(
            self.contract["stage"],
            6,
        )

        self.assertEqual(
            self.contract["status"],
            "FROZEN_PREIMPLEMENTATION_CONTRACT",
        )

    def test_stage5_handoff_boundary(self):

        self.assertEqual(
            self.handoff["status"],
            "FROZEN_HANDOFF",
        )

        self.assertFalse(
            self.handoff[
                "Stage6_required_next_layer"
            ][
                "communication_beam_actuator_reused_as_ADB"
            ]
        )

    def test_primary_mode(self):

        primary = (
            self.contract[
                "experimental_modes"
            ][
                "primary"
            ]
        )

        self.assertIn(
            "track-based PC-FMCW-like",
            primary,
        )

    def test_actor_population_is_not_receiver_only(self):

        policy = self.contract[
            "adb_actor_population"
        ]

        self.assertFalse(
            policy[
                "communication_receiver_only"
            ]
        )

        self.assertFalse(
            policy[
                "tracks_to_predict_only"
            ]
        )

        self.assertFalse(
            policy[
                "tracks_to_predict_as_control_selector"
            ]
        )

        self.assertFalse(
            policy[
                "future_truth_for_controller"
            ]
        )

    def test_full_box_contract(self):

        geometry = self.contract[
            "geometry"
        ]

        self.assertFalse(
            geometry[
                "centroid_only"
            ]
        )

        self.assertTrue(
            geometry[
                "box_corners"
            ]
        )

        self.assertTrue(
            geometry[
                "actor_width_height"
            ]
        )

        self.assertTrue(
            geometry[
                "predictive_covariance"
            ]
        )

        self.assertTrue(
            geometry[
                "class_margin"
            ]
        )

    def test_exact_mask_aggregation(self):

        occupancy = self.contract[
            "occupancy"
        ]

        self.assertEqual(
            occupancy[
                "aggregate_mask"
            ],
            (
                "M_tau = 1 - product_i(1 - M_i_tau)"
            ),
        )

        self.assertTrue(
            occupancy[
                "probability_kept_distinct_from_mask"
            ]
        )

        self.assertTrue(
            occupancy[
                "mask_kept_distinct_from_intensity"
            ]
        )

    def test_part_a_illumination_preserved(self):

        adb = self.contract[
            "part_a_illumination"
        ]

        required = (
            "preserve_L_theta_r",
            "radial_thresholds",
            "angular_shadow_zones",
            "raised_cosine_transitions",
            "minimum_illumination_floor",
            "temporal_smoothing",
            "actuation_rate_limits",
        )

        for key in required:
            self.assertTrue(
                adb[key]
            )

        self.assertFalse(
            adb[
                "arbitrary_replacement_of_ADB_model"
            ]
        )

    def test_class_aware_not_margin_only(self):

        policies = self.contract[
            "class_policies"
        ]

        self.assertTrue(
            policies[
                "class_awareness_not_margin_only"
            ]
        )

        self.assertEqual(
            set(
                policies.keys()
            )
            -
            {
                "class_awareness_not_margin_only",
                "allowed_policy_differences",
            },
            {
                "vehicle",
                "pedestrian",
                "cyclist",
            },
        )

        self.assertGreaterEqual(
            len(
                policies[
                    "allowed_policy_differences"
                ]
            ),
            5,
        )

    def test_all_pdf_adb_baselines(self):

        actual = set(
            self.contract[
                "adb_baselines"
            ][
                "required_pdf_labels"
            ]
        )

        expected = {
            "static_ADB",
            "original_reactive_ADB",
            "deterministic_predictive_ADB",
            "uncertainty_aware_predictive_ADB",
            "class_agnostic_predictive_ADB",
            "class_aware_predictive_ADB",
            "oracle_future_ADB",
        }

        self.assertEqual(
            actual,
            expected,
        )

    def test_all_pdf_adb_metrics(self):

        actual = set(
            self.contract[
                "adb_metrics"
            ][
                "required"
            ]
        )

        expected = {
            "mask_IoU_with_constructed_oracle_future_mask",
            "vehicle_shadow_zone_violation",
            "glare_risk_exposure",
            "over_masking_area",
            "road_illumination_retention",
            "pedestrian_visibility_proxy",
            "cyclist_visibility_proxy",
            "false_dimming",
            "temporal_smoothness",
            "flicker_change_rate",
            "energy_consumption",
            "actuation_latency",
        }

        self.assertEqual(
            actual,
            expected,
        )

    def test_oracle_is_constructed(self):

        self.assertEqual(
            self.contract[
                "adb_metrics"
            ][
                "oracle_semantics"
            ],
            "CONSTRUCTED_REFERENCE_NOT_MEASURED_GT",
        )

    def test_stage6_acceptance(self):

        acceptance = self.contract[
            "acceptance"
        ]

        self.assertTrue(
            acceptance[
                "must_reduce_vehicle_shadow_zone_violation"
            ]
        )

        self.assertTrue(
            acceptance[
                "must_not_uncontrollably_increase_over_masking"
            ]
        )

        self.assertTrue(
            acceptance[
                "must_not_uncontrollably_reduce_VRU_visibility"
            ]
        )

        self.assertFalse(
            acceptance[
                "formal_outcomes_used_for_threshold_selection"
            ]
        )

        self.assertFalse(
            acceptance[
                "formal_posthoc_tuning"
            ]
        )

    def test_stage_boundaries(self):

        scope = self.contract[
            "scope"
        ]

        self.assertEqual(
            scope[
                "shared_posterior_joint_evaluation"
            ],
            "STAGE7",
        )

        self.assertEqual(
            scope[
                "full_joint_end_to_end_latency"
            ],
            "STAGE7",
        )

        self.assertEqual(
            scope[
                "DeepSense"
            ],
            "STAGE8",
        )


if __name__ == "__main__":
    unittest.main()
