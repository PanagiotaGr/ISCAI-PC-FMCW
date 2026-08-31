from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

AUDIT = (
    S6
    / "artifacts/block60/"
      "part_a_adb_source_audit.json"
)

ADAPTER = (
    S6
    / "configs/"
      "part_a_reactive_adb_adapter_contract.json"
)

STAGE6 = (
    S6
    / "configs/stage6_contract.json"
)


class Block60PartAADBContractTest(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads(
            AUDIT.read_text(
                encoding="utf-8"
            )
        )

        cls.adapter = json.loads(
            ADAPTER.read_text(
                encoding="utf-8"
            )
        )

        cls.stage6 = json.loads(
            STAGE6.read_text(
                encoding="utf-8"
            )
        )

    def test_part_a_source_audit_passes(self):
        self.assertEqual(
            self.audit["status"],
            "PASS_PART_A_ADB_SOURCE_AUDIT",
        )

    def test_provenance_is_not_conflated(self):
        labels = self.adapter[
            "provenance_labels"
        ]

        self.assertNotEqual(
            labels["paper_ADB"],
            labels["Part_A_extension"],
        )

        self.assertNotEqual(
            labels["Part_A_extension"],
            labels["Stage6_contribution"],
        )

    def test_reactive_baseline_is_causal(self):
        inputs = self.adapter[
            "input_contract"
        ]

        self.assertFalse(
            inputs["future_ground_truth"]
        )

        self.assertFalse(
            inputs["predictive_mean"]
        )

        self.assertFalse(
            inputs["predictive_covariance"]
        )

    def test_reactive_output_is_L_theta_r(self):
        illumination = self.adapter[
            "illumination_semantics"
        ]

        self.assertEqual(
            illumination["output"],
            "normalized L(theta,r)",
        )

    def test_raised_cosine_is_preserved(self):
        illumination = self.adapter[
            "illumination_semantics"
        ]

        self.assertEqual(
            illumination[
                "radial_transition"
            ],
            "raised-cosine",
        )

        self.assertIn(
            "0.5 * (1 - cos(pi*u))",
            illumination[
                "raised_cosine_normalized_formula"
            ],
        )

    def test_multiple_actor_combination(self):
        illumination = self.adapter[
            "illumination_semantics"
        ]

        self.assertIn(
            "elementwise minimum",
            illumination[
                "multiple_actor_combination"
            ],
        )

    def test_stage6_does_not_replace_adb_model(self):
        preservation = self.adapter[
            "Stage6_preservation_contract"
        ]

        self.assertTrue(
            preservation[
                "preserve_basic_L_theta_r_semantics"
            ]
        )

        self.assertFalse(
            preservation[
                "predictive_extension_may_change_entire_ADB_model"
            ]
        )

    def test_predictive_change_is_region_source(self):
        change = self.adapter[
            "Stage6_preservation_contract"
        ][
            "predictive_extension_changes_region_source"
        ]

        self.assertIn(
            "current deterministic region",
            change,
        )

        self.assertIn(
            "future probabilistic region",
            change,
        )

    def test_oracle_not_measured_gt(self):
        oracle = self.adapter[
            "fair_baseline_semantics"
        ][
            "oracle_future_ADB"
        ]

        self.assertIn(
            "constructed",
            oracle,
        )

        self.assertIn(
            "not measured",
            oracle,
        )

    def test_part_a_not_modified(self):
        gate = self.adapter[
            "implementation_gate"
        ]

        self.assertTrue(
            gate[
                "must_not_modify_Part_A"
            ]
        )

    def test_adapter_not_implemented_in_block60(self):
        self.assertTrue(
            self.adapter[
                "not_yet_implementation"
            ]
        )

        self.assertEqual(
            self.adapter[
                "implementation_gate"
            ][
                "adapter_implementation"
            ],
            "BLOCK6.2",
        )


if __name__ == "__main__":
    unittest.main()
