from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from iscai_stage6.adb.noninferiority_bounds import (
    BOUND_CAP,
    BOUND_FLOOR,
    BOUND_SCALE,
    OVERMASK_REQUIRED_SUPPORT,
    SUPPORTED_BOUND_METRICS,
    VRU_MINIMUM_SUPPORT,
    derive_reactive_only_noninferiority_delta,
)


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

CONFIG = (
    S6
    / "configs/"
      "stage6_noninferiority_bound_selection_rule.json"
)


class Block68NoninferiorityBoundRuleTest(
    unittest.TestCase
):

    def test_01_constants_exact(self):

        self.assertEqual(
            BOUND_SCALE,
            1.96,
        )

        self.assertEqual(
            BOUND_FLOOR,
            0.02,
        )

        self.assertEqual(
            BOUND_CAP,
            0.05,
        )

        self.assertEqual(
            VRU_MINIMUM_SUPPORT,
            20,
        )

        self.assertEqual(
            OVERMASK_REQUIRED_SUPPORT,
            120,
        )

    def test_02_metric_set_exact(self):

        self.assertEqual(
            SUPPORTED_BOUND_METRICS,
            (
                "over_masking_area",
                "pedestrian_visibility_proxy",
                "cyclist_visibility_proxy",
            ),
        )

    def test_03_constant_reactive_values_hit_floor(self):

        result = (
            derive_reactive_only_noninferiority_delta(
                "pedestrian_visibility_proxy",
                [0.5] * 20,
            )
        )

        self.assertAlmostEqual(
            result.frozen_delta,
            0.02,
            places=15,
        )

    def test_04_high_variability_hits_cap(self):

        values = [
            0.0,
            1.0,
        ] * 10

        result = (
            derive_reactive_only_noninferiority_delta(
                "cyclist_visibility_proxy",
                values,
            )
        )

        self.assertAlmostEqual(
            result.frozen_delta,
            0.05,
            places=15,
        )

    def test_05_interior_rule_not_floor_or_cap(self):

        values = [
            0.45,
            0.55,
        ] * 10

        result = (
            derive_reactive_only_noninferiority_delta(
                "pedestrian_visibility_proxy",
                values,
            )
        )

        self.assertGreater(
            result.frozen_delta,
            0.02,
        )

        self.assertLess(
            result.frozen_delta,
            0.05,
        )

        expected = (
            1.96
            *
            np.std(
                np.asarray(
                    values,
                    dtype=np.float64,
                ),
                ddof=1,
            )
            /
            np.sqrt(
                20.0
            )
        )

        self.assertAlmostEqual(
            result.frozen_delta,
            float(expected),
            places=15,
        )

    def test_06_support_rules(self):

        with self.assertRaises(
            ValueError
        ):
            derive_reactive_only_noninferiority_delta(
                "pedestrian_visibility_proxy",
                [0.5] * 19,
            )

        with self.assertRaises(
            ValueError
        ):
            derive_reactive_only_noninferiority_delta(
                "over_masking_area",
                [0.1] * 119,
            )

        result = (
            derive_reactive_only_noninferiority_delta(
                "over_masking_area",
                [0.1] * 120,
            )
        )

        self.assertEqual(
            result.support_count,
            120,
        )

    def test_07_invalid_values_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            derive_reactive_only_noninferiority_delta(
                "pedestrian_visibility_proxy",
                [0.5] * 19
                +
                [float("nan")],
            )

        with self.assertRaises(
            ValueError
        ):
            derive_reactive_only_noninferiority_delta(
                "cyclist_visibility_proxy",
                [0.5] * 19
                +
                [1.1],
            )

    def test_08_config_forbids_predictive_bound_selection(self):

        config = json.loads(
            CONFIG.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            config[
                "status"
            ],
            "FROZEN_PREOUTCOME_NI_BOUND_SELECTION_RULE",
        )

        self.assertEqual(
            config[
                "data_dependency"
            ][
                "allowed"
            ],
            [
                "original_reactive_ADB_development_values",
            ],
        )

        self.assertFalse(
            config[
                "data_dependency"
            ][
                "class_aware_predictive_values_allowed"
            ]
        )

        self.assertFalse(
            config[
                "data_dependency"
            ][
                "formal_values_allowed"
            ]
        )

        self.assertEqual(
            config[
                "vehicle_shadow_zone_violation"
            ][
                "formal_rule"
            ],
            "predictive_mean < reactive_mean",
        )


if __name__ == "__main__":
    unittest.main()
