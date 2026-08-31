from __future__ import annotations

import json
from pathlib import Path
import unittest

from iscai_stage6.adb.baseline_metric_contract import (
    ADBMetric,
    required_metric_interfaces,
)


ROOT = Path(
    "/home/agni/waymo"
)

S6 = (
    ROOT
    / "iscai_stage6"
)

PROTOCOL = (
    S6
    / "configs/"
      "stage6_metric_freeze_protocol.json"
)


class Block68MetricReconciliationTest(
    unittest.TestCase
):

    @classmethod
    def setUpClass(
        cls,
    ):

        cls.protocol = json.loads(
            PROTOCOL.read_text(
                encoding="utf-8"
            )
        )

    def test_01_authority(
        self,
    ):

        self.assertEqual(
            self.protocol[
                "authority"
            ][
                "binding_source"
            ],
            "Block6.6_preoutcome_preregistration",
        )

    def test_02_temporal_lower_is_better(
        self,
    ):

        self.assertEqual(
            self.protocol[
                "metrics"
            ][
                "temporal_smoothness"
            ][
                "direction"
            ],
            "lower_is_better",
        )

    def test_03_no_flicker_threshold(
        self,
    ):

        flicker = (
            self.protocol[
                "metrics"
            ][
                "flicker_change_rate"
            ]
        )

        self.assertNotIn(
            "delta_threshold",
            flicker,
        )

    def test_04_overmask_not_road_roi_normalized(
        self,
    ):

        interfaces = {
            metric.name:
                metric
            for metric in (
                required_metric_interfaces()
            )
        }

        self.assertFalse(
            interfaces[
                ADBMetric.OVER_MASKING_AREA
            ].requires_road_ROI
        )

    def test_05_false_dimming_not_road_roi_normalized(
        self,
    ):

        interfaces = {
            metric.name:
                metric
            for metric in (
                required_metric_interfaces()
            )
        }

        self.assertFalse(
            interfaces[
                ADBMetric.FALSE_DIMMING
            ].requires_road_ROI
        )

    def test_06_energy_not_road_roi_normalized(
        self,
    ):

        interfaces = {
            metric.name:
                metric
            for metric in (
                required_metric_interfaces()
            )
        }

        self.assertFalse(
            interfaces[
                ADBMetric.ENERGY_CONSUMPTION
            ].requires_road_ROI
        )


if __name__ == "__main__":
    unittest.main()
