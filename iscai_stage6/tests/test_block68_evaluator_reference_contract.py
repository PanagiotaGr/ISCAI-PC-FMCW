from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from iscai_stage6.adb.evaluator_reference import (
    FROZEN_ADB_ACTOR_CLASSES,
    FROZEN_EVALUATOR_REFERENCE_SEMANTICS,
    FROZEN_HORIZONS_S,
    aggregate_constructed_oracle_supports,
    frozen_road_roi,
    validate_final_metric_illumination,
)


ROOT = Path(
    "/home/agni/waymo"
)

S6 = (
    ROOT
    / "iscai_stage6"
)

CONTRACT = (
    S6
    / "configs/"
      "stage6_evaluator_reference_contract.json"
)

ROAD = (
    S6
    / "artifacts/block68/"
      "frozen_road_roi_definition.json"
)


class Block68EvaluatorReferenceContractTest(
    unittest.TestCase
):

    def test_01_horizons_exact(
        self,
    ):

        self.assertEqual(
            FROZEN_HORIZONS_S,
            (
                0.1,
                0.3,
                0.5,
                1.0,
            ),
        )

    def test_02_classes_exact(
        self,
    ):

        self.assertEqual(
            FROZEN_ADB_ACTOR_CLASSES,
            (
                "TYPE_VEHICLE",
                "TYPE_PEDESTRIAN",
                "TYPE_CYCLIST",
            ),
        )

    def test_03_road_roi_is_full_grid(
        self,
    ):

        roi = frozen_road_roi(
            (
                5,
                7,
            )
        )

        self.assertEqual(
            roi.shape,
            (
                5,
                7,
            ),
        )

        self.assertEqual(
            roi.dtype,
            np.bool_,
        )

        self.assertTrue(
            bool(
                np.all(
                    roi
                )
            )
        )

    def test_04_constructed_oracle_is_union(
        self,
    ):

        first = np.zeros(
            (
                4,
                3,
                4,
            ),
            dtype=bool,
        )

        second = np.zeros_like(
            first
        )

        first[
            0,
            1,
            1,
        ] = True

        second[
            0,
            2,
            2,
        ] = True

        oracle = (
            aggregate_constructed_oracle_supports(
                [
                    first,
                    second,
                ],
                grid_shape=(
                    3,
                    4,
                ),
            )
        )

        self.assertTrue(
            oracle[
                0,
                1,
                1,
            ]
        )

        self.assertTrue(
            oracle[
                0,
                2,
                2,
            ]
        )

        self.assertEqual(
            int(
                np.count_nonzero(
                    oracle
                )
            ),
            2,
        )

    def test_05_empty_oracle_is_zero(
        self,
    ):

        oracle = (
            aggregate_constructed_oracle_supports(
                [],
                grid_shape=(
                    2,
                    3,
                ),
            )
        )

        self.assertFalse(
            bool(
                np.any(
                    oracle
                )
            )
        )

    def test_06_final_output_is_post_actuation(
        self,
    ):

        self.assertEqual(
            FROZEN_EVALUATOR_REFERENCE_SEMANTICS
            .final_metric_intensity,
            "RateLimitedSchedule.illumination",
        )

        output = np.ones(
            (
                4,
                2,
                3,
            ),
            dtype=np.float64,
        )

        validated = (
            validate_final_metric_illumination(
                output,
                grid_shape=(
                    2,
                    3,
                ),
            )
        )

        np.testing.assert_array_equal(
            output,
            validated,
        )

    def test_07_contract_and_artifact_frozen(
        self,
    ):

        contract = json.loads(
            CONTRACT.read_text(
                encoding="utf-8"
            )
        )

        road = json.loads(
            ROAD.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            contract[
                "status"
            ],
            "FROZEN_EVALUATOR_REFERENCE_CONTRACT",
        )

        self.assertEqual(
            contract[
                "road_ROI"
            ][
                "rule"
            ],
            (
                "all_valid_cells_of_frozen_"
                "theta_range_headlamp_actuator_grid"
            ),
        )

        self.assertFalse(
            contract[
                "constructed_oracle"
            ][
                "controller_input"
            ]
        )

        self.assertFalse(
            contract[
                "constructed_oracle"
            ][
                "measured_ADB_ground_truth"
            ]
        )

        self.assertEqual(
            road[
                "status"
            ],
            "FROZEN_ROAD_ROI_DEFINITION",
        )


if __name__ == "__main__":
    unittest.main()
