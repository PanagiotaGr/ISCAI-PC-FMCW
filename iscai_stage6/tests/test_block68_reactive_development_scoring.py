from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from iscai_stage6.adb.reactive_development_scoring import (
    REACTIVE_EVALUATION_HORIZONS_S,
    binary_dim_support,
    hold_current_reactive_schedule,
    scientific_input_contract,
    validate_supported_reactive_metric_values,
)


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

CONTRACT = (
    S6
    / "configs/"
      "stage6_reactive_development_scoring_contract.json"
)


class Block68ReactiveDevelopmentScoringTests(
    unittest.TestCase
):

    def test_01_horizons_are_exact(self):

        self.assertEqual(
            REACTIVE_EVALUATION_HORIZONS_S,
            (
                0.1,
                0.3,
                0.5,
                1.0,
            ),
        )

    def test_02_hold_current_replication(self):

        current = np.asarray(
            [
                [1.0, 0.8],
                [0.5, 1.0],
            ],
            dtype=np.float64,
        )

        schedule = (
            hold_current_reactive_schedule(
                current
            )
        )

        self.assertEqual(
            schedule.shape,
            (
                4,
                2,
                2,
            ),
        )

        for index in range(4):
            np.testing.assert_array_equal(
                schedule[index],
                current,
            )

    def test_03_output_is_independent_copy(self):

        current = np.ones(
            (
                2,
                3,
            ),
            dtype=np.float64,
        )

        schedule = (
            hold_current_reactive_schedule(
                current
            )
        )

        schedule[
            0,
            0,
            0,
        ] = 0.0

        self.assertEqual(
            current[
                0,
                0,
            ],
            1.0,
        )

        self.assertEqual(
            schedule[
                1,
                0,
                0,
            ],
            1.0,
        )

    def test_04_invalid_current_map_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            hold_current_reactive_schedule(
                np.ones(
                    (
                        2,
                        2,
                        2,
                    )
                )
            )

        bad = np.ones(
            (
                2,
                2,
            )
        )

        bad[
            0,
            0,
        ] = 1.1

        with self.assertRaises(
            ValueError
        ):
            hold_current_reactive_schedule(
                bad
            )

    def test_05_nonfrozen_horizons_rejected(self):

        current = np.ones(
            (
                2,
                2,
            )
        )

        with self.assertRaises(
            ValueError
        ):
            hold_current_reactive_schedule(
                current,
                horizons_s=(
                    0.1,
                    0.3,
                    0.5,
                    0.9,
                ),
            )

    def test_06_dim_support_is_strict_less_than_one(self):

        schedule = np.asarray(
            [
                [
                    [1.0, 0.999],
                ],
            ]
            *
            4,
            dtype=np.float64,
        )

        support = binary_dim_support(
            schedule
        )

        self.assertFalse(
            bool(
                support[
                    0,
                    0,
                    0,
                ]
            )
        )

        self.assertTrue(
            bool(
                support[
                    0,
                    0,
                    1,
                ]
            )
        )

        values = (
            validate_supported_reactive_metric_values(
                [
                    0.1,
                    0.2,
                    0.3,
                ],
                metric_name="synthetic",
            )
        )

        self.assertEqual(
            values.shape,
            (3,),
        )

    def test_07_contract_forbids_future_refresh(self):

        contract = json.loads(
            CONTRACT.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            contract[
                "status"
            ],
            "FROZEN_PREOUTCOME_REACTIVE_SCORING_SEMANTICS",
        )

        scoring = contract[
            "original_reactive_ADB"
        ]

        self.assertFalse(
            scoring[
                "future_reobservation"
            ]
        )

        self.assertFalse(
            scoring[
                "future_controller_refresh"
            ]
        )

        self.assertFalse(
            scoring[
                "future_truth_controller_input"
            ]
        )

        self.assertEqual(
            scoring[
                "horizons_s"
            ],
            [
                0.1,
                0.3,
                0.5,
                1.0,
            ],
        )

        parameters = tuple(
            inspect.signature(
                hold_current_reactive_schedule
            ).parameters.keys()
        )

        self.assertEqual(
            parameters,
            (
                "current_illumination",
                "horizons_s",
            ),
        )

        runtime_contract = (
            scientific_input_contract()
        )

        self.assertFalse(
            runtime_contract[
                "controller_future_input"
            ]
        )


if __name__ == "__main__":
    unittest.main()
