from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

S6 = (
    ROOT
    / "iscai_stage6"
)

EXTRACTION = (
    S6
    / "reports/"
      "block68_exact_oracle_roi_binding_extract.json"
)

I_FINAL_BIND = (
    S6
    / "reports/"
      "block68_exact_evaluator_binding_repair.json"
)

RECON = (
    S6
    / "reports/"
      "block68_preoutcome_metric_reconciliation.json"
)

PROTOCOL = (
    S6
    / "configs/"
      "stage6_metric_freeze_protocol.json"
)

METRICS = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

INTERFACE = (
    S6
    / "src/iscai_stage6/adb/"
      "baseline_metric_contract.py"
)

CLASS_POLICY = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

DETERMINISTIC = (
    S6
    / "src/iscai_stage6/adb/"
      "deterministic_predictive.py"
)

STOCHASTIC = (
    S6
    / "src/iscai_stage6/adb/"
      "probabilistic_full_box.py"
)

OCCUPANCY = (
    S6
    / "src/iscai_stage6/adb/"
      "probabilistic_occupancy.py"
)

MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "evaluator_reference.py"
)

CONTRACT = (
    S6
    / "configs/"
      "stage6_evaluator_reference_contract.json"
)

ROAD_ROI_ARTIFACT = (
    S6
    / "artifacts/block68/"
      "frozen_road_roi_definition.json"
)

TEST = (
    S6
    / "tests/"
      "test_block68_evaluator_reference_contract.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_evaluator_reference_contract_freeze.json"
)


EXPECTED_I_FINAL_BIND_SHA = (
    "1505c780423efbe27735c3729b95a18c"
    "3da2093f49826bd96cbfad8fdce5d5f1"
)

EXPECTED_RECON_SHA = (
    "077a89fa5163f8862a3a5b7489d98929"
    "d9ae3e160ed9e030eb75f42a91d48f9f"
)

EXPECTED_PROTOCOL_SHA = (
    "409fbb2785e4ff13f29fdff96c91d608"
    "c245e009fcbb86ce57099d6a4e1815c9"
)

EXPECTED_METRICS_SHA = (
    "bf8358b5a7edfe40ffac2718ca7cd5af"
    "6e7b5fdb3ff8a2823caad6d0f40f057a"
)

EXPECTED_INTERFACE_SHA = (
    "39518597079f65d556ee8f0fd3b00ce5"
    "25319623c9eb8a83f6e3c608954213e9"
)

EXPECTED_PRE_TESTS = 202

NEW_TESTS = 7

MIN_FREE_GIB = 250.0


# ============================================================
# Frozen scientific semantics
# ============================================================

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

ACTOR_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)


# ============================================================
# New evaluator-reference helper module
# ============================================================

MODULE_SOURCE = r'''from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FROZEN_HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

FROZEN_ADB_ACTOR_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)


@dataclass(frozen=True)
class EvaluatorReferenceSemantics:
    oracle_type: str
    measured_adb_ground_truth: bool
    controller_future_truth_access: bool
    parameter_tuning_future_truth_access: bool
    oracle_grid_semantics: str
    road_roi_semantics: str
    final_metric_intensity: str


FROZEN_EVALUATOR_REFERENCE_SEMANTICS = (
    EvaluatorReferenceSemantics(
        oracle_type=(
            "constructed_future_reference"
        ),

        measured_adb_ground_truth=False,

        controller_future_truth_access=False,

        parameter_tuning_future_truth_access=False,

        oracle_grid_semantics=(
            "same_frozen_theta_range_headlamp_"
            "actuator_grid_as_controller"
        ),

        road_roi_semantics=(
            "all_valid_cells_of_frozen_theta_range_"
            "headlamp_actuator_grid"
        ),

        final_metric_intensity=(
            "RateLimitedSchedule.illumination"
        ),
    )
)


def frozen_road_roi(
    grid_shape,
) -> np.ndarray:
    """
    Frozen Block6.8 evaluator implementation choice.

    The Stage-6 actuator grid is the frozen theta x range
    headlamp road-domain grid.

    The road-illumination-retention ROI is therefore every
    valid cell of that frozen actuator grid.

    This ROI:
      - is deterministic,
      - does not depend on actors,
      - does not depend on future truth,
      - does not depend on development outcomes,
      - does not alter the controller.
    """

    shape = tuple(
        int(value)
        for value in grid_shape
    )

    if len(shape) != 2:
        raise ValueError(
            "grid_shape must be "
            "(theta_cells, range_cells)"
        )

    if (
        shape[0] <= 0
        or
        shape[1] <= 0
    ):
        raise ValueError(
            "grid dimensions must be > 0"
        )

    return np.ones(
        shape,
        dtype=np.bool_,
    )


def _actor_support_array(
    value,
    *,
    grid_shape,
) -> np.ndarray:

    array = np.asarray(
        value
    )

    if (
        array.dtype
        !=
        np.bool_
    ):
        raise TypeError(
            "actor support masks must be boolean"
        )

    expected = (
        len(
            FROZEN_HORIZONS_S
        ),
        int(
            grid_shape[0]
        ),
        int(
            grid_shape[1]
        ),
    )

    if (
        tuple(
            array.shape
        )
        !=
        expected
    ):
        raise ValueError(
            "actor support shape must be "
            "[4, theta, range]"
        )

    return array


def aggregate_constructed_oracle_supports(
    actor_supports: Iterable[np.ndarray],
    *,
    grid_shape,
) -> np.ndarray:
    """
    O_tau is the union of evaluator-only future actor
    shadow supports on the frozen controller grid.

    Future truth is consumed only by the evaluator after
    controller output has already been produced.
    """

    shape = tuple(
        int(value)
        for value in grid_shape
    )

    if len(shape) != 2:
        raise ValueError(
            "grid_shape must be "
            "(theta_cells, range_cells)"
        )

    result = np.zeros(
        (
            len(
                FROZEN_HORIZONS_S
            ),
            shape[0],
            shape[1],
        ),
        dtype=np.bool_,
    )

    for support in actor_supports:

        result |= _actor_support_array(
            support,
            grid_shape=shape,
        )

    return result


def aggregate_class_future_region(
    actor_regions: Iterable[np.ndarray],
    *,
    grid_shape,
) -> np.ndarray:
    """
    Union of evaluator-only projected future-GT full-box
    footprints for one actor class.

    Used for pedestrian/cyclist visibility support and,
    with the vehicle-specific constructed surrogate,
    evaluator protection regions.
    """

    return aggregate_constructed_oracle_supports(
        actor_regions,
        grid_shape=grid_shape,
    )


def validate_final_metric_illumination(
    illumination,
    *,
    grid_shape,
) -> np.ndarray:
    """
    Validate the frozen evaluator-facing post-actuation output:
        RateLimitedSchedule.illumination
    """

    array = np.asarray(
        illumination,
        dtype=np.float64,
    )

    expected = (
        len(
            FROZEN_HORIZONS_S
        ),
        int(
            grid_shape[0]
        ),
        int(
            grid_shape[1]
        ),
    )

    if (
        tuple(
            array.shape
        )
        !=
        expected
    ):
        raise ValueError(
            "final illumination must have "
            "shape [4, theta, range]"
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "final illumination must be finite"
        )

    if (
        np.any(
            array < 0.0
        )
        or
        np.any(
            array > 1.0
        )
    ):
        raise ValueError(
            "final illumination must lie in [0,1]"
        )

    return array
'''


# ============================================================
# Tests
# ============================================================

TEST_SOURCE = r'''from __future__ import annotations

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
'''


# ============================================================
# Helpers
# ============================================================

def sha256_bytes(
    data: bytes,
) -> str:

    return sha256(
        data
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:

    digest = sha256()

    with path.open(
        "rb"
    ) as stream:

        while True:

            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def require(
    condition,
    message,
):

    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


def guarded_json(
    path: Path,
):

    require(
        "formal"
        not in
        str(
            path.resolve()
        ).lower(),
        (
            "FORMAL CONTENT ACCESS FORBIDDEN: "
            f"{path}"
        ),
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def canonical_json_bytes(
    value,
) -> bytes:

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def atomic_write(
    path: Path,
    data: bytes,
):

    temporary = (
        path.with_suffix(
            path.suffix
            +
            ".tmp"
        )
    )

    temporary.write_bytes(
        data
    )

    os.replace(
        temporary,
        path,
    )


def run_tests(
    pattern,
):

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            pattern,
        ],
        cwd=str(
            S6
        ),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    count = None

    for line in (
        process.stdout.splitlines()
    ):

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:

            count = int(
                match.group(1)
            )

    return (
        process.returncode,
        count,
        process.stdout,
    )


def tail(
    text,
    count=120,
):

    return "\n".join(
        text.splitlines()[
            -count:
        ]
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART 2/2"
    )
    print(
        "EVALUATOR REFERENCE CONTRACT FREEZE"
    )
    print(
        "PRE-DEVELOPMENT-OUTCOME IMPLEMENTATION COMPLETION"
    )
    print(
        "============================================================"
    )

    required = (
        EXTRACTION,
        I_FINAL_BIND,
        RECON,
        PROTOCOL,
        METRICS,
        INTERFACE,
        CLASS_POLICY,
        DETERMINISTIC,
        STOCHASTIC,
        OCCUPANCY,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing evaluator-contract dependency: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Existing successful freeze / idempotence
    # ========================================================

    if REPORT.is_file():

        existing = guarded_json(
            REPORT
        )

        if (
            existing.get(
                "status"
            )
            ==
            "PASS_EVALUATOR_REFERENCE_CONTRACT_FROZEN"
        ):

            print()
            print(
                "===== EXISTING FREEZE DETECTED ====="
            )

            for key, path in (
                (
                    "contract",
                    CONTRACT,
                ),
                (
                    "module",
                    MODULE,
                ),
                (
                    "road_roi_artifact",
                    ROAD_ROI_ARTIFACT,
                ),
                (
                    "test",
                    TEST,
                ),
            ):

                expected = existing[
                    "frozen_outputs"
                ][
                    key
                ][
                    "sha256"
                ]

                require(
                    path.is_file(),
                    (
                        f"Frozen output missing: "
                        f"{path}"
                    ),
                )

                require(
                    sha256_file(
                        path
                    )
                    ==
                    expected,
                    (
                        f"Frozen output changed: "
                        f"{path}"
                    ),
                )

            rc, tests, output = (
                run_tests(
                    "test_*.py"
                )
            )

            require(
                rc == 0,
                (
                    "Existing freeze regression failed:\n"
                    +
                    tail(
                        output
                    )
                ),
            )

            require(
                tests
                ==
                existing[
                    "regression"
                ][
                    "post"
                ],
                (
                    "Existing freeze regression "
                    "count changed."
                ),
            )

            print(
                "contract           = EXACT PASS"
            )

            print(
                "road ROI artifact  = EXACT PASS"
            )

            print(
                "helper module      = EXACT PASS"
            )

            print(
                "Stage6 regression  =",
                f"{tests} / {tests} PASS",
            )

            print(
                "STATUS = "
                "PASS_EVALUATOR_REFERENCE_CONTRACT_FROZEN"
            )

            print(
                "no rewrite performed = YES"
            )

            print(
                "terminal remains open = YES"
            )

            return

    # ========================================================
    # B. Upstream seals
    # ========================================================

    print()
    print(
        "===== A. UPSTREAM FROZEN SEALS ====="
    )

    require(
        sha256_file(
            I_FINAL_BIND
        )
        ==
        EXPECTED_I_FINAL_BIND_SHA,
        (
            "I_final binding report changed."
        ),
    )

    require(
        sha256_file(
            RECON
        )
        ==
        EXPECTED_RECON_SHA,
        (
            "Metric reconciliation changed."
        ),
    )

    require(
        sha256_file(
            PROTOCOL
        )
        ==
        EXPECTED_PROTOCOL_SHA,
        (
            "Metric protocol changed."
        ),
    )

    require(
        sha256_file(
            METRICS
        )
        ==
        EXPECTED_METRICS_SHA,
        (
            "Metric runtime changed."
        ),
    )

    require(
        sha256_file(
            INTERFACE
        )
        ==
        EXPECTED_INTERFACE_SHA,
        (
            "Metric interface changed."
        ),
    )

    extraction = guarded_json(
        EXTRACTION
    )

    require(
        extraction.get(
            "status"
        )
        ==
        "PASS_EXACT_EVALUATOR_REFERENCE_SOURCE_EXTRACTED",
        (
            "Exact evaluator-reference "
            "extraction is not PASS."
        ),
    )

    readiness = extraction.get(
        "readiness",
        {}
    )

    mandatory_proven = (
        "deterministic_future_full_box_API",
        "stochastic_full_box_API",
        "occupancy_raster_API",
        "oracle_future_truth_source",
        "oracle_mask_source",
        "pedestrian_future_full_box_evidence",
        "cyclist_future_full_box_evidence",
        "vehicle_surrogate_evidence",
        "road_roi_code_evidence",
    )

    for key in mandatory_proven:

        require(
            readiness.get(
                key
            )
            is True,
            (
                "Evaluator-reference evidence "
                f"not proven: {key}"
            ),
        )

    require(
        readiness.get(
            "road_roi_artifact_evidence"
        )
        is False,
        (
            "Expected pre-freeze road-ROI "
            "artifact gap was not reproduced."
        ),
    )

    print(
        "I_final binding              = EXACT PASS"
    )

    print(
        "full-box APIs                = PROVEN"
    )

    print(
        "oracle/future truth routes   = PROVEN"
    )

    print(
        "VRU evaluator regions        = PROVEN"
    )

    print(
        "vehicle surrogate            = PROVEN"
    )

    print(
        "road ROI code evidence       = PROVEN"
    )

    print(
        "pre-existing road ROI artifact = ABSENT CONFIRMED"
    )

    # ========================================================
    # C. Pre-write scientific boundary
    # ========================================================

    print()
    print(
        "===== B. PRE-OUTCOME SCIENTIFIC BOUNDARY ====="
    )

    print(
        "development performance metrics = NOT READ"
    )

    print(
        "numeric acceptance bounds        = NOT SELECTED"
    )

    print(
        "formal content opened            = NO"
    )

    print(
        "policy/controller modification   = NO"
    )

    print(
        "road ROI completion source       = PRE-OUTCOME RULE"
    )

    # ========================================================
    # D. Pre-freeze regression
    # ========================================================

    print()
    print(
        "===== C. PRE-FREEZE REGRESSION ====="
    )

    rc_before, tests_before, output_before = (
        run_tests(
            "test_*.py"
        )
    )

    require(
        rc_before == 0,
        (
            "Pre-freeze Stage6 regression failed:\n"
            +
            tail(
                output_before
            )
        ),
    )

    require(
        tests_before
        ==
        EXPECTED_PRE_TESTS,
        (
            "Expected "
            f"{EXPECTED_PRE_TESTS} "
            "pre-freeze tests; got "
            f"{tests_before}."
        ),
    )

    print(
        "pre-freeze regression =",
        f"{tests_before} / {tests_before} PASS",
    )

    # ========================================================
    # E. Build exact contract
    # ========================================================

    print()
    print(
        "===== D. BUILD EVALUATOR REFERENCE CONTRACT ====="
    )

    extraction_sha = (
        sha256_file(
            EXTRACTION
        )
    )

    exact_symbols = (
        extraction.get(
            "exact_symbols",
            {}
        )
    )

    contract = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "status":
            "FROZEN_EVALUATOR_REFERENCE_CONTRACT",

        "freeze_timing":
            "before_development_metric_outcomes",

        "authority": {
            "metric_semantics":
                "Block6.6_preoutcome_preregistration",

            "metric_reconciliation_sha256":
                EXPECTED_RECON_SHA,

            "exact_reference_extraction_sha256":
                extraction_sha,

            "I_final_binding_sha256":
                EXPECTED_I_FINAL_BIND_SHA,
        },

        "horizons_s":
            list(
                HORIZONS_S
            ),

        "actor_classes": [
            "TYPE_VEHICLE",
            "TYPE_PEDESTRIAN",
            "TYPE_CYCLIST",
        ],

        "controller_output": {
            "metric_input":
                "RateLimitedSchedule.illumination",

            "post_temporal_smoothing":
                True,

            "post_actuation_rate_limit":
                True,

            "raw_class_aware_map_used_for_metrics":
                False,
        },

        "constructed_oracle": {
            "type":
                "CONSTRUCTED_EVALUATOR_REFERENCE",

            "measured_ADB_ground_truth":
                False,

            "future_truth_access":
                "evaluator_after_controller_decision_only",

            "controller_input":
                False,

            "parameter_tuning_input":
                False,

            "actor_population":
                (
                    "all evaluator-valid vehicle, "
                    "pedestrian and cyclist actors"
                ),

            "actor_support":
                (
                    "future-GT full-box projected "
                    "shadow/footprint support on "
                    "frozen theta-range headlamp grid"
                ),

            "aggregation":
                (
                    "O_tau = union of per-actor "
                    "future supports"
                ),

            "grid":
                "same_frozen_theta_range_grid_as_controller",

            "horizons_s":
                list(
                    HORIZONS_S
                ),
        },

        "vehicle_reference_region": {
            "role":
                "evaluator_only",

            "semantics":
                (
                    "constructed vehicle "
                    "front-upper/rear-upper surrogate "
                    "reference region"
                ),

            "future_truth_controller_input":
                False,

            "existing_evidence":
                "PROVEN",
        },

        "pedestrian_visibility_region": {
            "role":
                "evaluator_only",

            "semantics":
                (
                    "future-GT pedestrian projected "
                    "full-box footprint cells"
                ),

            "future_truth_controller_input":
                False,
        },

        "cyclist_visibility_region": {
            "role":
                "evaluator_only",

            "semantics":
                (
                    "future-GT cyclist projected "
                    "full-box footprint cells"
                ),

            "future_truth_controller_input":
                False,
        },

        "road_ROI": {
            "status":
                "FROZEN_PRE_DEVELOPMENT_OUTCOMES",

            "rule":
                (
                    "all_valid_cells_of_frozen_"
                    "theta_range_headlamp_actuator_grid"
                ),

            "mask_value_on_valid_grid":
                True,

            "actor_dependent":
                False,

            "future_truth_dependent":
                False,

            "map_dependent":
                False,

            "development_outcome_dependent":
                False,

            "formal_outcome_dependent":
                False,

            "justification":
                (
                    "Block6.6 preregistered a frozen "
                    "road ROI but no narrower numerical "
                    "ROI artifact existed. The Stage6 "
                    "theta-range headlamp actuator grid "
                    "is the road-domain illumination "
                    "evaluation grid, so all valid cells "
                    "are frozen as the road ROI before "
                    "any development metric outcome."
                ),
        },

        "metric_supports": {
            "mask_IoU":
                "candidate_D_vs_constructed_O",

            "vehicle_shadow_zone_violation":
                "constructed_future_vehicle_required_region",

            "glare_risk_exposure":
                "constructed_vehicle_surrogate_region",

            "over_masking_area":
                "candidate_D_minus_O_over_full_actuator_grid",

            "road_illumination_retention":
                "I_final_over_frozen_road_ROI",

            "pedestrian_visibility_proxy":
                "I_final_over_future_GT_pedestrian_full_box",

            "cyclist_visibility_proxy":
                "I_final_over_future_GT_cyclist_full_box",

            "false_dimming":
                "candidate_D_minus_O_over_candidate_D",

            "temporal_smoothness":
                "post_actuation_I_final_change_rate",

            "flicker_change_rate":
                "post_actuation_exact_cell_change_fraction",

            "energy_consumption":
                "mean_post_actuation_I_final_full_horizon_grid",

            "actuation_latency":
                (
                    "loaded_P_occ_or_current_geometry_"
                    "to_final_four_horizon_schedule_seconds"
                ),
        },

        "exact_scientific_sources": {
            "deterministic_predictive": {
                "path":
                    str(
                        DETERMINISTIC
                    ),

                "sha256":
                    sha256_file(
                        DETERMINISTIC
                    ),
            },

            "probabilistic_full_box": {
                "path":
                    str(
                        STOCHASTIC
                    ),

                "sha256":
                    sha256_file(
                        STOCHASTIC
                    ),
            },

            "probabilistic_occupancy": {
                "path":
                    str(
                        OCCUPANCY
                    ),

                "sha256":
                    sha256_file(
                        OCCUPANCY
                    ),
            },

            "class_aware_policy": {
                "path":
                    str(
                        CLASS_POLICY
                    ),

                "sha256":
                    sha256_file(
                        CLASS_POLICY
                    ),
            },

            "exact_symbols":
                exact_symbols,
        },

        "data_use_boundary": {
            "future_truth_allowed_for_controller":
                False,

            "future_truth_allowed_for_evaluator":
                True,

            "future_truth_access_time":
                "after_controller_decision",

            "constructed_oracle_used_for_tuning":
                False,

            "formal_data_allowed_now":
                False,
        },

        "formal_evaluation_allowed":
            False,

        "next":
            (
                "development-only ADB metric execution "
                "and numeric non-inferiority bound freeze"
            ),
    }

    road_artifact = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "status":
            "FROZEN_ROAD_ROI_DEFINITION",

        "freeze_timing":
            "before_development_metric_outcomes",

        "representation":
            "deterministic_runtime_mask_rule",

        "rule":
            (
                "road_roi = np.ones("
                "(theta_cells, range_cells), dtype=bool)"
            ),

        "domain":
            "frozen_theta_range_headlamp_actuator_grid",

        "all_valid_grid_cells_included":
            True,

        "actor_dependent":
            False,

        "future_truth_dependent":
            False,

        "development_outcome_dependent":
            False,

        "formal_outcome_dependent":
            False,

        "concrete_shape":
            (
                "bound at evaluator runtime to the "
                "already-frozen controller grid; "
                "no data-dependent shape selection"
            ),
    }

    contract_bytes = (
        canonical_json_bytes(
            contract
        )
    )

    road_bytes = (
        canonical_json_bytes(
            road_artifact
        )
    )

    module_bytes = (
        MODULE_SOURCE.encode(
            "utf-8"
        )
    )

    test_bytes = (
        TEST_SOURCE.encode(
            "utf-8"
        )
    )

    compile(
        MODULE_SOURCE,
        str(
            MODULE
        ),
        "exec",
    )

    compile(
        TEST_SOURCE,
        str(
            TEST
        ),
        "exec",
    )

    json.loads(
        contract_bytes.decode(
            "utf-8"
        )
    )

    json.loads(
        road_bytes.decode(
            "utf-8"
        )
    )

    print(
        "contract JSON syntax       = PASS"
    )

    print(
        "road ROI artifact syntax   = PASS"
    )

    print(
        "helper Python syntax        = PASS"
    )

    print(
        "road ROI                    = ALL VALID GRID CELLS"
    )

    print(
        "road ROI outcome dependence = NONE"
    )

    # ========================================================
    # F. New-file-only transactional write
    # ========================================================

    print()
    print(
        "===== E. TRANSACTIONAL FREEZE ====="
    )

    targets = (
        CONTRACT,
        ROAD_ROI_ARTIFACT,
        MODULE,
        TEST,
    )

    preexisting = [
        str(
            path
        )
        for path in targets
        if path.exists()
    ]

    require(
        not preexisting,
        (
            "Unexpected pre-existing evaluator "
            "freeze files without PASS report: "
            +
            repr(
                preexisting
            )
        ),
    )

    written = []

    try:

        for path, data in (
            (
                CONTRACT,
                contract_bytes,
            ),
            (
                ROAD_ROI_ARTIFACT,
                road_bytes,
            ),
            (
                MODULE,
                module_bytes,
            ),
            (
                TEST,
                test_bytes,
            ),
        ):

            atomic_write(
                path,
                data,
            )

            written.append(
                path
            )

        print(
            "evaluator contract = WRITTEN"
        )

        print(
            "road ROI artifact  = WRITTEN"
        )

        print(
            "helper module      = WRITTEN"
        )

        print(
            "tests              = WRITTEN"
        )

        # ====================================================
        # G. Targeted tests
        # ====================================================

        print()
        print(
            "===== F. TARGETED CONTRACT TESTS ====="
        )

        rc_target, n_target, out_target = (
            run_tests(
                "test_block68_evaluator_reference_contract.py"
            )
        )

        require(
            rc_target == 0,
            (
                "Evaluator contract tests failed:\n"
                +
                tail(
                    out_target
                )
            ),
        )

        require(
            n_target
            ==
            NEW_TESTS,
            (
                "Expected "
                f"{NEW_TESTS} targeted tests; "
                f"got {n_target}."
            ),
        )

        print(
            "targeted tests =",
            f"{n_target} / {n_target} PASS",
        )

        # ====================================================
        # H. Full regression
        # ====================================================

        print()
        print(
            "===== G. FULL STAGE6 REGRESSION ====="
        )

        rc_after, tests_after, out_after = (
            run_tests(
                "test_*.py"
            )
        )

        require(
            rc_after == 0,
            (
                "Post-freeze regression failed:\n"
                +
                tail(
                    out_after
                )
            ),
        )

        expected_after = (
            tests_before
            +
            NEW_TESTS
        )

        require(
            tests_after
            ==
            expected_after,
            (
                "Unexpected post-freeze "
                f"test count {tests_after}; "
                f"expected {expected_after}."
            ),
        )

        print(
            "full Stage6 regression =",
            f"{tests_after} / {tests_after} PASS",
        )

    except BaseException:

        print()
        print(
            "FREEZE FAILURE -> REMOVING NEW FILES"
        )

        for path in reversed(
            written
        ):

            try:
                if path.exists():
                    path.unlink()

            except Exception:
                pass

        rc_rollback, n_rollback, rollback_output = (
            run_tests(
                "test_*.py"
            )
        )

        print(
            "rollback regression =",
            n_rollback,
            "/",
            n_rollback,
            (
                "PASS"
                if rc_rollback == 0
                else "FAIL"
            ),
        )

        require(
            rc_rollback == 0,
            (
                "Rollback regression failed:\n"
                +
                tail(
                    rollback_output
                )
            ),
        )

        require(
            n_rollback
            ==
            EXPECTED_PRE_TESTS,
            (
                "Rollback test count mismatch."
            ),
        )

        raise

    # ========================================================
    # I. Exact semantic readback
    # ========================================================

    print()
    print(
        "===== H. EXACT SEMANTIC READBACK ====="
    )

    frozen_contract = guarded_json(
        CONTRACT
    )

    frozen_road = guarded_json(
        ROAD_ROI_ARTIFACT
    )

    require(
        frozen_contract[
            "road_ROI"
        ][
            "rule"
        ]
        ==
        (
            "all_valid_cells_of_frozen_"
            "theta_range_headlamp_actuator_grid"
        ),
        (
            "Road ROI rule changed."
        ),
    )

    require(
        frozen_contract[
            "constructed_oracle"
        ][
            "controller_input"
        ]
        is False,
        (
            "Oracle leaked into controller."
        ),
    )

    require(
        frozen_contract[
            "constructed_oracle"
        ][
            "measured_ADB_ground_truth"
        ]
        is False,
        (
            "Constructed oracle mislabeled "
            "as measured ground truth."
        ),
    )

    require(
        frozen_contract[
            "controller_output"
        ][
            "metric_input"
        ]
        ==
        "RateLimitedSchedule.illumination",
        (
            "I_final contract changed."
        ),
    )

    require(
        frozen_road[
            "all_valid_grid_cells_included"
        ]
        is True,
        (
            "Road ROI artifact semantics changed."
        ),
    )

    print(
        "constructed oracle = EVALUATOR ONLY"
    )

    print(
        "measured ADB GT     = NO"
    )

    print(
        "future GT controller= NO"
    )

    print(
        "I_final             = POST-ACTUATION"
    )

    print(
        "road ROI            = FULL VALID ACTUATOR GRID"
    )

    print(
        "road ROI artifact   = NOW EXPLICITLY FROZEN"
    )

    # ========================================================
    # J. Upstream immutability
    # ========================================================

    print()
    print(
        "===== I. UPSTREAM IMMUTABILITY ====="
    )

    require(
        sha256_file(
            I_FINAL_BIND
        )
        ==
        EXPECTED_I_FINAL_BIND_SHA,
        (
            "I_final binder changed."
        ),
    )

    require(
        sha256_file(
            RECON
        )
        ==
        EXPECTED_RECON_SHA,
        (
            "Metric reconciliation changed."
        ),
    )

    require(
        sha256_file(
            PROTOCOL
        )
        ==
        EXPECTED_PROTOCOL_SHA,
        (
            "Metric protocol changed."
        ),
    )

    require(
        sha256_file(
            METRICS
        )
        ==
        EXPECTED_METRICS_SHA,
        (
            "Metric runtime changed."
        ),
    )

    require(
        sha256_file(
            INTERFACE
        )
        ==
        EXPECTED_INTERFACE_SHA,
        (
            "Metric interface changed."
        ),
    )

    print(
        "controller runtime       = UNCHANGED"
    )

    print(
        "metric runtime           = UNCHANGED"
    )

    print(
        "metric authority         = UNCHANGED"
    )

    print(
        "previous evidence reports= UNCHANGED"
    )

    # ========================================================
    # K. Storage
    # ========================================================

    print()
    print(
        "===== J. STORAGE ====="
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    require(
        free_gib
        >=
        MIN_FREE_GIB,
        (
            "250-GiB reserve violated."
        ),
    )

    print(
        "free GiB        =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve = PASS"
    )

    # ========================================================
    # L. Freeze report
    # ========================================================

    print()
    print(
        "===== K. FREEZE REPORT ====="
    )

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "2/2_evaluator_reference_contract_freeze",

        "status":
            "PASS_EVALUATOR_REFERENCE_CONTRACT_FROZEN",

        "scientific_change": {
            "controller":
                False,

            "predictor":
                False,

            "class_policy":
                False,

            "metric_formula":
                False,

            "new_preoutcome_evaluator_definition":
                (
                    "road_ROI_all_valid_frozen_"
                    "theta_range_grid_cells"
                ),
        },

        "road_ROI_resolution": {
            "previous_artifact_evidence":
                "NOT_PROVEN",

            "resolution":
                "EXPLICIT_PREOUTCOME_FREEZE",

            "rule":
                (
                    "all_valid_cells_of_frozen_"
                    "theta_range_headlamp_actuator_grid"
                ),

            "development_outcomes_seen_before_resolution":
                False,

            "formal_outcomes_seen_before_resolution":
                False,
        },

        "constructed_oracle": {
            "type":
                "CONSTRUCTED_EVALUATOR_REFERENCE",

            "future_truth_controller_input":
                False,

            "measured_ADB_ground_truth":
                False,

            "future_GT_read_after_decision_only":
                True,
        },

        "frozen_outputs": {
            "contract": {
                "path":
                    str(
                        CONTRACT
                    ),

                "sha256":
                    sha256_file(
                        CONTRACT
                    ),
            },

            "module": {
                "path":
                    str(
                        MODULE
                    ),

                "sha256":
                    sha256_file(
                        MODULE
                    ),
            },

            "road_roi_artifact": {
                "path":
                    str(
                        ROAD_ROI_ARTIFACT
                    ),

                "sha256":
                    sha256_file(
                        ROAD_ROI_ARTIFACT
                    ),
            },

            "test": {
                "path":
                    str(
                        TEST
                    ),

                "sha256":
                    sha256_file(
                        TEST
                    ),
            },
        },

        "upstream": {
            "reference_extraction": {
                "path":
                    str(
                        EXTRACTION
                    ),

                "sha256":
                    extraction_sha,
            },

            "I_final_binding_sha256":
                EXPECTED_I_FINAL_BIND_SHA,

            "metric_reconciliation_sha256":
                EXPECTED_RECON_SHA,

            "metric_protocol_sha256":
                EXPECTED_PROTOCOL_SHA,

            "metric_runtime_sha256":
                EXPECTED_METRICS_SHA,

            "metric_interface_sha256":
                EXPECTED_INTERFACE_SHA,
        },

        "scientific_boundary": {
            "development_performance_metrics_computed":
                False,

            "numeric_noninferiority_bounds_selected":
                False,

            "policy_tuning":
                False,

            "parameter_tuning":
                False,

            "formal_content_opened":
                False,

            "formal_outcomes_read":
                False,

            "formal_evaluation":
                False,
        },

        "regression": {
            "pre":
                tests_before,

            "targeted":
                n_target,

            "post":
                tests_after,

            "status":
                "PASS",
        },

        "storage": {
            "free_GiB":
                free_gib,

            "reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "next":
            (
                "execute development-only class-aware "
                "predictive vs original-reactive ADB "
                "metrics and freeze numeric "
                "non-inferiority bounds"
            ),
    }

    atomic_write(
        REPORT,
        canonical_json_bytes(
            result
        ),
    )

    print(
        "contract SHA256      =",
        sha256_file(
            CONTRACT
        ),
    )

    print(
        "road ROI SHA256      =",
        sha256_file(
            ROAD_ROI_ARTIFACT
        ),
    )

    print(
        "helper module SHA256 =",
        sha256_file(
            MODULE
        ),
    )

    print(
        "freeze report SHA256 =",
        sha256_file(
            REPORT
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 EVALUATOR REFERENCE CONTRACT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "I_final                     = FROZEN POST-ACTUATION"
    )

    print(
        "constructed oracle          = FROZEN EVALUATOR ONLY"
    )

    print(
        "future GT controller input  = NO"
    )

    print(
        "measured ADB ground truth   = NO"
    )

    print(
        "vehicle reference region    = FROZEN"
    )

    print(
        "pedestrian future full-box  = FROZEN"
    )

    print(
        "cyclist future full-box     = FROZEN"
    )

    print(
        "road ROI                    = ALL VALID ACTUATOR CELLS"
    )

    print(
        "road ROI artifact           = FROZEN"
    )

    print(
        "development metrics         = NOT COMPUTED"
    )

    print(
        "numeric bounds              = NOT SELECTED"
    )

    print(
        "formal content              = NOT OPENED"
    )

    print(
        "policy/parameter tuning     = NO"
    )

    print(
        "targeted regression         =",
        f"{n_target} / {n_target} PASS",
    )

    print(
        "full Stage6 regression      =",
        f"{tests_after} / {tests_after} PASS",
    )

    print(
        "STATUS = PASS_EVALUATOR_REFERENCE_CONTRACT_FROZEN"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "terminal remains open = YES"
    )


try:

    main()

except BaseException as exc:

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 EVALUATOR REFERENCE CONTRACT = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(
            exc
        ).__name__,
        str(
            exc
        ),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "development metrics computed = NO"
    )

    print(
        "numeric bounds selected      = NO"
    )

    print(
        "formal content opened        = NO"
    )

    print(
        "formal outcomes read         = NO"
    )

    print(
        "formal evaluation            = NO"
    )

    print(
        "controller modified          = NO"
    )

    print()
    print(
        "Do not run formal Stage6 evaluation."
    )

    print(
        "Send this BLOCKED output for targeted repair."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
