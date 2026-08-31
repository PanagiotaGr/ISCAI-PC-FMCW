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


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

# ============================================================
# Frozen dependencies
# ============================================================

BLOCK67_CLOSURE = (
    S6
    / "reports/block67_final_closure.json"
)

RECONCILIATION = (
    S6
    / "reports/block68_preoutcome_metric_reconciliation.json"
)

METRIC_PROTOCOL = (
    S6
    / "configs/stage6_metric_freeze_protocol.json"
)

METRIC_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/metric_semantics.py"
)

METRIC_INTERFACE = (
    S6
    / "src/iscai_stage6/adb/baseline_metric_contract.py"
)

I_FINAL_BIND = (
    S6
    / "reports/block68_exact_evaluator_binding_repair.json"
)

EVALUATOR_REPORT = (
    S6
    / "reports/block68_evaluator_reference_contract_freeze.json"
)

EVALUATOR_CONTRACT = (
    S6
    / "configs/stage6_evaluator_reference_contract.json"
)

ROAD_ROI = (
    S6
    / "artifacts/block68/frozen_road_roi_definition.json"
)

EVALUATOR_HELPER = (
    S6
    / "src/iscai_stage6/adb/evaluator_reference.py"
)

NI_REPORT = (
    S6
    / "reports/block68_noninferiority_bound_rule_freeze.json"
)

NI_CONFIG = (
    S6
    / "configs/stage6_noninferiority_bound_selection_rule.json"
)

NI_MODULE = (
    S6
    / "src/iscai_stage6/adb/noninferiority_bounds.py"
)

CLASS_AWARE_POLICY = (
    S6
    / "src/iscai_stage6/adb/class_aware_policy.py"
)


EXPECTED_SHA256 = {
    BLOCK67_CLOSURE:
        "67d168ed7678164b8d85885e6e81f41f"
        "7206bd8f754e13ec5cc28c3a65aa20e7",

    RECONCILIATION:
        "077a89fa5163f8862a3a5b7489d98929"
        "d9ae3e160ed9e030eb75f42a91d48f9f",

    METRIC_PROTOCOL:
        "409fbb2785e4ff13f29fdff96c91d608"
        "c245e009fcbb86ce57099d6a4e1815c9",

    METRIC_RUNTIME:
        "bf8358b5a7edfe40ffac2718ca7cd5af"
        "6e7b5fdb3ff8a2823caad6d0f40f057a",

    METRIC_INTERFACE:
        "39518597079f65d556ee8f0fd3b00ce5"
        "25319623c9eb8a83f6e3c608954213e9",

    I_FINAL_BIND:
        "1505c780423efbe27735c3729b95a18c"
        "3da2093f49826bd96cbfad8fdce5d5f1",

    EVALUATOR_REPORT:
        "5885424d164a058a151c50dd663acec4"
        "da7aeba66a56d5e0191c777682e113a8",

    EVALUATOR_CONTRACT:
        "c9ff0390820a3c37702fb391ba603244f"
        "ffcf7c03c476045b590aa483e89bd7e",

    ROAD_ROI:
        "832f4bc3215fe585fee781080c98b2928"
        "157f26e8a66d22f428f15604afb75e0",

    EVALUATOR_HELPER:
        "2015136fdcbfe3f837bcba5a2ee05578a"
        "4d8a01c4ecfd52e4dfac9f864349891",

    NI_REPORT:
        "289f9369735566daa2344a90cb312d4c0"
        "6ede76b73827f07e408855c4e06ce6b",

    NI_CONFIG:
        "b5c0f40725e26578a4739f0582e4467e"
        "c48b72db8bb077d13862cc83dee98d24",

    NI_MODULE:
        "3c67a7f152359710cb24db1a6e01c4b5"
        "80728e6480b7c5d8f9c40a41349cf1cd",

    CLASS_AWARE_POLICY:
        "b998f468b98c2770c48c84a2c0aaaa21"
        "77c2d84b8fc838e80fef9b9da1ebc14b",
}


# ============================================================
# New frozen outputs
# ============================================================

CONTRACT = (
    S6
    / "configs/stage6_reactive_development_scoring_contract.json"
)

MODULE = (
    S6
    / "src/iscai_stage6/adb/reactive_development_scoring.py"
)

TEST = (
    S6
    / "tests/test_block68_reactive_development_scoring.py"
)

REPORT = (
    S6
    / "reports/block68_reactive_development_scoring_freeze.json"
)

EXPECTED_PRE_TESTS = 217
NEW_TESTS = 7
EXPECTED_POST_TESTS = EXPECTED_PRE_TESTS + NEW_TESTS

MIN_FREE_GIB = 250.0

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)


MODULE_SOURCE = r'''from __future__ import annotations

from typing import Sequence

import numpy as np


REACTIVE_EVALUATION_HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)


def _validate_horizons(
    horizons_s: Sequence[float],
) -> tuple[float, ...]:

    values = tuple(
        float(value)
        for value in horizons_s
    )

    if values != REACTIVE_EVALUATION_HORIZONS_S:
        raise ValueError(
            "Reactive development scoring is frozen to "
            "horizons (0.1, 0.3, 0.5, 1.0) seconds."
        )

    return values


def _validate_current_illumination(
    current_illumination,
) -> np.ndarray:

    current = np.asarray(
        current_illumination,
        dtype=np.float64,
    )

    if current.ndim != 2:
        raise ValueError(
            "current_illumination must be a 2-D "
            "[theta,range] Part-A reactive map"
        )

    if not np.all(
        np.isfinite(current)
    ):
        raise ValueError(
            "current_illumination contains non-finite values"
        )

    if (
        np.any(current < 0.0)
        or
        np.any(current > 1.0)
    ):
        raise ValueError(
            "current_illumination must lie in [0,1]"
        )

    return current


def hold_current_reactive_schedule(
    current_illumination,
    *,
    horizons_s: Sequence[float] = (
        0.1,
        0.3,
        0.5,
        1.0,
    ),
) -> np.ndarray:
    """
    Frozen Block6.8 single-decision evaluator semantics.

    The original reactive ADB controller produces one causal
    current-state Part-A-style illumination action at t0.

    For future-horizon comparison against predictive ADB,
    that already-made action is held unchanged over the four
    frozen Stage6 evaluation horizons.

    No future observation, future state, oracle geometry,
    future trajectory, or posterior update enters this
    operation.

    This is an evaluator representation of the t0 reactive
    decision. It is NOT a simulated closed-loop reactive
    controller with future re-observation.
    """

    _validate_horizons(
        horizons_s
    )

    current = (
        _validate_current_illumination(
            current_illumination
        )
    )

    schedule = np.repeat(
        current[
            None,
            :,
            :,
        ],
        repeats=len(
            REACTIVE_EVALUATION_HORIZONS_S
        ),
        axis=0,
    )

    return np.asarray(
        schedule,
        dtype=np.float64,
    ).copy()


def binary_dim_support(
    final_illumination_schedule,
) -> np.ndarray:
    """
    Reconciled Block6.6 metric semantics:

        D_tau = [I_final_tau < 1.0]

    Equality at exactly 1.0 is NOT dimmed support.
    """

    values = np.asarray(
        final_illumination_schedule,
        dtype=np.float64,
    )

    if values.ndim != 3:
        raise ValueError(
            "final_illumination_schedule must have shape "
            "[horizon,theta,range]"
        )

    if values.shape[0] != len(
        REACTIVE_EVALUATION_HORIZONS_S
    ):
        raise ValueError(
            "final illumination must contain exactly "
            "four frozen horizons"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "final illumination contains non-finite values"
        )

    if (
        np.any(values < 0.0)
        or
        np.any(values > 1.0)
    ):
        raise ValueError(
            "final illumination must lie in [0,1]"
        )

    return (
        values
        <
        1.0
    )


def validate_supported_reactive_metric_values(
    values,
    *,
    metric_name: str,
) -> np.ndarray:
    """
    Validate an already support-filtered reactive metric
    vector before it is passed to the frozen NI-bound rule.

    This helper does not remove NaN/Inf values silently.
    Unsupported scenarios must be excluded explicitly by the
    evaluator before this function is called.
    """

    array = np.asarray(
        tuple(values),
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{metric_name}: metric values must be 1-D"
        )

    if array.size == 0:
        raise ValueError(
            f"{metric_name}: no supported observations"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{metric_name}: non-finite supported value"
        )

    if (
        np.any(array < 0.0)
        or
        np.any(array > 1.0)
    ):
        raise ValueError(
            f"{metric_name}: normalized values "
            "must lie in [0,1]"
        )

    return array


def scientific_input_contract() -> dict:
    """
    Machine-readable semantic boundary used by tests and
    the development runner.
    """

    return {
        "baseline":
            "original_reactive_ADB",

        "decision_time":
            "t0_current_causal_state",

        "controller_future_input":
            False,

        "future_reobservation":
            False,

        "future_controller_refresh":
            False,

        "future_truth_controller_input":
            False,

        "posterior_controller_input":
            False,

        "constructed_oracle_controller_input":
            False,

        "horizons_s":
            list(
                REACTIVE_EVALUATION_HORIZONS_S
            ),

        "schedule_semantics":
            (
                "hold_t0_current_reactive_PartA_action_"
                "unchanged_across_future_scoring_horizons"
            ),

        "metric_dim_support":
            "D_tau = [I_final_tau < 1.0]",

        "role":
            "single_decision_future_evaluation_comparator",
    }
'''


TEST_SOURCE = r'''from __future__ import annotations

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
'''


def require(
    condition,
    message,
):

    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:

    digest = sha256()

    with path.open("rb") as stream:

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

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(
        data
    )

    os.replace(
        temporary,
        path,
    )


def read_json_nonformal(
    path: Path,
):

    require(
        "formal"
        not in
        str(
            path.resolve()
        ).lower(),
        (
            "Formal-named content access "
            f"is forbidden in this block: {path}"
        ),
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def run_tests(
    pattern: str,
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
        cwd=str(S6),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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
    value: str,
    lines: int = 100,
):

    return "\n".join(
        value.splitlines()[
            -lines:
        ]
    )


def exact_upstream_gate():

    for path, expected in (
        EXPECTED_SHA256.items()
    ):

        require(
            path.is_file(),
            (
                "Missing frozen dependency: "
                f"{path}"
            ),
        )

        actual = sha256_file(
            path
        )

        require(
            actual == expected,
            (
                "Frozen dependency changed:\n"
                f"path={path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            ),
        )


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART 2A/2"
    )
    print(
        "ORIGINAL-REACTIVE FUTURE-SCORING SEMANTICS FREEZE"
    )
    print(
        "PRE-DEVELOPMENT-OUTCOME / NO FORMAL CONTENT"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Idempotent frozen readback
    # --------------------------------------------------------

    if REPORT.is_file():

        existing = (
            read_json_nonformal(
                REPORT
            )
        )

        if (
            existing.get(
                "status"
            )
            ==
            "PASS_REACTIVE_SCORING_SEMANTICS_FROZEN"
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
                    "test",
                    TEST,
                ),
            ):

                require(
                    path.is_file(),
                    (
                        "Frozen output missing: "
                        f"{path}"
                    ),
                )

                expected = (
                    existing[
                        "frozen_outputs"
                    ][
                        key
                    ][
                        "sha256"
                    ]
                )

                require(
                    sha256_file(
                        path
                    )
                    ==
                    expected,
                    (
                        "Frozen output changed: "
                        f"{path}"
                    ),
                )

            exact_upstream_gate()

            rc, count, output = (
                run_tests(
                    "test_*.py"
                )
            )

            require(
                rc == 0,
                (
                    "Regression failed:\n"
                    +
                    tail(
                        output
                    )
                ),
            )

            require(
                count
                ==
                EXPECTED_POST_TESTS,
                (
                    "Expected "
                    f"{EXPECTED_POST_TESTS} tests; "
                    f"got {count}"
                ),
            )

            print(
                "contract readback   = EXACT PASS"
            )
            print(
                "runtime readback    = EXACT PASS"
            )
            print(
                "upstream immutability = PASS"
            )
            print(
                "Stage6 regression   =",
                f"{count} / {count} PASS",
            )
            print(
                "STATUS = "
                "PASS_REACTIVE_SCORING_SEMANTICS_FROZEN"
            )
            print(
                "no rewrite performed = YES"
            )
            print(
                "terminal remains open = YES"
            )

            return

    # --------------------------------------------------------
    # A. Upstream frozen seals
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT UPSTREAM SEALS ====="
    )

    exact_upstream_gate()

    ni_report = (
        read_json_nonformal(
            NI_REPORT
        )
    )

    require(
        ni_report.get(
            "status"
        )
        ==
        "PASS_NI_BOUND_SELECTION_RULE_FROZEN",
        (
            "NI bound-selection rule "
            "is not frozen PASS."
        ),
    )

    require(
        ni_report[
            "scientific_boundary"
        ][
            "development_metric_values_read"
        ]
        is False,
        (
            "Development metrics were already "
            "read before reactive scoring freeze."
        ),
    )

    require(
        ni_report[
            "scientific_boundary"
        ][
            "formal_content_opened"
        ]
        is False,
        (
            "Formal content was already opened."
        ),
    )

    evaluator_report = (
        read_json_nonformal(
            EVALUATOR_REPORT
        )
    )

    require(
        evaluator_report.get(
            "status"
        )
        ==
        "PASS_EVALUATOR_REFERENCE_CONTRACT_FROZEN",
        (
            "Evaluator reference contract "
            "is not frozen PASS."
        ),
    )

    print(
        "Block6.7 closure        = EXACT PASS"
    )
    print(
        "metric authority        = EXACT PASS"
    )
    print(
        "I_final evaluator bind  = EXACT PASS"
    )
    print(
        "oracle/ROI contract     = EXACT PASS"
    )
    print(
        "NI bound rule           = EXACT PASS"
    )
    print(
        "class-aware policy      = EXACT PASS"
    )

    # --------------------------------------------------------
    # B. Scientific boundary
    # --------------------------------------------------------

    print()
    print(
        "===== B. PRE-OUTCOME SCIENTIFIC BOUNDARY ====="
    )

    print(
        "development metric values = NOT READ"
    )
    print(
        "reactive outcomes          = NOT READ"
    )
    print(
        "predictive outcomes        = NOT READ"
    )
    print(
        "numeric NI deltas          = NOT COMPUTED"
    )
    print(
        "formal content             = NOT OPENED"
    )
    print(
        "controller modification    = NO"
    )
    print(
        "policy tuning              = NO"
    )

    # --------------------------------------------------------
    # C. Pre-freeze regression
    # --------------------------------------------------------

    print()
    print(
        "===== C. PRE-FREEZE REGRESSION ====="
    )

    rc_pre, n_pre, out_pre = (
        run_tests(
            "test_*.py"
        )
    )

    require(
        rc_pre == 0,
        (
            "Pre-freeze regression failed:\n"
            +
            tail(
                out_pre
            )
        ),
    )

    require(
        n_pre == EXPECTED_PRE_TESTS,
        (
            "Expected "
            f"{EXPECTED_PRE_TESTS} "
            "pre-freeze tests; "
            f"got {n_pre}"
        ),
    )

    print(
        "pre-freeze regression =",
        f"{n_pre} / {n_pre} PASS",
    )

    # --------------------------------------------------------
    # D. Freeze evaluator semantics BEFORE outcomes
    # --------------------------------------------------------

    print()
    print(
        "===== D. BUILD REACTIVE SCORING CONTRACT ====="
    )

    contract = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8_Part2A",

        "status":
            "FROZEN_PREOUTCOME_REACTIVE_SCORING_SEMANTICS",

        "freeze_timing":
            "before_any_reactive_or_predictive_development_metric",

        "scientific_role":
            (
                "bind original_reactive_ADB current-state "
                "Part-A action to the four-horizon "
                "single-decision Stage6 evaluator"
            ),

        "original_reactive_ADB": {
            "controller_state_source":
                "current_causal_state_t0_only",

            "base_illumination_source":
                (
                    "frozen_PartA_current_state_reactive_"
                    "illumination_map"
                ),

            "future_input":
                False,

            "posterior_input":
                False,

            "future_reobservation":
                False,

            "future_controller_refresh":
                False,

            "future_truth_controller_input":
                False,

            "oracle_controller_input":
                False,

            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "future_scoring_schedule":
                (
                    "repeat_the_already_made_t0_"
                    "reactive_illumination_map_unchanged_"
                    "at_each_frozen_horizon"
                ),

            "interpretation":
                (
                    "single_decision_future_evaluation; "
                    "not closed_loop future re-observation"
                ),
        },

        "metric_binding": {
            "I_final":
                (
                    "held current Part-A reactive "
                    "illumination schedule"
                ),

            "binary_dim_support":
                "D_tau = [I_final_tau < 1.0]",

            "strict_less_than_one":
                True,

            "metric_authority":
                (
                    "reconciled earliest Block6.6 "
                    "pre-outcome preregistration"
                ),

            "metric_runtime_modified":
                False,
        },

        "oracle_boundary": {
            "constructed_future_reference":
                True,

            "future_truth_role":
                "evaluator_only_after_controller_decision",

            "measured_ADB_ground_truth":
                False,

            "controller_input":
                False,

            "bound_size_input":
                False,

            "actor_population_selector":
                False,
        },

        "bound_freeze_boundary": {
            "over_masking_delta_source":
                (
                    "reactive development values only"
                ),

            "pedestrian_visibility_delta_source":
                (
                    "reactive supported development "
                    "values only"
                ),

            "cyclist_visibility_delta_source":
                (
                    "reactive supported development "
                    "values only"
                ),

            "predictive_values_may_set_delta":
                False,

            "formal_values_may_set_delta":
                False,
        },

        "anti_leakage": {
            "future_observation_for_reactive_controller":
                False,

            "future_GT_for_reactive_controller":
                False,

            "oracle_for_reactive_controller":
                False,

            "future_validity_as_actor_selector":
                False,

            "future_track_duration_as_actor_selector":
                False,

            "tracks_to_predict_as_control_selector":
                False,
        },

        "scientific_change":
            (
                "evaluator representation only; "
                "no controller or policy modification"
            ),

        "formal_evaluation_allowed_after_this_freeze":
            False,

        "next":
            (
                "run actual reactive-only development "
                "evaluation, compute three reactive-only "
                "NI deltas, and freeze exact formal "
                "acceptance policy before any predictive "
                "development outcome is used for bound size"
            ),
    }

    contract_bytes = (
        canonical_json_bytes(
            contract
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

    # Syntax/JSON gate before any write.
    json.loads(
        contract_bytes.decode(
            "utf-8"
        )
    )

    compile(
        MODULE_SOURCE,
        str(MODULE),
        "exec",
    )

    compile(
        TEST_SOURCE,
        str(TEST),
        "exec",
    )

    print(
        "contract JSON syntax = PASS"
    )
    print(
        "helper Python syntax = PASS"
    )
    print(
        "test Python syntax   = PASS"
    )
    print(
        "reactive t0 action   = CURRENT CAUSAL ONLY"
    )
    print(
        "future refresh       = PROHIBITED"
    )
    print(
        "4-horizon binding    = HOLD CURRENT ACTION"
    )
    print(
        "future GT            = EVALUATOR ONLY"
    )

    # --------------------------------------------------------
    # E. Transactional write
    # --------------------------------------------------------

    print()
    print(
        "===== E. TRANSACTIONAL FREEZE ====="
    )

    targets = (
        CONTRACT,
        MODULE,
        TEST,
    )

    unexpected = [
        str(path)
        for path in targets
        if path.exists()
    ]

    require(
        not unexpected,
        (
            "Unexpected pre-existing output "
            "without PASS freeze report: "
            +
            repr(
                unexpected
            )
        ),
    )

    written = []

    try:

        for path, payload in (
            (
                CONTRACT,
                contract_bytes,
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
                payload,
            )

            written.append(
                path
            )

        print(
            "reactive scoring contract = WRITTEN"
        )
        print(
            "isolated helper module     = WRITTEN"
        )
        print(
            "targeted tests             = WRITTEN"
        )

        # ----------------------------------------------------
        # F. Targeted tests
        # ----------------------------------------------------

        print()
        print(
            "===== F. TARGETED TESTS ====="
        )

        (
            rc_target,
            n_target,
            out_target,
        ) = run_tests(
            "test_block68_reactive_development_scoring.py"
        )

        require(
            rc_target == 0,
            (
                "Targeted tests failed:\n"
                +
                tail(
                    out_target
                )
            ),
        )

        require(
            n_target == NEW_TESTS,
            (
                "Expected "
                f"{NEW_TESTS} targeted tests; "
                f"got {n_target}"
            ),
        )

        print(
            "targeted regression =",
            f"{n_target} / {n_target} PASS",
        )

        # ----------------------------------------------------
        # G. Full regression
        # ----------------------------------------------------

        print()
        print(
            "===== G. FULL STAGE6 REGRESSION ====="
        )

        (
            rc_post,
            n_post,
            out_post,
        ) = run_tests(
            "test_*.py"
        )

        require(
            rc_post == 0,
            (
                "Full Stage6 regression failed:\n"
                +
                tail(
                    out_post
                )
            ),
        )

        require(
            n_post
            ==
            EXPECTED_POST_TESTS,
            (
                "Expected "
                f"{EXPECTED_POST_TESTS} "
                "post-freeze tests; "
                f"got {n_post}"
            ),
        )

        print(
            "full Stage6 regression =",
            f"{n_post} / {n_post} PASS",
        )

    except BaseException:

        print()
        print(
            "FREEZE FAILURE -> "
            "ROLLING BACK NEW FILES"
        )

        for path in reversed(
            written
        ):

            try:

                if path.exists():
                    path.unlink()

            except Exception:
                pass

        (
            rc_roll,
            n_roll,
            out_roll,
        ) = run_tests(
            "test_*.py"
        )

        print(
            "rollback regression =",
            n_roll,
            "/",
            n_roll,
            (
                "PASS"
                if rc_roll == 0
                else "FAIL"
            ),
        )

        require(
            rc_roll == 0,
            (
                "Rollback regression failed:\n"
                +
                tail(
                    out_roll
                )
            ),
        )

        require(
            n_roll
            ==
            EXPECTED_PRE_TESTS,
            (
                "Rollback did not restore "
                "original test population."
            ),
        )

        raise

    # --------------------------------------------------------
    # H. Exact semantic readback
    # --------------------------------------------------------

    print()
    print(
        "===== H. EXACT SEMANTIC READBACK ====="
    )

    frozen = (
        read_json_nonformal(
            CONTRACT
        )
    )

    require(
        frozen[
            "status"
        ]
        ==
        "FROZEN_PREOUTCOME_REACTIVE_SCORING_SEMANTICS",
        "Frozen contract status changed.",
    )

    reactive = frozen[
        "original_reactive_ADB"
    ]

    require(
        reactive[
            "future_input"
        ]
        is False,
        "Reactive future input unexpectedly enabled.",
    )

    require(
        reactive[
            "future_reobservation"
        ]
        is False,
        "Future re-observation unexpectedly enabled.",
    )

    require(
        reactive[
            "future_controller_refresh"
        ]
        is False,
        "Future reactive refresh unexpectedly enabled.",
    )

    require(
        reactive[
            "future_truth_controller_input"
        ]
        is False,
        "Future GT unexpectedly reaches controller.",
    )

    require(
        reactive[
            "horizons_s"
        ]
        ==
        [
            0.1,
            0.3,
            0.5,
            1.0,
        ],
        "Frozen horizon list changed.",
    )

    require(
        frozen[
            "metric_binding"
        ][
            "binary_dim_support"
        ]
        ==
        "D_tau = [I_final_tau < 1.0]",
        "Dim-support semantics changed.",
    )

    print(
        "t0 reactive action      = FROZEN"
    )
    print(
        "future re-observation   = NO"
    )
    print(
        "future controller update= NO"
    )
    print(
        "held schedule horizons  = 4"
    )
    print(
        "D_tau semantics         = I_final < 1.0"
    )
    print(
        "future GT controller    = NO"
    )
    print(
        "oracle controller       = NO"
    )

    # --------------------------------------------------------
    # I. Upstream immutability
    # --------------------------------------------------------

    print()
    print(
        "===== I. UPSTREAM IMMUTABILITY ====="
    )

    exact_upstream_gate()

    print(
        "metric runtime          = UNCHANGED"
    )
    print(
        "metric protocol         = UNCHANGED"
    )
    print(
        "NI rule                 = UNCHANGED"
    )
    print(
        "evaluator reference     = UNCHANGED"
    )
    print(
        "class-aware controller  = UNCHANGED"
    )
    print(
        "Block6.7 closure        = UNCHANGED"
    )

    # --------------------------------------------------------
    # J. Storage
    # --------------------------------------------------------

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
            "250-GiB storage reserve violated."
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

    # --------------------------------------------------------
    # K. Freeze report
    # --------------------------------------------------------

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8_Part2A",

        "status":
            "PASS_REACTIVE_SCORING_SEMANTICS_FROZEN",

        "scientific_boundary": {
            "development_metric_values_read":
                False,

            "reactive_outcomes_read":
                False,

            "predictive_outcomes_read":
                False,

            "numeric_NI_deltas_computed":
                False,

            "formal_content_opened":
                False,

            "formal_outcomes_read":
                False,

            "formal_evaluation":
                False,

            "controller_modified":
                False,

            "policy_tuning":
                False,

            "parameter_tuning":
                False,
        },

        "reactive_scoring": {
            "decision":
                "current_causal_t0",

            "future_reobservation":
                False,

            "future_controller_refresh":
                False,

            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "schedule":
                "hold_t0_action_across_all_horizons",

            "dim_support":
                "I_final < 1.0",
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
            str(path):
                expected
            for path, expected
            in EXPECTED_SHA256.items()
        },

        "regression": {
            "pre":
                n_pre,

            "targeted":
                n_target,

            "post":
                n_post,

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

        "formal_evaluation_allowed":
            False,

        "next":
            (
                "reactive-only development metric "
                "execution and exact NI delta freeze"
            ),
    }

    atomic_write(
        REPORT,
        canonical_json_bytes(
            result
        ),
    )

    print()
    print(
        "===== K. FREEZE REPORT ====="
    )

    print(
        "contract SHA256 =",
        sha256_file(
            CONTRACT
        ),
    )

    print(
        "helper SHA256   =",
        sha256_file(
            MODULE
        ),
    )

    print(
        "freeze SHA256   =",
        sha256_file(
            REPORT
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 REACTIVE SCORING SEMANTICS — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "reactive decision source    = CURRENT CAUSAL t0"
    )
    print(
        "future input                = NO"
    )
    print(
        "future re-observation       = NO"
    )
    print(
        "future controller refresh   = NO"
    )
    print(
        "evaluation horizons         = 0.1 / 0.3 / 0.5 / 1.0 s"
    )
    print(
        "future schedule semantics   = HOLD t0 ACTION"
    )
    print(
        "binary dim support          = I_final < 1.0"
    )
    print(
        "constructed oracle          = EVALUATOR ONLY"
    )
    print(
        "measured ADB ground truth   = NO"
    )
    print(
        "development metrics         = NOT READ"
    )
    print(
        "reactive outcomes           = NOT READ"
    )
    print(
        "predictive outcomes         = NOT READ"
    )
    print(
        "exact NI deltas             = NOT COMPUTED"
    )
    print(
        "formal content              = NOT OPENED"
    )
    print(
        "controller/policy tuning    = NO"
    )
    print(
        "targeted regression         =",
        f"{n_target} / {n_target} PASS",
    )
    print(
        "full Stage6 regression      =",
        f"{n_post} / {n_post} PASS",
    )
    print(
        "formal evaluation allowed   = NO"
    )
    print(
        "STATUS = "
        "PASS_REACTIVE_SCORING_SEMANTICS_FROZEN"
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
        "BLOCK 6.8 REACTIVE SCORING SEMANTICS = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "development metric values read = NO"
    )
    print(
        "reactive outcomes read          = NO"
    )
    print(
        "predictive outcomes read        = NO"
    )
    print(
        "exact NI deltas computed        = NO"
    )
    print(
        "formal content opened           = NO"
    )
    print(
        "formal evaluation               = NO"
    )
    print(
        "controller modified             = NO"
    )
    print()
    print(
        "Do not run development metric evaluation."
    )
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
