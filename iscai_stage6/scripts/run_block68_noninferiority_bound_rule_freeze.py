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

EVALUATOR_REPORT = (
    S6
    / "reports/"
      "block68_evaluator_reference_contract_freeze.json"
)

EVALUATOR_CONTRACT = (
    S6
    / "configs/"
      "stage6_evaluator_reference_contract.json"
)

ROAD_ROI = (
    S6
    / "artifacts/block68/"
      "frozen_road_roi_definition.json"
)

EVALUATOR_HELPER = (
    S6
    / "src/iscai_stage6/adb/"
      "evaluator_reference.py"
)

METRIC_PROTOCOL = (
    S6
    / "configs/"
      "stage6_metric_freeze_protocol.json"
)

METRIC_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

METRIC_INTERFACE = (
    S6
    / "src/iscai_stage6/adb/"
      "baseline_metric_contract.py"
)

RECONCILIATION = (
    S6
    / "reports/"
      "block68_preoutcome_metric_reconciliation.json"
)

FINAL_I_BIND = (
    S6
    / "reports/"
      "block68_exact_evaluator_binding_repair.json"
)

RULE_CONFIG = (
    S6
    / "configs/"
      "stage6_noninferiority_bound_selection_rule.json"
)

RULE_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "noninferiority_bounds.py"
)

RULE_TEST = (
    S6
    / "tests/"
      "test_block68_noninferiority_bound_rule.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_noninferiority_bound_rule_freeze.json"
)


EXPECTED = {
    EVALUATOR_REPORT:
        "5885424d164a058a151c50dd663acec4"
        "da7aeba66a56d5e0191c777682e113a8",

    EVALUATOR_CONTRACT:
        "c9ff0390820a3c37702fb391ba603244"
        "fffcf7c03c476045b590aa483e89bd7e",

    ROAD_ROI:
        "832f4bc3215fe585fee781080c98b292"
        "8157f26e8a66d22f428f15604afb75e0",

    EVALUATOR_HELPER:
        "2015136fdcbfe3f837bcba5a2ee05578"
        "a4d8a01c4ecfd52e4dfac9f864349891",

    METRIC_PROTOCOL:
        "409fbb2785e4ff13f29fdff96c91d608"
        "c245e009fcbb86ce57099d6a4e1815c9",

    METRIC_RUNTIME:
        "bf8358b5a7edfe40ffac2718ca7cd5af"
        "6e7b5fdb3ff8a2823caad6d0f40f057a",

    METRIC_INTERFACE:
        "39518597079f65d556ee8f0fd3b00ce5"
        "25319623c9eb8a83f6e3c608954213e9",

    RECONCILIATION:
        "077a89fa5163f8862a3a5b7489d98929"
        "d9ae3e160ed9e030eb75f42a91d48f9f",

    FINAL_I_BIND:
        "1505c780423efbe27735c3729b95a18c"
        "3da2093f49826bd96cbfad8fdce5d5f1",
}

EXPECTED_PRE_TESTS = 209
NEW_TESTS = 8
EXPECTED_POST_TESTS = (
    EXPECTED_PRE_TESTS
    +
    NEW_TESTS
)

MIN_FREE_GIB = 250.0


MODULE_SOURCE = r'''from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOUND_SCALE = 1.96
BOUND_FLOOR = 0.02
BOUND_CAP = 0.05

OVERMASK_REQUIRED_SUPPORT = 120
VRU_MINIMUM_SUPPORT = 20

SUPPORTED_BOUND_METRICS = (
    "over_masking_area",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
)


@dataclass(frozen=True)
class ReactiveVariabilityBound:
    metric_name: str
    support_count: int
    reactive_mean: float
    reactive_sample_std: float
    reactive_standard_error: float
    scaled_standard_error: float
    frozen_delta: float
    semantics: str


def _validated_reactive_values(
    reactive_values,
) -> np.ndarray:

    values = np.asarray(
        tuple(reactive_values),
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "reactive_values must be one-dimensional"
        )

    if values.size == 0:
        raise ValueError(
            "reactive_values cannot be empty"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "reactive_values must contain only "
            "explicitly supported finite observations"
        )

    if (
        np.any(values < 0.0)
        or
        np.any(values > 1.0)
    ):
        raise ValueError(
            "reactive metric values must lie in [0,1]"
        )

    return values


def derive_reactive_only_noninferiority_delta(
    metric_name: str,
    reactive_values,
) -> ReactiveVariabilityBound:
    """
    Frozen Block6.8 pre-outcome rule.

    The non-inferiority tolerance is derived from the
    ORIGINAL REACTIVE comparator only:

        SE_R = sample_std(R) / sqrt(n)
        raw  = 1.96 * SE_R
        delta = clip(raw, 0.02, 0.05)

    Predictive/class-aware values are deliberately absent
    from this API, preventing target-system outcomes from
    influencing the tolerance.

    The 1.96 multiplier is a frozen variability scale.
    This function does NOT claim that delta is a formal
    confidence interval.
    """

    metric = str(metric_name)

    if metric not in SUPPORTED_BOUND_METRICS:
        raise ValueError(
            f"unsupported bound metric: {metric}"
        )

    values = _validated_reactive_values(
        reactive_values
    )

    n = int(values.size)

    if metric == "over_masking_area":

        if n != OVERMASK_REQUIRED_SUPPORT:
            raise ValueError(
                "over_masking_area requires exactly "
                f"{OVERMASK_REQUIRED_SUPPORT} "
                "development scenario observations"
            )

    else:

        if n < VRU_MINIMUM_SUPPORT:
            raise ValueError(
                f"{metric} requires at least "
                f"{VRU_MINIMUM_SUPPORT} supported "
                "development observations"
            )

    sample_std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    standard_error = float(
        sample_std
        /
        np.sqrt(
            float(n)
        )
    )

    scaled = float(
        BOUND_SCALE
        *
        standard_error
    )

    delta = float(
        min(
            BOUND_CAP,
            max(
                BOUND_FLOOR,
                scaled,
            ),
        )
    )

    return ReactiveVariabilityBound(
        metric_name=metric,
        support_count=n,
        reactive_mean=float(
            np.mean(values)
        ),
        reactive_sample_std=sample_std,
        reactive_standard_error=standard_error,
        scaled_standard_error=scaled,
        frozen_delta=delta,
        semantics=(
            "reactive_comparator_only_variability_scaled_"
            "absolute_noninferiority_tolerance"
        ),
    )
'''


TEST_SOURCE = r'''from __future__ import annotations

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

            digest.update(chunk)

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
    ).encode("utf-8")


def atomic_write(
    path: Path,
    data: bytes,
):

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_bytes(data)

    os.replace(
        temp,
        path,
    )


def safe_json(
    path: Path,
):

    resolved = str(
        path.resolve()
    ).lower()

    require(
        "formal"
        not in
        resolved,
        (
            "FORMAL-NAMED CONTENT ACCESS "
            f"FORBIDDEN: {path}"
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
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    count = None

    for line in process.stdout.splitlines():

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
    n=100,
):

    return "\n".join(
        text.splitlines()[-n:]
    )


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "PRE-OUTCOME NON-INFERIORITY BOUND-SELECTION RULE FREEZE"
    )
    print(
        "NO DEVELOPMENT PERFORMANCE VALUES / NO FORMAL CONTENT"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Idempotent readback
    # --------------------------------------------------------

    if REPORT.is_file():

        existing = safe_json(
            REPORT
        )

        if (
            existing.get("status")
            ==
            "PASS_NI_BOUND_SELECTION_RULE_FROZEN"
        ):

            print()
            print(
                "===== EXISTING RULE FREEZE DETECTED ====="
            )

            for key, path in (
                ("config", RULE_CONFIG),
                ("module", RULE_MODULE),
                ("test", RULE_TEST),
            ):

                require(
                    path.is_file(),
                    f"Frozen output missing: {path}",
                )

                require(
                    sha256_file(path)
                    ==
                    existing[
                        "frozen_outputs"
                    ][key]["sha256"],
                    f"Frozen output changed: {path}",
                )

            rc, count, output = run_tests(
                "test_*.py"
            )

            require(
                rc == 0,
                (
                    "Regression failed:\n"
                    +
                    tail(output)
                ),
            )

            require(
                count
                ==
                EXPECTED_POST_TESTS,
                (
                    "Expected "
                    f"{EXPECTED_POST_TESTS} tests; "
                    f"got {count}."
                ),
            )

            print(
                "rule config       = EXACT PASS"
            )
            print(
                "rule runtime      = EXACT PASS"
            )
            print(
                "Stage6 regression =",
                f"{count} / {count} PASS",
            )
            print(
                "STATUS = "
                "PASS_NI_BOUND_SELECTION_RULE_FROZEN"
            )
            print(
                "no rewrite performed = YES"
            )
            print(
                "terminal remains open = YES"
            )

            return

    # --------------------------------------------------------
    # A. Exact upstream freeze
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT UPSTREAM FREEZE ====="
    )

    for path, expected_sha in EXPECTED.items():

        require(
            path.is_file(),
            f"Missing frozen dependency: {path}",
        )

        actual = sha256_file(path)

        require(
            actual == expected_sha,
            (
                "Frozen dependency changed:\n"
                f"{path}\n"
                f"expected={expected_sha}\n"
                f"actual={actual}"
            ),
        )

    evaluator = safe_json(
        EVALUATOR_REPORT
    )

    require(
        evaluator.get("status")
        ==
        "PASS_EVALUATOR_REFERENCE_CONTRACT_FROZEN",
        (
            "Evaluator reference contract "
            "is not frozen PASS."
        ),
    )

    require(
        evaluator[
            "scientific_boundary"
        ][
            "development_performance_metrics_computed"
        ]
        is False,
        (
            "Development metrics were already "
            "computed before bound-rule freeze."
        ),
    )

    require(
        evaluator[
            "scientific_boundary"
        ][
            "formal_content_opened"
        ]
        is False,
        (
            "Formal content was already opened."
        ),
    )

    print(
        "evaluator reference contract = EXACT PASS"
    )
    print(
        "post-actuation I_final       = FROZEN"
    )
    print(
        "constructed oracle           = FROZEN"
    )
    print(
        "road ROI                     = FROZEN"
    )
    print(
        "metric authority             = EXACT PASS"
    )

    # --------------------------------------------------------
    # B. Outcome boundary before writing rule
    # --------------------------------------------------------

    print()
    print(
        "===== B. PRE-OUTCOME BOUNDARY ====="
    )

    print(
        "development metric values = NOT READ"
    )
    print(
        "predictive outcomes        = NOT READ"
    )
    print(
        "reactive outcomes          = NOT READ"
    )
    print(
        "formal content             = NOT OPENED"
    )
    print(
        "exact NI deltas            = NOT YET COMPUTED"
    )
    print(
        "policy tuning              = NO"
    )

    # --------------------------------------------------------
    # C. Existing regression
    # --------------------------------------------------------

    print()
    print(
        "===== C. PRE-FREEZE REGRESSION ====="
    )

    rc_pre, n_pre, out_pre = run_tests(
        "test_*.py"
    )

    require(
        rc_pre == 0,
        (
            "Pre-freeze regression failed:\n"
            +
            tail(out_pre)
        ),
    )

    require(
        n_pre == EXPECTED_PRE_TESTS,
        (
            f"Expected {EXPECTED_PRE_TESTS} "
            f"pre-freeze tests; got {n_pre}."
        ),
    )

    print(
        "pre-freeze regression =",
        f"{n_pre} / {n_pre} PASS",
    )

    # --------------------------------------------------------
    # D. Build frozen rule
    # --------------------------------------------------------

    print()
    print(
        "===== D. BUILD PRE-OUTCOME RULE ====="
    )

    rule = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "status":
            "FROZEN_PREOUTCOME_NI_BOUND_SELECTION_RULE",

        "freeze_timing":
            "before_any_development_performance_metric",

        "purpose":
            (
                "convert development-only original-reactive "
                "comparator variability into fixed formal "
                "non-inferiority tolerances without using "
                "class-aware predictive outcomes"
            ),

        "primary_comparison":
            (
                "class_aware_predictive_ADB_vs_"
                "original_reactive_ADB"
            ),

        "vehicle_shadow_zone_violation": {
            "tolerance":
                None,

            "formal_rule":
                "predictive_mean < reactive_mean",

            "direction":
                "strict_improvement_required",
        },

        "noninferiority_metrics": {
            "over_masking_area": {
                "direction":
                    "lower_is_better",

                "formal_rule":
                    (
                        "predictive_minus_reactive "
                        "<= frozen_delta"
                    ),

                "required_supported_development_observations":
                    120,
            },

            "pedestrian_visibility_proxy": {
                "direction":
                    "higher_is_better",

                "formal_rule":
                    (
                        "predictive_minus_reactive "
                        ">= -frozen_delta"
                    ),

                "minimum_supported_development_observations":
                    20,
            },

            "cyclist_visibility_proxy": {
                "direction":
                    "higher_is_better",

                "formal_rule":
                    (
                        "predictive_minus_reactive "
                        ">= -frozen_delta"
                    ),

                "minimum_supported_development_observations":
                    20,
            },
        },

        "bound_derivation": {
            "source_values":
                (
                    "original_reactive_ADB "
                    "development values only"
                ),

            "sample_standard_deviation":
                "ddof=1",

            "standard_error":
                "sample_std / sqrt(n)",

            "variability_scale":
                1.96,

            "raw_delta":
                "1.96 * standard_error",

            "absolute_floor":
                0.02,

            "absolute_cap":
                0.05,

            "final_delta":
                "clip(raw_delta, 0.02, 0.05)",

            "units":
                "absolute normalized metric units",

            "confidence_interval_claim":
                False,

            "interpretation":
                (
                    "pre-frozen comparator-variability "
                    "scale, not a CI claim"
                ),
        },

        "data_dependency": {
            "allowed": [
                "original_reactive_ADB_development_values",
            ],

            "class_aware_predictive_values_allowed":
                False,

            "other_predictive_baseline_values_allowed":
                False,

            "constructed_oracle_values_allowed_for_bound_size":
                False,

            "formal_values_allowed":
                False,
        },

        "anti_posthoc_guards": {
            "predictive_values_must_not_enter_delta_function":
                True,

            "bound_rule_modification_after_development_metrics":
                False,

            "bound_rule_modification_after_formal_metrics":
                False,

            "formal_parameter_tuning":
                False,

            "if_support_requirement_fails":
                (
                    "BLOCK_STAGE6; do not lower support "
                    "requirement after seeing outcomes"
                ),
        },

        "exact_delta_freeze_timing":
            (
                "after development reactive comparator "
                "metrics are computed and before formal "
                "evaluation"
            ),

        "formal_evaluation_allowed_after_this_rule_freeze":
            False,

        "next":
            (
                "run actual development-only ADB evaluator, "
                "derive the three deltas from reactive-only "
                "values, then freeze exact acceptance policy"
            ),
    }

    config_bytes = canonical_json_bytes(
        rule
    )

    module_bytes = MODULE_SOURCE.encode(
        "utf-8"
    )

    test_bytes = TEST_SOURCE.encode(
        "utf-8"
    )

    json.loads(
        config_bytes.decode("utf-8")
    )

    compile(
        MODULE_SOURCE,
        str(RULE_MODULE),
        "exec",
    )

    compile(
        TEST_SOURCE,
        str(RULE_TEST),
        "exec",
    )

    print(
        "rule JSON syntax   = PASS"
    )
    print(
        "runtime syntax     = PASS"
    )
    print(
        "test syntax        = PASS"
    )
    print(
        "delta source       = REACTIVE COMPARATOR ONLY"
    )
    print(
        "scale              = 1.96"
    )
    print(
        "absolute floor     = 0.02"
    )
    print(
        "absolute cap       = 0.05"
    )
    print(
        "VRU min support    = 20"
    )
    print(
        "overmask support   = 120 EXACT"
    )

    # --------------------------------------------------------
    # E. Transactional new-file freeze
    # --------------------------------------------------------

    print()
    print(
        "===== E. TRANSACTIONAL WRITE ====="
    )

    targets = (
        RULE_CONFIG,
        RULE_MODULE,
        RULE_TEST,
    )

    existing_targets = [
        str(path)
        for path in targets
        if path.exists()
    ]

    require(
        not existing_targets,
        (
            "Unexpected pre-existing rule files "
            "without frozen PASS report: "
            +
            repr(existing_targets)
        ),
    )

    written = []

    try:

        for path, data in (
            (RULE_CONFIG, config_bytes),
            (RULE_MODULE, module_bytes),
            (RULE_TEST, test_bytes),
        ):

            atomic_write(
                path,
                data,
            )

            written.append(path)

        print(
            "rule config  = WRITTEN"
        )
        print(
            "rule runtime = WRITTEN"
        )
        print(
            "rule tests   = WRITTEN"
        )

        # ----------------------------------------------------
        # F. Targeted
        # ----------------------------------------------------

        print()
        print(
            "===== F. TARGETED TESTS ====="
        )

        rc_target, n_target, out_target = run_tests(
            "test_block68_noninferiority_bound_rule.py"
        )

        require(
            rc_target == 0,
            (
                "Targeted tests failed:\n"
                +
                tail(out_target)
            ),
        )

        require(
            n_target == NEW_TESTS,
            (
                f"Expected {NEW_TESTS} targeted tests; "
                f"got {n_target}."
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

        rc_post, n_post, out_post = run_tests(
            "test_*.py"
        )

        require(
            rc_post == 0,
            (
                "Full Stage6 regression failed:\n"
                +
                tail(out_post)
            ),
        )

        require(
            n_post == EXPECTED_POST_TESTS,
            (
                f"Expected {EXPECTED_POST_TESTS} "
                f"post-freeze tests; got {n_post}."
            ),
        )

        print(
            "full Stage6 regression =",
            f"{n_post} / {n_post} PASS",
        )

    except BaseException:

        print()
        print(
            "RULE FREEZE FAILURE -> ROLLBACK NEW FILES"
        )

        for path in reversed(written):

            try:

                if path.exists():
                    path.unlink()

            except Exception:
                pass

        rc_roll, n_roll, out_roll = run_tests(
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
                tail(out_roll)
            ),
        )

        require(
            n_roll == EXPECTED_PRE_TESTS,
            (
                "Rollback did not restore "
                "expected test population."
            ),
        )

        raise

    # --------------------------------------------------------
    # H. Readback
    # --------------------------------------------------------

    print()
    print(
        "===== H. EXACT RULE READBACK ====="
    )

    frozen = safe_json(
        RULE_CONFIG
    )

    require(
        frozen["status"]
        ==
        "FROZEN_PREOUTCOME_NI_BOUND_SELECTION_RULE",
        (
            "Rule status changed."
        ),
    )

    require(
        frozen[
            "bound_derivation"
        ][
            "variability_scale"
        ]
        ==
        1.96,
        (
            "Scale changed."
        ),
    )

    require(
        frozen[
            "bound_derivation"
        ][
            "absolute_floor"
        ]
        ==
        0.02,
        (
            "Floor changed."
        ),
    )

    require(
        frozen[
            "bound_derivation"
        ][
            "absolute_cap"
        ]
        ==
        0.05,
        (
            "Cap changed."
        ),
    )

    require(
        frozen[
            "data_dependency"
        ][
            "class_aware_predictive_values_allowed"
        ]
        is False,
        (
            "Predictive outcomes leaked into "
            "bound-size rule."
        ),
    )

    require(
        frozen[
            "data_dependency"
        ][
            "formal_values_allowed"
        ]
        is False,
        (
            "Formal outcomes allowed unexpectedly."
        ),
    )

    print(
        "predictive outcome dependency = NONE"
    )
    print(
        "formal outcome dependency     = NONE"
    )
    print(
        "vehicle gate                  = STRICT IMPROVEMENT"
    )
    print(
        "NI rule                       = FROZEN"
    )
    print(
        "exact deltas                  = NOT YET COMPUTED"
    )

    # --------------------------------------------------------
    # I. Upstream immutability
    # --------------------------------------------------------

    print()
    print(
        "===== I. UPSTREAM IMMUTABILITY ====="
    )

    for path, expected_sha in EXPECTED.items():

        require(
            sha256_file(path)
            ==
            expected_sha,
            f"Upstream changed: {path}",
        )

    print(
        "controller runtime    = UNCHANGED"
    )
    print(
        "metric runtime        = UNCHANGED"
    )
    print(
        "evaluator contract    = UNCHANGED"
    )
    print(
        "road ROI              = UNCHANGED"
    )
    print(
        "scientific policies   = UNCHANGED"
    )

    # --------------------------------------------------------
    # J. Storage
    # --------------------------------------------------------

    print()
    print(
        "===== J. STORAGE ====="
    )

    free_gib = (
        shutil.disk_usage(ROOT).free
        /
        1024**3
    )

    require(
        free_gib >= MIN_FREE_GIB,
        (
            "250-GiB storage reserve violated."
        ),
    )

    print(
        "free GiB        =",
        round(free_gib, 3),
    )
    print(
        "250-GiB reserve = PASS"
    )

    # --------------------------------------------------------
    # K. Report
    # --------------------------------------------------------

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "status":
            "PASS_NI_BOUND_SELECTION_RULE_FROZEN",

        "freeze_timing":
            "before_any_development_performance_metric",

        "rule": {
            "source":
                "original_reactive_ADB_development_only",

            "scale":
                1.96,

            "floor":
                0.02,

            "cap":
                0.05,

            "VRU_min_support":
                20,

            "overmask_required_support":
                120,
        },

        "scientific_boundary": {
            "development_metric_values_read":
                False,

            "predictive_outcomes_read":
                False,

            "reactive_outcomes_read":
                False,

            "exact_deltas_computed":
                False,

            "formal_content_opened":
                False,

            "formal_outcomes_read":
                False,

            "formal_evaluation":
                False,

            "policy_tuning":
                False,

            "parameter_tuning":
                False,
        },

        "frozen_outputs": {
            "config": {
                "path":
                    str(RULE_CONFIG),

                "sha256":
                    sha256_file(RULE_CONFIG),
            },

            "module": {
                "path":
                    str(RULE_MODULE),

                "sha256":
                    sha256_file(RULE_MODULE),
            },

            "test": {
                "path":
                    str(RULE_TEST),

                "sha256":
                    sha256_file(RULE_TEST),
            },
        },

        "upstream": {
            str(path):
                expected_sha
            for path, expected_sha
            in EXPECTED.items()
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
                "actual development-only reactive and "
                "class-aware predictive ADB metric execution; "
                "derive exact three deltas using the frozen "
                "reactive-only rule; freeze formal acceptance "
                "policy before Block6.9"
            ),
    }

    atomic_write(
        REPORT,
        canonical_json_bytes(result),
    )

    print()
    print(
        "===== K. FREEZE REPORT ====="
    )

    print(
        "rule config SHA256 =",
        sha256_file(RULE_CONFIG),
    )

    print(
        "rule module SHA256 =",
        sha256_file(RULE_MODULE),
    )

    print(
        "freeze report SHA256 =",
        sha256_file(REPORT),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 NI BOUND-SELECTION RULE — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "development metrics        = NOT READ"
    )
    print(
        "predictive outcomes         = NOT READ"
    )
    print(
        "reactive outcomes           = NOT READ"
    )
    print(
        "formal content              = NOT OPENED"
    )
    print(
        "bound source                = REACTIVE COMPARATOR ONLY"
    )
    print(
        "variability scale           = 1.96"
    )
    print(
        "absolute delta floor        = 0.02"
    )
    print(
        "absolute delta cap          = 0.05"
    )
    print(
        "VRU minimum support         = 20"
    )
    print(
        "overmask required support   = 120"
    )
    print(
        "vehicle violation tolerance = NONE"
    )
    print(
        "vehicle formal gate         = STRICT IMPROVEMENT"
    )
    print(
        "exact NI deltas             = NOT YET COMPUTED"
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
        f"{n_post} / {n_post} PASS",
    )
    print(
        "formal evaluation allowed   = NO"
    )
    print(
        "STATUS = PASS_NI_BOUND_SELECTION_RULE_FROZEN"
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
        "BLOCK 6.8 NI BOUND-SELECTION RULE = BLOCKED"
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
        "predictive outcomes read        = NO"
    )
    print(
        "reactive outcomes read          = NO"
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
