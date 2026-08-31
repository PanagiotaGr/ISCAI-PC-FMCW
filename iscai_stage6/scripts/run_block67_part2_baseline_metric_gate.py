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

S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

CONTRACT = (
    S6
    / "configs/stage6_contract.json"
)

PART1 = (
    S6
    / "reports/"
      "block67_part1_margin_binding_gate.json"
)

POLICY = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

MARGIN_BINDING = (
    S6
    / "src/iscai_stage6/adb/"
      "class_margin_binding.py"
)

RUNTIME_CONTRACT = (
    S6
    / "src/iscai_stage6/adb/"
      "baseline_metric_contract.py"
)

TARGET_TEST = (
    S6
    / "tests/"
      "test_block67_baseline_metric_contract.py"
)

REPORT = (
    S6
    / "reports/"
      "block67_part2_baseline_metric_gate.json"
)

CLOSURE = (
    S6
    / "reports/"
      "block67_final_closure.json"
)

FAILURE = (
    S6
    / "reports/"
      "block67_part2_failure.json"
)

EXPECTED_POLICY_SHA = (
    "b998f468b98c2770c48c84a2c0aaaa2"
    "177c2d84b8fc838e80fef9b9da1ebc14b"
)

MIN_FREE_GIB = 250.0


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


def load_json(
    path: Path,
):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def require(
    condition,
    message: str,
):

    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


def write_json(
    path: Path,
    payload,
):

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
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

    return (
        process.returncode,
        process.stdout,
    )


def count_tests(
    output: str,
):

    matches = re.findall(
        r"Ran\s+(\d+)\s+tests?",
        output,
    )

    if not matches:
        return None

    return int(
        matches[-1]
    )


def concise_failure(
    output: str,
):

    return "\n".join(
        output.splitlines()[
            -80:
        ]
    )


def protected_paths():

    items = [
        S5
        / "reports/stage5_final_closure.json",

        S5
        / "artifacts/block510/"
          "stage5_to_stage6_handoff.json",

        S5
        / "artifacts/block510/"
          "stage5_final_freeze_manifest.json",

        CONTRACT,
        PART1,
        POLICY,
        MARGIN_BINDING,
    ]

    # Frozen evidence from Blocks6.0–6.6.
    for block in range(
        0,
        7,
    ):
        items.extend(
            sorted(
                (
                    S6
                    / "reports"
                ).glob(
                    f"block6{block}*.json"
                )
            )
        )

    unique = []
    seen = set()

    for path in items:

        if not path.is_file():
            continue

        # Never protect the report being created here.
        if path in (
            REPORT,
            CLOSURE,
            FAILURE,
        ):
            continue

        key = str(
            path
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            path
        )

    return tuple(
        unique
    )


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.7 PART 2/2"
    )
    print(
        "BASELINE MATRIX + ORACLE/METRIC INTERFACE GATE"
    )
    print(
        "============================================================"
    )

    required = (
        CONTRACT,
        PART1,
        POLICY,
        MARGIN_BINDING,
        RUNTIME_CONTRACT,
        TARGET_TEST,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing Block6.7 dependency: "
            +
            ", ".join(
                missing
            )
        ),
    )

    protected = (
        protected_paths()
    )

    before = {
        str(path):
            sha256_file(
                path
            )
        for path in protected
    }

    # ========================================================
    # A. Part1 continuity
    # ========================================================

    print()
    print(
        "===== A. PART1 CONTINUITY ====="
    )

    part1 = load_json(
        PART1
    )

    require(
        part1.get(
            "status"
        )
        ==
        "PASS_RUNTIME_BINDING_FROZEN",
        (
            "Block6.7 Part1 is not frozen PASS."
        ),
    )

    require(
        sha256_file(
            POLICY
        )
        ==
        EXPECTED_POLICY_SHA,
        (
            "Frozen class_aware_policy.py changed."
        ),
    )

    require(
        part1[
            "runtime_binding"
        ][
            "symbolic_semantics"
        ]
        ==
        "m_extra_only",
        (
            "Part1 runtime binding is not "
            "m_extra-only."
        ),
    )

    require(
        part1[
            "runtime_binding"
        ][
            "part_a_generic_margin_reapplied"
        ]
        is False,
        (
            "Part-A generic margin "
            "would be double counted."
        ),
    )

    previous_full_tests = int(
        part1[
            "regression"
        ][
            "full_stage6_tests"
        ]
    )

    print(
        "Part1 status              = PASS"
    )

    print(
        "m_extra runtime binding   = FROZEN PASS"
    )

    print(
        "previous Stage6 tests     =",
        previous_full_tests,
    )

    # ========================================================
    # B. Frozen Stage6 contract
    # ========================================================

    print()
    print(
        "===== B. STRICT STAGE6 CONTRACT CROSS-CHECK ====="
    )

    contract = load_json(
        CONTRACT
    )

    contract_baselines = set(
        contract[
            "adb_baselines"
        ][
            "required_pdf_labels"
        ]
    )

    contract_metrics = set(
        contract[
            "adb_metrics"
        ][
            "required"
        ]
    )

    from iscai_stage6.adb.baseline_metric_contract import (
        ADBBaseline,
        ADBMetric,
        ConstructedOracleContract,
        required_baseline_specs,
        required_metric_interfaces,
    )

    runtime_baselines = {
        item.value
        for item in ADBBaseline
    }

    runtime_metrics = {
        item.value
        for item in ADBMetric
    }

    require(
        runtime_baselines
        ==
        contract_baselines,
        (
            "Runtime ADB baseline labels do not "
            "exactly match frozen Stage6 contract."
        ),
    )

    require(
        runtime_metrics
        ==
        contract_metrics,
        (
            "Runtime ADB metric labels do not "
            "exactly match frozen Stage6 contract."
        ),
    )

    require(
        len(
            runtime_baselines
        )
        ==
        7,
        "Required baseline count != 7.",
    )

    require(
        len(
            runtime_metrics
        )
        ==
        12,
        "Required metric count != 12.",
    )

    print(
        "PDF baseline labels       = 7 / 7 PASS"
    )

    print(
        "PDF metric labels         = 12 / 12 PASS"
    )

    # ========================================================
    # C. Baseline causality matrix
    # ========================================================

    print()
    print(
        "===== C. BASELINE CAUSALITY MATRIX ====="
    )

    specs = (
        required_baseline_specs()
    )

    require(
        len(specs)
        ==
        7,
        "Runtime baseline spec count != 7.",
    )

    for spec in specs:

        require(
            spec.future_truth_controller_input
            is False,
            (
                f"{spec.label.value} incorrectly "
                "uses future truth as controller input."
            ),
        )

        if (
            spec.label
            !=
            ADBBaseline.ORACLE_FUTURE_ADB
        ):
            require(
                spec.evaluator_future_truth
                is False,
                (
                    f"{spec.label.value} incorrectly "
                    "uses future evaluator truth."
                ),
            )

        print(
            f"{spec.label.value:38s}"
            f" | predictive={str(spec.predictive):5s}"
            f" | uncertainty="
            f"{str(spec.predictive_uncertainty):5s}"
            f" | class-aware="
            f"{str(spec.class_aware_policy):5s}"
            f" | futureGT-controller=NO"
        )

    primary = [
        spec
        for spec in specs
        if (
            spec.label
            ==
            ADBBaseline.CLASS_AWARE_PREDICTIVE_ADB
        )
    ]

    require(
        len(primary)
        ==
        1,
        (
            "Primary class-aware predictive "
            "baseline not unique."
        ),
    )

    primary = primary[0]

    require(
        primary.predictive
        and
        primary.future_mean
        and
        primary.predictive_uncertainty
        and
        primary.class_aware_policy,
        (
            "Primary Stage6 baseline semantics "
            "are incomplete."
        ),
    )

    print()
    print(
        "primary Stage6 controller = "
        "CLASS-AWARE PREDICTIVE PASS"
    )

    # ========================================================
    # D. Honest overlap semantics
    # ========================================================

    print()
    print(
        "===== D. PDF LABEL OVERLAP DISCLOSURE ====="
    )

    uncertainty = next(
        spec
        for spec in specs
        if (
            spec.label
            ==
            ADBBaseline.UNCERTAINTY_AWARE_PREDICTIVE_ADB
        )
    )

    agnostic = next(
        spec
        for spec in specs
        if (
            spec.label
            ==
            ADBBaseline.CLASS_AGNOSTIC_PREDICTIVE_ADB
        )
    )

    require(
        uncertainty.runtime_configuration_id
        ==
        agnostic.runtime_configuration_id,
        (
            "Overlapping PDF labels were "
            "silently given artificial semantics."
        ),
    )

    require(
        contract[
            "adb_baselines"
        ][
            "artificial_algorithm_difference_for_label_only"
        ]
        is False,
        (
            "Frozen contract permits artificial "
            "algorithm differences for labels."
        ),
    )

    print(
        "uncertainty-aware / class-agnostic"
    )

    print(
        "runtime config overlap    = EXPLICIT PASS"
    )

    print(
        "fabricated distinction    = NO"
    )

    # ========================================================
    # E. Constructed oracle boundary
    # ========================================================

    print()
    print(
        "===== E. CONSTRUCTED ORACLE BOUNDARY ====="
    )

    oracle = (
        ConstructedOracleContract()
    )

    require(
        oracle.measured_ADB_ground_truth
        is False,
        (
            "Oracle incorrectly claims "
            "measured ADB ground truth."
        ),
    )

    require(
        oracle.available_to_controller
        is False,
        (
            "Constructed oracle is available "
            "to controller."
        ),
    )

    require(
        oracle.available_for_parameter_tuning
        is False,
        (
            "Constructed oracle is available "
            "for parameter tuning."
        ),
    )

    require(
        oracle.available_for_scoring
        is True,
        (
            "Constructed oracle is not "
            "available for evaluator scoring."
        ),
    )

    require(
        oracle.same_grid_as_predictive_ADB
        is True,
        (
            "Oracle/predictive grid mismatch."
        ),
    )

    require(
        oracle.same_Part_A_intensity_model
        is True,
        (
            "Oracle changes Part-A "
            "illumination model."
        ),
    )

    print(
        "oracle type               = CONSTRUCTED"
    )

    print(
        "measured ADB GT           = NO"
    )

    print(
        "controller access         = NO"
    )

    print(
        "parameter tuning access   = NO"
    )

    print(
        "evaluator scoring         = YES"
    )

    # ========================================================
    # F. Metric-interface contract
    # ========================================================

    print()
    print(
        "===== F. METRIC INTERFACE CONTRACT ====="
    )

    metrics = (
        required_metric_interfaces()
    )

    require(
        len(metrics)
        ==
        12,
        "Metric interface count != 12.",
    )

    for metric in metrics:

        require(
            metric.controller_input_metric
            is False,
            (
                f"{metric.name.value} is "
                "incorrectly controller-facing."
            ),
        )

    oracle_metric_names = {
        metric.name.value
        for metric in metrics
        if metric.requires_constructed_oracle
    }

    require(
        {
            "mask_IoU_with_constructed_oracle_future_mask",
            "over_masking_area",
            "false_dimming",
        }.issubset(
            oracle_metric_names
        ),
        (
            "Constructed-oracle metric "
            "dependencies are incomplete."
        ),
    )

    print(
        "mandatory metrics         = 12 / 12 PASS"
    )

    print(
        "metrics as controller input= NO"
    )

    print(
        "oracle-dependent metrics  = EXPLICIT PASS"
    )

    print(
        "numerical metric formulas = NOT YET FROZEN"
    )

    print(
        "formula/ROI freeze        = BLOCK6.8"
    )

    # ========================================================
    # G. No formal/development outcome use
    # ========================================================

    print()
    print(
        "===== G. DATA-USE BOUNDARY ====="
    )

    require(
        part1[
            "causality"
        ][
            "future_GT_read"
        ]
        is False,
        "Part1 read future GT.",
    )

    require(
        part1[
            "causality"
        ][
            "formal_outcomes_read"
        ]
        is False,
        "Part1 read formal outcomes.",
    )

    require(
        part1[
            "causality"
        ][
            "development_outcomes_read"
        ]
        is False,
        "Part1 read development outcomes.",
    )

    print(
        "development outcomes      = NO"
    )

    print(
        "formal outcomes           = NO"
    )

    print(
        "future GT controller read = NO"
    )

    print(
        "policy sweep              = NO"
    )

    # ========================================================
    # H. Targeted tests
    # ========================================================

    print()
    print(
        "===== H. BLOCK6.7 PART2 TARGETED TESTS ====="
    )

    targeted_rc, targeted_output = (
        run_tests(
            "test_block67_baseline_metric_contract.py"
        )
    )

    targeted_n = count_tests(
        targeted_output
    )

    if targeted_rc != 0:
        print(
            concise_failure(
                targeted_output
            )
        )

    require(
        targeted_rc == 0,
        (
            "Block6.7 Part2 targeted "
            "tests failed."
        ),
    )

    require(
        targeted_n == 10,
        (
            "Expected exactly 10 new "
            f"Part2 tests; got {targeted_n}."
        ),
    )

    print(
        "targeted tests = 10 / 10 PASS"
    )

    # ========================================================
    # I. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== I. FULL STAGE6 REGRESSION ====="
    )

    full_rc, full_output = (
        run_tests(
            "test_*.py"
        )
    )

    full_n = count_tests(
        full_output
    )

    if full_rc != 0:
        print(
            concise_failure(
                full_output
            )
        )

    require(
        full_rc == 0,
        (
            "Full Stage6 regression failed."
        ),
    )

    expected_full_n = (
        previous_full_tests
        +
        targeted_n
    )

    require(
        full_n
        ==
        expected_full_n,
        (
            "Unexpected full regression count: "
            f"{full_n}; expected "
            f"{expected_full_n}."
        ),
    )

    print(
        "full Stage6 regression =",
        f"{full_n} / {full_n} PASS",
    )

    # ========================================================
    # J. Upstream immutability
    # ========================================================

    print()
    print(
        "===== J. IMMUTABILITY READBACK ====="
    )

    after = {
        str(path):
            sha256_file(
                path
            )
        for path in protected
    }

    changed = [
        path
        for path in before
        if before[path]
        !=
        after[path]
    ]

    require(
        not changed,
        (
            "Frozen upstream evidence changed: "
            + repr(
                changed
            )
        ),
    )

    require(
        sha256_file(
            POLICY
        )
        ==
        EXPECTED_POLICY_SHA,
        (
            "class_aware_policy.py changed."
        ),
    )

    print(
        "Stage5 frozen evidence    = UNCHANGED"
    )

    print(
        "Blocks6.0–6.6 evidence    = UNCHANGED"
    )

    print(
        "Block6.7 Part1 evidence   = UNCHANGED"
    )

    print(
        "class-aware policy        = UNCHANGED"
    )

    # ========================================================
    # K. Storage
    # ========================================================

    print()
    print(
        "===== K. STORAGE ====="
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
        "250-GiB reserve violated.",
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
    # L. Part2 report + Block6.7 closure
    # ========================================================

    print()
    print(
        "===== L. BLOCK6.7 CLOSURE ====="
    )

    runtime_sha = (
        sha256_file(
            RUNTIME_CONTRACT
        )
    )

    part2_report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.7",

        "part":
            "2/2",

        "status":
            "PASS_BASELINE_METRIC_CONTRACT_FROZEN",

        "runtime_contract": {
            "path":
                str(
                    RUNTIME_CONTRACT
                ),

            "sha256":
                runtime_sha,

            "baseline_count":
                7,

            "metric_count":
                12,
        },

        "baselines": {
            "labels":
                sorted(
                    runtime_baselines
                ),

            "future_truth_controller_input":
                False,

            "primary":
                "class_aware_predictive_ADB",

            "uncertainty_class_agnostic_overlap":
                "EXPLICIT_SHARED_RUNTIME_CONFIGURATION",

            "fabricated_label_only_algorithm_difference":
                False,
        },

        "constructed_oracle": {
            "measured_ADB_ground_truth":
                False,

            "controller_access":
                False,

            "parameter_tuning_access":
                False,

            "evaluator_scoring_access":
                True,

            "same_grid":
                True,

            "same_Part_A_intensity_model":
                True,
        },

        "metric_interface": {
            "labels":
                sorted(
                    runtime_metrics
                ),

            "formula_freeze":
                "BLOCK6.8",

            "ROI_freeze":
                "BLOCK6.8",

            "acceptance_bound_freeze":
                "BLOCK6.8",

            "controller_input_metrics":
                False,
        },

        "regression": {
            "new_targeted_tests":
                targeted_n,

            "previous_stage6_tests":
                previous_full_tests,

            "full_stage6_tests":
                full_n,

            "status":
                "PASS",
        },

        "causality": {
            "development_outcomes_read":
                False,

            "formal_outcomes_read":
                False,

            "future_GT_controller_read":
                False,

            "policy_sweep":
                False,

            "parameter_tuning":
                False,
        },

        "next":
            (
                "Block6.8: freeze numerical ADB "
                "metric formulas, ROIs, development-only "
                "acceptance bounds and pre-formal policy"
            ),
    }

    write_json(
        REPORT,
        part2_report,
    )

    closure_payload = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.7",

        "status":
            "PASS_COMPLETE",

        "Part1":
            {
                "status":
                    "PASS_RUNTIME_BINDING_FROZEN",

                "report":
                    str(
                        PART1
                    ),

                "sha256":
                    sha256_file(
                        PART1
                    ),
            },

        "Part2":
            {
                "status":
                    "PASS_BASELINE_METRIC_CONTRACT_FROZEN",

                "report":
                    str(
                        REPORT
                    ),

                "sha256":
                    sha256_file(
                        REPORT
                    ),
            },

        "frozen_scientific_contract": {
            "class_margin_runtime_binding":
                "m_extra_only",

            "Part_A_generic_margin_reapplied":
                False,

            "ADB_baselines":
                7,

            "ADB_metrics":
                12,

            "oracle":
                "CONSTRUCTED_EVALUATOR_ONLY",

            "metric_formula_freeze":
                "BLOCK6.8",
        },

        "formal_evaluation_started":
            False,

        "formal_parameter_tuning":
            False,

        "upstream_modified":
            False,

        "full_stage6_regression":
            {
                "tests":
                    full_n,

                "status":
                    "PASS",
            },
    }

    write_json(
        CLOSURE,
        closure_payload,
    )

    if FAILURE.exists():
        FAILURE.unlink()

    print(
        "Part2 report SHA256 =",
        sha256_file(
            REPORT
        ),
    )

    print(
        "Block6.7 closure SHA256 =",
        sha256_file(
            CLOSURE
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.7 PART 2/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 m_extra binding      = FROZEN PASS"
    )

    print(
        "ADB baseline matrix        = 7 / 7 PASS"
    )

    print(
        "ADB metric interface       = 12 / 12 PASS"
    )

    print(
        "primary predictive ADB     = "
        "CLASS-AWARE + UNCERTAINTY"
    )

    print(
        "constructed oracle         = "
        "EVALUATOR ONLY PASS"
    )

    print(
        "measured ADB ground truth  = NO"
    )

    print(
        "future GT controller input = NO"
    )

    print(
        "label-overlap disclosure   = PASS"
    )

    print(
        "metric formulas frozen     = NOT YET / BLOCK6.8"
    )

    print(
        "development outcomes       = NO"
    )

    print(
        "formal outcomes            = NO"
    )

    print(
        "policy tuning              = NO"
    )

    print(
        "targeted regression        =",
        f"{targeted_n} / {targeted_n} PASS",
    )

    print(
        "full Stage6 regression     =",
        f"{full_n} / {full_n} PASS",
    )

    print(
        "upstream modified          = NO"
    )

    print(
        "STATUS = PASS_COMPLETE"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "closure =",
        CLOSURE,
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    payload = {
        "stage":
            6,

        "block":
            "6.7",

        "part":
            "2/2",

        "status":
            "BLOCKED",

        "exception_type":
            type(
                exc
            ).__name__,

        "reason":
            str(
                exc
            ),

        "recovery":
            (
                "Do not run Block6.8 or formal "
                "evaluation. Do not modify frozen "
                "Blocks6.0–6.7 Part1. Send this "
                "BLOCKED section for targeted repair."
            ),

        "formal_evaluation":
            False,

        "parameter_tuning":
            False,
    }

    try:
        write_json(
            FAILURE,
            payload,
        )
    except Exception:
        pass

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.7 PART 2/2 = BLOCKED"
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
        "RECOVERY:"
    )

    print(
        "Do not run Block6.8."
    )

    print(
        "Do not run formal evaluation."
    )

    print(
        "Do not tune policies."
    )

    print(
        "Send this BLOCKED section only."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
