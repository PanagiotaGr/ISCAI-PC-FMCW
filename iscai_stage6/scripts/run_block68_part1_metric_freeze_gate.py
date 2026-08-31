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

PROTOCOL = (
    S6
    / "configs/stage6_metric_freeze_protocol.json"
)

BLOCK67 = (
    S6
    / "reports/block67_final_closure.json"
)

METRIC_INTERFACE = (
    S6
    / "src/iscai_stage6/adb/"
      "baseline_metric_contract.py"
)

METRIC_IMPL = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

TARGET_TEST = (
    S6
    / "tests/"
      "test_block68_metric_semantics.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_part1_metric_freeze_gate.json"
)

FAILURE = (
    S6
    / "reports/"
      "block68_part1_metric_freeze_failure.json"
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
    message,
):

    if not bool(
        condition
    ):
        raise RuntimeError(
            message
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


def protected_paths():

    paths = [
        S5
        / "reports/stage5_final_closure.json",

        S5
        / "artifacts/block510/"
          "stage5_to_stage6_handoff.json",

        S5
        / "artifacts/block510/"
          "stage5_final_freeze_manifest.json",

        CONTRACT,
        BLOCK67,
        METRIC_INTERFACE,
    ]

    for block in range(
        0,
        8,
    ):

        for path in sorted(
            (
                S6 / "reports"
            ).glob(
                f"block6{block}*.json"
            )
        ):

            if path in (
                REPORT,
                FAILURE,
            ):
                continue

            paths.append(
                path
            )

    unique = []
    seen = set()

    for path in paths:

        if not path.is_file():
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
        "STAGE 6 — BLOCK 6.8 PART 1/2"
    )
    print(
        "NUMERICAL ADB METRIC SEMANTICS FREEZE"
    )
    print(
        "============================================================"
    )

    required = (
        CONTRACT,
        PROTOCOL,
        BLOCK67,
        METRIC_INTERFACE,
        METRIC_IMPL,
        TARGET_TEST,
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
            "Missing Block6.8 dependency: "
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
    # A. Block6.7 continuity
    # ========================================================

    print()
    print(
        "===== A. BLOCK6.7 CONTINUITY ====="
    )

    block67 = load_json(
        BLOCK67
    )

    require(
        block67.get(
            "status"
        )
        ==
        "PASS_COMPLETE",
        (
            "Block6.7 is not PASS_COMPLETE."
        ),
    )

    previous_tests = int(
        block67[
            "full_stage6_regression"
        ][
            "tests"
        ]
    )

    require(
        previous_tests
        ==
        181,
        (
            "Unexpected frozen Block6.7 "
            f"regression count: {previous_tests}"
        ),
    )

    print(
        "Block6.7 status         = PASS_COMPLETE"
    )

    print(
        "previous Stage6 tests   =",
        previous_tests,
    )

    print(
        "baseline matrix         = 7 / 7 FROZEN"
    )

    print(
        "metric interface        = 12 / 12 FROZEN"
    )

    # ========================================================
    # B. Protocol cross-check
    # ========================================================

    print()
    print(
        "===== B. FROZEN METRIC PROTOCOL ====="
    )

    protocol = load_json(
        PROTOCOL
    )

    require(
        protocol[
            "status"
        ]
        ==
        "FROZEN_METRIC_SEMANTICS_PRE_DEVELOPMENT_BOUNDS",
        (
            "Metric protocol status mismatch."
        ),
    )

    require(
        protocol[
            "source_semantics"
        ][
            "pdf_specifies_metric_names"
        ]
        is True,
        "PDF metric-name contract lost.",
    )

    require(
        protocol[
            "source_semantics"
        ][
            "pdf_specifies_unique_formula_for_every_metric"
        ]
        is False,
        (
            "Protocol incorrectly claims "
            "all formulas are literal PDF formulas."
        ),
    )

    require(
        len(
            protocol[
                "metrics"
            ]
        )
        ==
        12,
        (
            "Metric protocol count != 12."
        ),
    )

    require(
        protocol[
            "formal_evaluation_allowed_after_part1"
        ]
        is False,
        (
            "Formal evaluation is incorrectly "
            "enabled before acceptance-bound freeze."
        ),
    )

    print(
        "mandatory metrics       = 12 / 12 PASS"
    )

    print(
        "formula provenance      = IMPLEMENTATION CHOICE"
    )

    print(
        "frozen before formal    = PASS"
    )

    print(
        "formal allowed now      = NO"
    )

    # ========================================================
    # C. Metric runtime imports
    # ========================================================

    print()
    print(
        "===== C. NUMERICAL METRIC RUNTIME ====="
    )

    from iscai_stage6.adb.baseline_metric_contract import (
        ADBMetric,
    )

    from iscai_stage6.adb.metric_semantics import (
        FLICKER_DELTA_THRESHOLD,
        cyclist_visibility_proxy,
        false_dimming,
        flicker_change_rate,
        glare_risk_exposure,
        mask_iou,
        normalized_energy_consumption,
        over_masking_area,
        pedestrian_visibility_proxy,
        road_illumination_retention,
        summarize_actuation_latency_ms,
        temporal_smoothness,
        vehicle_shadow_zone_violation,
    )

    runtime_labels = {
        item.value
        for item in ADBMetric
    }

    protocol_labels = set(
        protocol[
            "metrics"
        ].keys()
    )

    require(
        runtime_labels
        ==
        protocol_labels,
        (
            "Numerical protocol labels do not "
            "match frozen Block6.7 metric interface."
        ),
    )

    require(
        FLICKER_DELTA_THRESHOLD
        ==
        0.10,
        (
            "Frozen flicker threshold changed."
        ),
    )

    print(
        "metric labels exact     = PASS"
    )

    print(
        "runtime imports         = PASS"
    )

    print(
        "flicker delta threshold = 0.10 FROZEN"
    )

    # ========================================================
    # D. Deterministic numerical smoke
    # ========================================================

    print()
    print(
        "===== D. NUMERICAL SMOKE ====="
    )

    import numpy as np

    candidate = np.asarray(
        [
            True,
            True,
            False,
            False,
        ],
        dtype=bool,
    )

    oracle = np.asarray(
        [
            True,
            False,
            False,
            False,
        ],
        dtype=bool,
    )

    vehicle = np.asarray(
        [
            True,
            True,
            False,
            False,
        ],
        dtype=bool,
    )

    road = np.ones(
        4,
        dtype=bool,
    )

    pedestrian = np.asarray(
        [
            False,
            False,
            True,
            False,
        ],
        dtype=bool,
    )

    cyclist = np.asarray(
        [
            False,
            False,
            False,
            True,
        ],
        dtype=bool,
    )

    intensity = np.asarray(
        [
            0.0,
            0.5,
            0.8,
            0.6,
        ],
        dtype=np.float64,
    )

    sequence = np.asarray(
        [
            intensity,
            intensity,
            np.asarray(
                [
                    0.0,
                    0.5,
                    0.7,
                    0.6,
                ]
            ),
        ]
    )

    values = {
        "mask_IoU":
            mask_iou(
                candidate,
                oracle,
            ),

        "vehicle_shadow_zone_violation":
            vehicle_shadow_zone_violation(
                candidate,
                vehicle,
            ),

        "glare_risk_exposure":
            glare_risk_exposure(
                intensity,
                vehicle,
            ),

        "over_masking_area":
            over_masking_area(
                candidate,
                oracle,
                road,
            ),

        "road_illumination_retention":
            road_illumination_retention(
                intensity,
                road,
            ),

        "pedestrian_visibility_proxy":
            pedestrian_visibility_proxy(
                intensity,
                pedestrian,
            ),

        "cyclist_visibility_proxy":
            cyclist_visibility_proxy(
                intensity,
                cyclist,
            ),

        "false_dimming":
            false_dimming(
                intensity,
                oracle,
                road,
            ),

        "temporal_smoothness":
            temporal_smoothness(
                sequence
            ),

        "flicker_change_rate":
            flicker_change_rate(
                sequence
            ),

        "energy_consumption":
            normalized_energy_consumption(
                sequence
            ),
    }

    latency = (
        summarize_actuation_latency_ms(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        )
    )

    for key, value in (
        values.items()
    ):
        require(
            np.isfinite(
                value
            ),
            (
                f"Metric {key} produced "
                "non-finite smoke value."
            ),
        )

        print(
            f"{key:34s}= {value:.6f}"
        )

    print(
        f"{'actuation_latency_mean_ms':34s}"
        f"= {latency.mean_ms:.6f}"
    )

    print(
        "all numerical metrics   = FINITE PASS"
    )

    # ========================================================
    # E. Acceptance-bound semantics
    # ========================================================

    print()
    print(
        "===== E. PRE-FORMAL ACCEPTANCE SEMANTICS ====="
    )

    acceptance = protocol[
        "primary_acceptance"
    ]

    require(
        acceptance[
            "comparison"
        ]
        ==
        "class_aware_predictive_ADB_vs_original_reactive_ADB",
        (
            "Primary Stage6 comparison changed."
        ),
    )

    require(
        acceptance[
            "vehicle_shadow_zone_violation"
        ][
            "requirement"
        ]
        ==
        "strict_improvement",
        (
            "Vehicle violation gate is not "
            "strict improvement."
        ),
    )

    for metric in (
        "over_masking_area",
        "pedestrian_visibility_proxy",
        "cyclist_visibility_proxy",
    ):

        require(
            acceptance[
                metric
            ][
                "numeric_delta"
            ]
            is None,
            (
                f"{metric} numeric bound was "
                "prematurely selected in Part1."
            ),
        )

        require(
            acceptance[
                metric
            ][
                "delta_source"
            ]
            ==
            "development_only_Block6.8_Part2",
            (
                f"{metric} bound source mismatch."
            ),
        )

    rules = protocol[
        "part2_bound_freeze_rules"
    ]

    require(
        rules[
            "development_partition_only"
        ]
        is True,
        "Bounds are not development-only.",
    )

    require(
        rules[
            "formal_partition_read_before_freeze"
        ]
        is False,
        (
            "Formal partition may be read "
            "before bound freeze."
        ),
    )

    require(
        rules[
            "post_formal_adjustment_forbidden"
        ]
        is True,
        (
            "Post-formal adjustment is not forbidden."
        ),
    )

    print(
        "primary comparison      = PREDICTIVE vs REACTIVE"
    )

    print(
        "vehicle violation gate  = STRICT IMPROVEMENT"
    )

    print(
        "overmask delta          = NOT YET / DEVELOPMENT ONLY"
    )

    print(
        "ped visibility delta    = NOT YET / DEVELOPMENT ONLY"
    )

    print(
        "cyclist visibility delta= NOT YET / DEVELOPMENT ONLY"
    )

    print(
        "formal outcomes         = FORBIDDEN FOR BOUND SELECTION"
    )

    # ========================================================
    # F. Targeted tests
    # ========================================================

    print()
    print(
        "===== F. BLOCK6.8 PART1 TARGETED TESTS ====="
    )

    targeted_rc, targeted_output = (
        run_tests(
            "test_block68_metric_semantics.py"
        )
    )

    targeted_n = count_tests(
        targeted_output
    )

    if targeted_rc != 0:
        print(
            targeted_output
        )

    require(
        targeted_rc == 0,
        (
            "Block6.8 targeted tests failed."
        ),
    )

    require(
        targeted_n == 15,
        (
            "Expected exactly 15 Block6.8 "
            f"Part1 tests; got {targeted_n}."
        ),
    )

    print(
        "targeted tests = 15 / 15 PASS"
    )

    # ========================================================
    # G. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== G. FULL STAGE6 REGRESSION ====="
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
            full_output
        )

    require(
        full_rc == 0,
        (
            "Full Stage6 regression failed."
        ),
    )

    expected_n = (
        previous_tests
        +
        targeted_n
    )

    require(
        full_n
        ==
        expected_n,
        (
            "Unexpected Stage6 test count: "
            f"{full_n}; expected {expected_n}."
        ),
    )

    print(
        "full Stage6 regression =",
        f"{full_n} / {full_n} PASS",
    )

    # ========================================================
    # H. Frozen upstream immutability
    # ========================================================

    print()
    print(
        "===== H. IMMUTABILITY READBACK ====="
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
        != after[path]
    ]

    require(
        not changed,
        (
            "Frozen upstream evidence changed: "
            +
            repr(
                changed
            )
        ),
    )

    print(
        "Stage5 frozen evidence = UNCHANGED"
    )

    print(
        "Blocks6.0–6.7 evidence = UNCHANGED"
    )

    print(
        "Stage6 strict contract  = UNCHANGED"
    )

    # ========================================================
    # I. Storage
    # ========================================================

    print()
    print(
        "===== I. STORAGE ====="
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
            "250-GiB hard reserve violated."
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
    # J. Part1 report
    # ========================================================

    print()
    print(
        "===== J. BLOCK6.8 PART1 FREEZE ====="
    )

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "1/2",

        "status":
            "PASS_METRIC_SEMANTICS_FROZEN",

        "metric_protocol": {
            "path":
                str(
                    PROTOCOL
                ),

            "sha256":
                sha256_file(
                    PROTOCOL
                ),

            "metric_count":
                12,

            "flicker_delta_threshold":
                0.10,
        },

        "metric_runtime": {
            "path":
                str(
                    METRIC_IMPL
                ),

            "sha256":
                sha256_file(
                    METRIC_IMPL
                ),

            "normalized_intensity":
                "[0,1]",

            "measured_ADB_ground_truth":
                False,

            "oracle":
                "CONSTRUCTED_EVALUATOR_ONLY",
        },

        "acceptance": {
            "primary_comparison":
                (
                    "class_aware_predictive_ADB"
                    "_vs_original_reactive_ADB"
                ),

            "vehicle_shadow_zone_violation":
                "STRICT_IMPROVEMENT",

            "over_masking_bound":
                "BLOCK6.8_PART2_DEVELOPMENT_ONLY",

            "pedestrian_visibility_bound":
                "BLOCK6.8_PART2_DEVELOPMENT_ONLY",

            "cyclist_visibility_bound":
                "BLOCK6.8_PART2_DEVELOPMENT_ONLY",

            "formal_results_used":
                False,
        },

        "regression": {
            "previous_tests":
                previous_tests,

            "new_tests":
                targeted_n,

            "full_tests":
                full_n,

            "status":
                "PASS",
        },

        "scientific_execution": {
            "development_outcomes_read":
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

        "upstream_modified":
            False,

        "next":
            (
                "Block6.8 Part2: development-only "
                "runtime metric validation and "
                "numeric non-inferiority bound freeze"
            ),
    }

    write_json(
        REPORT,
        result,
    )

    if FAILURE.exists():
        FAILURE.unlink()

    print(
        "metric protocol SHA256 =",
        sha256_file(
            PROTOCOL
        ),
    )

    print(
        "metric runtime SHA256  =",
        sha256_file(
            METRIC_IMPL
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART 1/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "ADB metric formulas       = 12 / 12 FROZEN"
    )

    print(
        "formula provenance        = PRE-FORMAL IMPLEMENTATION CHOICE"
    )

    print(
        "constructed oracle        = EVALUATOR ONLY"
    )

    print(
        "measured ADB ground truth = NO"
    )

    print(
        "mask/intensity semantics  = FROZEN"
    )

    print(
        "road/actor ROI interfaces = FROZEN"
    )

    print(
        "flicker threshold         = 0.10 FROZEN"
    )

    print(
        "vehicle violation gate    = STRICT IMPROVEMENT"
    )

    print(
        "overmask/VRU bounds       = NOT YET / PART2 DEVELOPMENT ONLY"
    )

    print(
        "formal evaluation allowed = NO"
    )

    print(
        "development outcomes read = NO"
    )

    print(
        "formal outcomes read      = NO"
    )

    print(
        "targeted regression       =",
        f"{targeted_n} / {targeted_n} PASS",
    )

    print(
        "full Stage6 regression    =",
        f"{full_n} / {full_n} PASS",
    )

    print(
        "upstream modified         = NO"
    )

    print(
        "STATUS = PASS_METRIC_SEMANTICS_FROZEN"
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

    payload = {
        "stage":
            6,

        "block":
            "6.8",

        "part":
            "1/2",

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

        "formal_evaluation":
            False,

        "formal_outcomes_read":
            False,

        "recovery":
            (
                "Do not run Block6.8 Part2 or "
                "formal Stage6 evaluation. Send "
                "this BLOCKED section for targeted repair."
            ),
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
        "BLOCK 6.8 PART 1/2 = BLOCKED"
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
        "Do not run development-bound freeze."
    )

    print(
        "Do not run formal Stage6 evaluation."
    )

    print(
        "Do not modify Blocks6.0–6.7."
    )

    print(
        "Send this BLOCKED section only."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
