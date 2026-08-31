from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np


ROOT = Path(
    "/home/agni/waymo"
)

S6 = ROOT / "iscai_stage6"

ROUTE_EXTRACT = (
    S6
    / "reports/"
      "block68_final_illumination_route_extract.json"
)

ROUTE_BIND = (
    S6
    / "reports/"
      "block68_part2_development_route_schema_bind.json"
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

MARGIN_BINDING = (
    S6
    / "src/iscai_stage6/adb/"
      "class_margin_binding.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_exact_evaluator_binding_repair.json"
)

EXPECTED_ROUTE_EXTRACT_STATUS = (
    "PASS_FINAL_ILLUMINATION_ROUTE_EXTRACTED"
)

EXPECTED_ROUTE_BIND_STATUS = (
    "PASS_DEVELOPMENT_ROUTE_SCHEMA_BOUND"
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

EXPECTED_STAGE6_TESTS = 202

MIN_FREE_GIB = 250.0


# ============================================================
# Helpers
# ============================================================

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


def write_json(
    path: Path,
    payload,
):

    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
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


def dataclass_field_names(
    cls,
):

    require(
        is_dataclass(
            cls
        ),
        (
            f"{cls.__name__} "
            "must be a dataclass"
        ),
    )

    return tuple(
        field.name
        for field in fields(
            cls
        )
    )


def run_regression():

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
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
        "EXACT EVALUATOR BINDING REPAIR"
    )
    print(
        "EVIDENCE ROUTE ONLY — NO SCIENTIFIC CHANGE"
    )
    print(
        "============================================================"
    )

    required = (
        ROUTE_EXTRACT,
        ROUTE_BIND,
        RECON,
        PROTOCOL,
        METRICS,
        INTERFACE,
        CLASS_POLICY,
        MARGIN_BINDING,
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
            "Missing binding-repair dependency: "
            +
            ", ".join(
                missing
            )
        ),
    )

    protected_before = {
        str(
            path
        ):
            sha256_file(
                path
            )
        for path in (
            RECON,
            PROTOCOL,
            METRICS,
            INTERFACE,
            CLASS_POLICY,
            MARGIN_BINDING,
        )
    }

    # ========================================================
    # A. Frozen evidence continuity
    # ========================================================

    print()
    print(
        "===== A. FROZEN EVIDENCE CONTINUITY ====="
    )

    route_extract = (
        guarded_json(
            ROUTE_EXTRACT
        )
    )

    route_bind = (
        guarded_json(
            ROUTE_BIND
        )
    )

    require(
        route_extract.get(
            "status"
        )
        ==
        EXPECTED_ROUTE_EXTRACT_STATUS,
        (
            "Final illumination route "
            "extract is not PASS."
        ),
    )

    require(
        route_bind.get(
            "status"
        )
        ==
        EXPECTED_ROUTE_BIND_STATUS,
        (
            "Development route bind "
            "is not PASS."
        ),
    )

    exact_files = (
        (
            RECON,
            EXPECTED_RECON_SHA,
            "reconciliation",
        ),
        (
            PROTOCOL,
            EXPECTED_PROTOCOL_SHA,
            "metric protocol",
        ),
        (
            METRICS,
            EXPECTED_METRICS_SHA,
            "metric runtime",
        ),
        (
            INTERFACE,
            EXPECTED_INTERFACE_SHA,
            "metric interface",
        ),
    )

    for (
        path,
        expected,
        label,
    ) in exact_files:

        require(
            sha256_file(
                path
            )
            ==
            expected,
            (
                f"{label} SHA changed."
            ),
        )

        print(
            f"{label:24s}= EXACT PASS"
        )

    print(
        "development route bind   = PASS"
    )

    print(
        "illumination extraction  = PASS"
    )

    # ========================================================
    # B. Exact final-output type proof
    # ========================================================

    print()
    print(
        "===== B. FINAL OUTPUT TYPE PROOF ====="
    )

    from iscai_stage6.adb.class_aware_policy import (
        ClassAwareComposition,
        RateLimitedSchedule,
        TYPE_VEHICLE,
        apply_actuation_rate_limit,
        compose_class_aware_illumination,
        temporal_smooth_schedule,
    )

    composition_fields = (
        dataclass_field_names(
            ClassAwareComposition
        )
    )

    rate_fields = (
        dataclass_field_names(
            RateLimitedSchedule
        )
    )

    require(
        "raw_class_aware_illumination"
        in
        composition_fields,
        (
            "ClassAwareComposition lost "
            "raw_class_aware_illumination."
        ),
    )

    require(
        "illumination"
        not in
        composition_fields,
        (
            "Unexpected direct final illumination "
            "field in ClassAwareComposition."
        ),
    )

    require(
        "illumination"
        in
        rate_fields,
        (
            "RateLimitedSchedule does not "
            "expose final illumination."
        ),
    )

    require(
        "safety_override_mask"
        in
        rate_fields,
        (
            "RateLimitedSchedule lost "
            "safety_override_mask."
        ),
    )

    print(
        "ClassAwareComposition raw I = PROVEN"
    )

    print(
        "ClassAwareComposition final I= ABSENT / EXPECTED"
    )

    print(
        "RateLimitedSchedule final I  = PROVEN"
    )

    print(
        "final actuator field          = "
        "RateLimitedSchedule.illumination"
    )

    # ========================================================
    # C. Exact producer chain
    # ========================================================

    print()
    print(
        "===== C. EXACT PRODUCER CHAIN ====="
    )

    compose_signature = str(
        inspect.signature(
            compose_class_aware_illumination
        )
    )

    smooth_signature = str(
        inspect.signature(
            temporal_smooth_schedule
        )
    )

    rate_signature = str(
        inspect.signature(
            apply_actuation_rate_limit
        )
    )

    require(
        "raw_schedule"
        in
        smooth_signature,
        (
            "temporal_smooth_schedule "
            "no longer accepts raw_schedule."
        ),
    )

    require(
        "smoothed_schedule"
        in
        rate_signature,
        (
            "apply_actuation_rate_limit "
            "no longer accepts smoothed_schedule."
        ),
    )

    require(
        "RateLimitedSchedule"
        in
        str(
            inspect.signature(
                apply_actuation_rate_limit
            ).return_annotation
        ),
        (
            "Rate-limit operator return type "
            "is not RateLimitedSchedule."
        ),
    )

    print(
        "compose:"
    )

    print(
        "  ",
        compose_signature,
    )

    print(
        "smooth:"
    )

    print(
        "  ",
        smooth_signature,
    )

    print(
        "rate limit:"
    )

    print(
        "  ",
        rate_signature,
    )

    print()
    print(
        "producer chain ="
    )

    print(
        "  ClassAwareComposition."
        "raw_class_aware_illumination"
    )

    print(
        "    -> temporal_smooth_schedule"
    )

    print(
        "    -> apply_actuation_rate_limit"
    )

    print(
        "    -> RateLimitedSchedule.illumination"
    )

    # ========================================================
    # D. Synthetic structural execution
    # ========================================================

    print()
    print(
        "===== D. SYNTHETIC STRUCTURAL EXECUTION ====="
    )

    # Pure structural smoke.
    # These are NOT scientific tuning values.
    horizons = (
        0.1,
        0.3,
        0.5,
        1.0,
    )

    theta_cells = 5
    range_cells = 7

    vehicle_map = np.ones(
        (
            4,
            theta_cells,
            range_cells,
        ),
        dtype=np.float64,
    )

    vehicle_map[
        :,
        2,
        2:5,
    ] = 0.25

    vehicle_active = (
        vehicle_map
        <
        1.0
    )

    composition = (
        compose_class_aware_illumination(
            class_illumination={
                TYPE_VEHICLE:
                    vehicle_map,
            },

            class_active_masks={
                TYPE_VEHICLE:
                    vehicle_active,
            },

            class_floors={
                TYPE_VEHICLE:
                    0.0,
            },
        )
    )

    require(
        isinstance(
            composition,
            ClassAwareComposition,
        ),
        (
            "Composition runtime type mismatch."
        ),
    )

    require(
        composition.raw_class_aware_illumination.shape
        ==
        (
            4,
            theta_cells,
            range_cells,
        ),
        (
            "Raw class-aware illumination "
            "shape mismatch."
        ),
    )

    current = np.ones(
        (
            theta_cells,
            range_cells,
        ),
        dtype=np.float64,
    )

    smoothed = (
        temporal_smooth_schedule(
            composition.raw_class_aware_illumination,

            current_illumination=current,

            horizons_s=horizons,

            # Structural smoke only.
            time_constant_s=0.2,
        )
    )

    require(
        isinstance(
            smoothed,
            np.ndarray,
        ),
        (
            "Temporal smoothing did not "
            "return ndarray."
        ),
    )

    require(
        smoothed.shape
        ==
        (
            4,
            theta_cells,
            range_cells,
        ),
        (
            "Smoothed schedule shape mismatch."
        ),
    )

    final = (
        apply_actuation_rate_limit(
            smoothed,

            current_illumination=current,

            horizons_s=horizons,

            # Structural smoke only.
            rho_dim_per_s=5.0,
            rho_bright_per_s=5.0,

            vru_floor_guard=(
                composition.vru_floor_guard
            ),
        )
    )

    require(
        isinstance(
            final,
            RateLimitedSchedule,
        ),
        (
            "Final runtime result is not "
            "RateLimitedSchedule."
        ),
    )

    require(
        final.illumination.shape
        ==
        (
            4,
            theta_cells,
            range_cells,
        ),
        (
            "Final illumination schedule "
            "shape mismatch."
        ),
    )

    require(
        final.illumination.dtype
        ==
        np.float64,
        (
            "Final illumination dtype mismatch."
        ),
    )

    require(
        np.all(
            np.isfinite(
                final.illumination
            )
        ),
        (
            "Final illumination contains "
            "non-finite values."
        ),
    )

    require(
        np.all(
            final.illumination
            >=
            0.0
        )
        and
        np.all(
            final.illumination
            <=
            1.0
        ),
        (
            "Final illumination outside [0,1]."
        ),
    )

    print(
        "composition output shape =",
        composition.raw_class_aware_illumination.shape,
    )

    print(
        "smoothed output shape    =",
        smoothed.shape,
    )

    print(
        "final output type        =",
        type(
            final
        ).__name__,
    )

    print(
        "final I shape            =",
        final.illumination.shape,
    )

    print(
        "final I finite/[0,1]     = PASS"
    )

    print(
        "synthetic performance claim = NO"
    )

    # ========================================================
    # E. Bind evaluator semantics
    # ========================================================

    print()
    print(
        "===== E. EVALUATOR I_FINAL BINDING ====="
    )

    binding = {
        "raw_predictive_class_aware_map":
            (
                "ClassAwareComposition."
                "raw_class_aware_illumination"
            ),

        "temporal_operator":
            "temporal_smooth_schedule",

        "rate_limit_operator":
            "apply_actuation_rate_limit",

        "final_result_type":
            "RateLimitedSchedule",

        "final_metric_intensity":
            "RateLimitedSchedule.illumination",

        "final_shape_semantics":
            "[horizon,theta,range]",

        "safety_override_diagnostic":
            (
                "RateLimitedSchedule."
                "safety_override_mask"
            ),

        "pre_temporal_raw_map_used_as_final_metric_input":
            False,

        "smoothed_pre_rate_limit_map_used_as_final_metric_input":
            False,
    }

    for key, value in (
        binding.items()
    ):

        print(
            f"{key} = {value}"
        )

    require(
        binding[
            "final_metric_intensity"
        ]
        ==
        "RateLimitedSchedule.illumination",
        (
            "Final metric intensity "
            "binding failed."
        ),
    )

    # ========================================================
    # F. Previous binder diagnosis
    # ========================================================

    print()
    print(
        "===== F. PREVIOUS BINDER DIAGNOSIS ====="
    )

    diagnosis = (
        route_extract[
            "diagnosis"
        ]
    )

    require(
        diagnosis[
            "scientific_bug"
        ]
        is False,
        (
            "Route extract indicates "
            "scientific bug."
        ),
    )

    require(
        diagnosis[
            "evidence_route_bug"
        ]
        is True,
        (
            "Expected evidence-route bug "
            "not recorded."
        ),
    )

    print(
        "scientific controller bug = NO"
    )

    print(
        "binder/evidence bug       = YES"
    )

    print(
        "scientific repair needed  = NO"
    )

    print(
        "binder repair             = THIS REPORT ONLY"
    )

    # ========================================================
    # G. Scientific boundary
    # ========================================================

    print()
    print(
        "===== G. SCIENTIFIC OUTCOME BOUNDARY ====="
    )

    print(
        "development outcome records read = NO"
    )

    print(
        "development metrics computed     = NO"
    )

    print(
        "numeric acceptance bounds        = NOT SELECTED"
    )

    print(
        "policy parameters modified       = NO"
    )

    print(
        "scientific controller modified   = NO"
    )

    print(
        "formal content opened            = NO"
    )

    print(
        "formal outcomes read             = NO"
    )

    print(
        "formal evaluation                = NO"
    )

    # ========================================================
    # H. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== H. FULL STAGE6 REGRESSION ====="
    )

    rc, test_count, output = (
        run_regression()
    )

    if rc != 0:

        print(
            "\n".join(
                output.splitlines()[
                    -100:
                ]
            )
        )

    require(
        rc == 0,
        (
            "Stage6 regression failed."
        ),
    )

    require(
        test_count
        ==
        EXPECTED_STAGE6_TESTS,
        (
            "Unexpected Stage6 test count: "
            f"{test_count}; expected "
            f"{EXPECTED_STAGE6_TESTS}."
        ),
    )

    print(
        "Stage6 regression =",
        f"{test_count} / {test_count} PASS",
    )

    # ========================================================
    # I. Immutability readback
    # ========================================================

    print()
    print(
        "===== I. IMMUTABILITY READBACK ====="
    )

    protected_after = {
        str(
            path
        ):
            sha256_file(
                path
            )
        for path in (
            RECON,
            PROTOCOL,
            METRICS,
            INTERFACE,
            CLASS_POLICY,
            MARGIN_BINDING,
        )
    }

    require(
        protected_before
        ==
        protected_after,
        (
            "Protected Stage6 scientific "
            "state changed."
        ),
    )

    print(
        "reconciled protocol    = UNCHANGED"
    )

    print(
        "metric runtime         = UNCHANGED"
    )

    print(
        "metric interface       = UNCHANGED"
    )

    print(
        "class-aware controller = UNCHANGED"
    )

    print(
        "margin binding         = UNCHANGED"
    )

    # ========================================================
    # J. Storage
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
    # K. Repaired binder report
    # ========================================================

    print()
    print(
        "===== K. REPAIRED BINDER FREEZE ====="
    )

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "2/2_exact_evaluator_binding_repair",

        "status":
            "PASS_FINAL_ACTUATION_OUTPUT_BOUND",

        "repair_type":
            "EVIDENCE_ROUTE_ONLY",

        "scientific_controller_changed":
            False,

        "previous_false_assumption":
            (
                "ClassAwareComposition.illumination"
            ),

        "correct_runtime_chain":
            binding,

        "synthetic_structural_execution": {
            "performed":
                True,

            "scientific_performance_claim":
                False,

            "raw_shape":
                list(
                    composition
                    .raw_class_aware_illumination
                    .shape
                ),

            "smoothed_shape":
                list(
                    smoothed.shape
                ),

            "final_type":
                type(
                    final
                ).__name__,

            "final_shape":
                list(
                    final
                    .illumination
                    .shape
                ),

            "finite":
                True,

            "bounded_0_1":
                True,
        },

        "scientific_boundary": {
            "development_outcomes_read":
                False,

            "development_metrics_computed":
                False,

            "numeric_bounds_selected":
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
            "tests":
                test_count,

            "status":
                "PASS",
        },

        "storage": {
            "free_GiB":
                free_gib,

            "hard_reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "next":
            (
                "Bind the evaluator-only future-oracle/full-box "
                "and road-ROI inputs, then execute the "
                "development-only ADB evaluation and freeze "
                "numeric non-inferiority bounds."
            ),
    }

    write_json(
        REPORT,
        result,
    )

    print(
        "repair report SHA256 =",
        sha256_file(
            REPORT
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 EXACT EVALUATOR BINDER REPAIR — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "previous binder assumption = REJECTED"
    )

    print(
        "raw class-aware I           = "
        "ClassAwareComposition.raw_class_aware_illumination"
    )

    print(
        "temporal operator           = "
        "temporal_smooth_schedule"
    )

    print(
        "rate-limit operator         = "
        "apply_actuation_rate_limit"
    )

    print(
        "final actuator result       = RateLimitedSchedule"
    )

    print(
        "final metric I_final        = "
        "RateLimitedSchedule.illumination"
    )

    print(
        "synthetic structural route  = PASS"
    )

    print(
        "scientific controller change= NO"
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
        "formal evaluation           = NO"
    )

    print(
        "Stage6 regression           =",
        f"{test_count} / {test_count} PASS",
    )

    print(
        "STATUS = PASS_FINAL_ACTUATION_OUTPUT_BOUND"
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
        "BLOCK 6.8 EXACT EVALUATOR BINDER REPAIR = BLOCKED"
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
        "scientific controller modified = NO"
    )

    print(
        "development metrics computed   = NO"
    )

    print(
        "numeric bounds selected        = NO"
    )

    print(
        "formal content opened          = NO"
    )

    print(
        "formal outcomes read           = NO"
    )

    print(
        "formal evaluation              = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
