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

BLOCK66_PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

BLOCK68_PROTOCOL = (
    S6
    / "configs/"
      "stage6_metric_freeze_protocol.json"
)

BLOCK68_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

BLOCK68_PART1 = (
    S6
    / "reports/"
      "block68_part1_metric_freeze_gate.json"
)

BLOCK68_PREFLIGHT = (
    S6
    / "reports/"
      "block68_part2_development_preflight.json"
)

BLOCK67 = (
    S6
    / "reports/block67_final_closure.json"
)

STAGE5 = (
    S5
    / "reports/stage5_final_closure.json"
)

REPORT = (
    S6
    / "reports/"
      "block68_part2_metric_authority_audit.json"
)

EXPECTED_PROTOCOL_SHA = (
    "c12b95c24ba64f48ac8dc3890fa55061"
    "3ea597875f6a5a759762b88629623ac0"
)

EXPECTED_RUNTIME_SHA = (
    "fdca66a1a9af74d24baaafe4e17d12ac"
    "db2db538e037503aba6ee04684eac362"
)

EXPECTED_BLOCK67_SHA = (
    "67d168ed7678164b8d85885e6e81f41f"
    "7206bd8f754e13ec5cc28c3a65aa20e7"
)

EXPECTED_STAGE6_TESTS = 196

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


def flatten(
    value,
    prefix="",
):

    if isinstance(
        value,
        dict,
    ):

        for key, child in (
            value.items()
        ):

            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
                )
            )

            yield from flatten(
                child,
                child_prefix,
            )

    elif isinstance(
        value,
        list,
    ):

        for index, child in enumerate(
            value
        ):

            yield from flatten(
                child,
                f"{prefix}[{index}]",
            )

    else:

        yield (
            prefix,
            value,
        )


def canonical_text(
    value,
) -> str:

    text = str(
        value
    ).strip().lower()

    text = text.replace(
        "–",
        "-",
    )

    text = text.replace(
        "−",
        "-",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


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
# Metric alias map
# ============================================================

PREREG_TO_BLOCK68 = {
    "mask_IoU":
        "mask_IoU_with_constructed_oracle_future_mask",

    "vehicle_shadow_zone_violation":
        "vehicle_shadow_zone_violation",

    "glare_risk_exposure":
        "glare_risk_exposure",

    "over_masking_area":
        "over_masking_area",

    "road_illumination_retention":
        "road_illumination_retention",

    "pedestrian_visibility_proxy":
        "pedestrian_visibility_proxy",

    "cyclist_visibility_proxy":
        "cyclist_visibility_proxy",

    "false_dimming":
        "false_dimming",

    "temporal_smoothness":
        "temporal_smoothness",

    "flicker_change_rate":
        "flicker_change_rate",

    "energy_consumption":
        "energy_consumption",

    "actuation_latency":
        "actuation_latency",
}


def get_metric_object(
    prereg,
    metric_name,
):

    formulas = prereg.get(
        "ADB_metric_formulas",
        {}
    )

    value = formulas.get(
        metric_name
    )

    if isinstance(
        value,
        dict,
    ):
        return value

    return {}


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
        "PRE-FORMAL METRIC AUTHORITY RECONCILIATION AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        BLOCK66_PREREG,
        BLOCK68_PROTOCOL,
        BLOCK68_RUNTIME,
        BLOCK68_PART1,
        BLOCK68_PREFLIGHT,
        BLOCK67,
        STAGE5,
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
            "Missing required frozen evidence: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Continuity
    # ========================================================

    print()
    print(
        "===== A. FROZEN CONTINUITY ====="
    )

    require(
        sha256_file(
            BLOCK68_PROTOCOL
        )
        ==
        EXPECTED_PROTOCOL_SHA,
        (
            "Block6.8 metric protocol SHA changed."
        ),
    )

    require(
        sha256_file(
            BLOCK68_RUNTIME
        )
        ==
        EXPECTED_RUNTIME_SHA,
        (
            "Block6.8 metric runtime SHA changed."
        ),
    )

    require(
        sha256_file(
            BLOCK67
        )
        ==
        EXPECTED_BLOCK67_SHA,
        (
            "Block6.7 closure SHA changed."
        ),
    )

    part1 = load_json(
        BLOCK68_PART1
    )

    preflight = load_json(
        BLOCK68_PREFLIGHT
    )

    require(
        part1.get(
            "status"
        )
        ==
        "PASS_METRIC_SEMANTICS_FROZEN",
        (
            "Block6.8 Part1 status changed."
        ),
    )

    require(
        preflight.get(
            "status"
        )
        ==
        "PASS_DEVELOPMENT_API_DISCOVERED",
        (
            "Block6.8 development preflight "
            "is not PASS."
        ),
    )

    print(
        "Block6.7                = EXACT PASS"
    )

    print(
        "Block6.8 Part1          = EXACT PASS"
    )

    print(
        "Block6.8 preflight      = PASS"
    )

    print(
        "Block6.8 protocol/runtime = EXACT PASS"
    )

    # ========================================================
    # B. Scientific outcome boundary
    # ========================================================

    print()
    print(
        "===== B. SCIENTIFIC OUTCOME BOUNDARY ====="
    )

    scientific = preflight.get(
        "scientific_outcomes",
        {}
    )

    require(
        scientific.get(
            "development_outcomes_read"
        )
        is False,
        (
            "Development outcomes were already read."
        ),
    )

    require(
        scientific.get(
            "formal_outcomes_read"
        )
        is False,
        (
            "Formal outcomes were already read."
        ),
    )

    require(
        scientific.get(
            "formal_manifest_content_read"
        )
        is False,
        (
            "Formal manifest content was read."
        ),
    )

    print(
        "development outcomes read = NO"
    )

    print(
        "formal outcomes read      = NO"
    )

    print(
        "formal manifest read      = NO"
    )

    print(
        "metric reconciliation     = PRE-OUTCOME"
    )

    # ========================================================
    # C. Exact Block6.6 preregistration dump
    # ========================================================

    print()
    print(
        "===== C. BLOCK6.6 PREREGISTERED METRIC AUTHORITY ====="
    )

    prereg = load_json(
        BLOCK66_PREREG
    )

    prereg_formula_root = (
        prereg.get(
            "ADB_metric_formulas"
        )
    )

    require(
        isinstance(
            prereg_formula_root,
            dict,
        ),
        (
            "Block6.6 preregistration has no "
            "ADB_metric_formulas object."
        ),
    )

    print(
        "Block6.6 prereg SHA256 =",
        sha256_file(
            BLOCK66_PREREG
        ),
    )

    for key, value in flatten(
        prereg_formula_root
    ):

        text = str(
            value
        )

        if len(
            text
        ) > 220:
            text = (
                text[:217]
                +
                "..."
            )

        print(
            f"{key} = {text}"
        )

    # ========================================================
    # D. Exact Block6.8 protocol dump
    # ========================================================

    print()
    print(
        "===== D. BLOCK6.8 PART1 METRIC PROTOCOL ====="
    )

    protocol = load_json(
        BLOCK68_PROTOCOL
    )

    metrics68 = protocol.get(
        "metrics",
        {}
    )

    require(
        isinstance(
            metrics68,
            dict,
        )
        and
        len(
            metrics68
        )
        ==
        12,
        (
            "Block6.8 protocol does not "
            "contain exactly 12 metrics."
        ),
    )

    for metric_name in sorted(
        metrics68
    ):

        metric = metrics68[
            metric_name
        ]

        print()
        print(
            "METRIC =",
            metric_name,
        )

        for key, value in flatten(
            metric
        ):

            print(
                f"  {key} = {value}"
            )

    # ========================================================
    # E. Formula-level comparison
    # ========================================================

    print()
    print(
        "===== E. PREREGISTRATION CONSISTENCY MATRIX ====="
    )

    matrix = {}

    mismatch_count = 0
    missing_count = 0
    direction_conflicts = 0

    for prereg_name, current_name in (
        PREREG_TO_BLOCK68.items()
    ):

        old = get_metric_object(
            prereg,
            prereg_name,
        )

        new = metrics68.get(
            current_name,
            {}
        )

        old_formula = old.get(
            "formula"
        )

        new_formula = new.get(
            "formula"
        )

        old_direction = old.get(
            "direction"
        )

        new_direction = new.get(
            "direction"
        )

        formula_status = (
            "NOT_COMPARABLE"
        )

        if (
            old_formula is not None
            and
            new_formula is not None
        ):

            formula_status = (
                "EXACT_TEXT_EQUIVALENT"
                if canonical_text(
                    old_formula
                )
                ==
                canonical_text(
                    new_formula
                )
                else
                "DIFFERENT"
            )

        elif (
            old_formula is None
            or
            new_formula is None
        ):

            formula_status = (
                "MISSING_FORMULA_FIELD"
            )

        direction_status = (
            "NOT_COMPARABLE"
        )

        if (
            old_direction is not None
            and
            new_direction is not None
        ):

            direction_status = (
                "SAME"
                if canonical_text(
                    old_direction
                )
                ==
                canonical_text(
                    new_direction
                )
                else
                "DIFFERENT"
            )

        elif (
            old_direction is None
            or
            new_direction is None
        ):

            direction_status = (
                "MISSING_DIRECTION_FIELD"
            )

        if (
            formula_status
            ==
            "DIFFERENT"
        ):
            mismatch_count += 1

        if (
            formula_status
            ==
            "MISSING_FORMULA_FIELD"
        ):
            missing_count += 1

        if (
            direction_status
            ==
            "DIFFERENT"
        ):
            direction_conflicts += 1

        matrix[
            current_name
        ] = {
            "Block66_name":
                prereg_name,

            "Block66_formula":
                old_formula,

            "Block68_formula":
                new_formula,

            "formula_status":
                formula_status,

            "Block66_direction":
                old_direction,

            "Block68_direction":
                new_direction,

            "direction_status":
                direction_status,
        }

        print()
        print(
            "METRIC =",
            current_name,
        )

        print(
            "  Block6.6 formula =",
            old_formula,
        )

        print(
            "  Block6.8 formula =",
            new_formula,
        )

        print(
            "  formula status   =",
            formula_status,
        )

        print(
            "  Block6.6 direction =",
            old_direction,
        )

        print(
            "  Block6.8 direction =",
            new_direction,
        )

        print(
            "  direction status =",
            direction_status,
        )

    print()
    print(
        "formula text mismatches =",
        mismatch_count,
    )

    print(
        "formula missing fields  =",
        missing_count,
    )

    print(
        "direction conflicts     =",
        direction_conflicts,
    )

    # ========================================================
    # F. Known semantic conflicts
    # ========================================================

    print()
    print(
        "===== F. KNOWN SEMANTIC CONFLICT CHECK ====="
    )

    old_temporal = get_metric_object(
        prereg,
        "temporal_smoothness",
    )

    new_temporal = metrics68[
        "temporal_smoothness"
    ]

    old_overmask = get_metric_object(
        prereg,
        "over_masking_area",
    )

    new_overmask = metrics68[
        "over_masking_area"
    ]

    old_false = get_metric_object(
        prereg,
        "false_dimming",
    )

    new_false = metrics68[
        "false_dimming"
    ]

    old_flicker = get_metric_object(
        prereg,
        "flicker_change_rate",
    )

    new_flicker = metrics68[
        "flicker_change_rate"
    ]

    semantic_conflicts = {
        "temporal_smoothness":
            (
                canonical_text(
                    old_temporal.get(
                        "formula",
                        ""
                    )
                )
                !=
                canonical_text(
                    new_temporal.get(
                        "formula",
                        ""
                    )
                )
                or
                canonical_text(
                    old_temporal.get(
                        "direction",
                        ""
                    )
                )
                !=
                canonical_text(
                    new_temporal.get(
                        "direction",
                        ""
                    )
                )
            ),

        "over_masking_area":
            (
                canonical_text(
                    old_overmask.get(
                        "formula",
                        ""
                    )
                )
                !=
                canonical_text(
                    new_overmask.get(
                        "formula",
                        ""
                    )
                )
            ),

        "false_dimming":
            (
                canonical_text(
                    old_false.get(
                        "formula",
                        ""
                    )
                )
                !=
                canonical_text(
                    new_false.get(
                        "formula",
                        ""
                    )
                )
            ),

        "flicker_change_rate":
            (
                canonical_text(
                    old_flicker.get(
                        "formula",
                        ""
                    )
                )
                !=
                canonical_text(
                    new_flicker.get(
                        "formula",
                        ""
                    )
                )
            ),
    }

    for key, conflict in (
        semantic_conflicts.items()
    ):

        print(
            f"{key:28s}=",
            (
                "CONFLICT"
                if conflict
                else "CONSISTENT"
            ),
        )

    require(
        any(
            semantic_conflicts.values()
        ),
        (
            "Expected preregistration conflict "
            "was not reproduced; inspect manually."
        ),
    )

    # ========================================================
    # G. Authority rule
    # ========================================================

    print()
    print(
        "===== G. PRE-FORMAL AUTHORITY RULE ====="
    )

    authority = {
        "PDF_role":
            (
                "defines required metric set and "
                "Stage6 completion objective; "
                "does not uniquely define every "
                "numerical metric formula"
            ),

        "earliest_internal_preregistration":
            str(
                BLOCK66_PREREG
            ),

        "earliest_preregistration_precedes_Block68":
            True,

        "development_outcomes_seen_before_reconciliation":
            False,

        "formal_outcomes_seen_before_reconciliation":
            False,

        "binding_rule":
            (
                "earliest explicit pre-outcome "
                "Block6.6 metric preregistration "
                "is authoritative unless an explicit "
                "scientific amendment is documented "
                "before any development/formal "
                "outcome inspection"
            ),

        "silent_Block68_override_allowed":
            False,

        "recommended_resolution":
            (
                "repair Block6.8 protocol/runtime "
                "to preserve Block6.6 preregistered "
                "semantics, then rerun regression "
                "and only afterwards permit "
                "development-bound evaluation"
            ),
    }

    for key, value in (
        authority.items()
    ):

        print(
            f"{key} = {value}"
        )

    # ========================================================
    # H. Full regression
    # ========================================================

    print()
    print(
        "===== H. FULL STAGE6 REGRESSION ====="
    )

    rc, tests, output = (
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
        "Stage6 regression failed.",
    )

    require(
        tests
        ==
        EXPECTED_STAGE6_TESTS,
        (
            "Unexpected Stage6 regression "
            f"count {tests}; expected "
            f"{EXPECTED_STAGE6_TESTS}."
        ),
    )

    print(
        "Stage6 regression =",
        f"{tests} / {tests} PASS",
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
    # J. Audit report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "2/2_metric_authority_audit",

        "status":
            "BLOCKED_PREOUTCOME_METRIC_RECONCILIATION_REQUIRED",

        "scientific_boundary": {
            "development_outcomes_read":
                False,

            "formal_outcomes_read":
                False,

            "policy_tuning":
                False,

            "formal_evaluation":
                False,
        },

        "authority": authority,

        "comparison_matrix":
            matrix,

        "semantic_conflicts":
            semantic_conflicts,

        "counts": {
            "formula_text_mismatches":
                mismatch_count,

            "formula_missing_fields":
                missing_count,

            "direction_conflicts":
                direction_conflicts,
        },

        "frozen_hashes": {
            "Block66_preregistration":
                sha256_file(
                    BLOCK66_PREREG
                ),

            "Block68_protocol":
                sha256_file(
                    BLOCK68_PROTOCOL
                ),

            "Block68_runtime":
                sha256_file(
                    BLOCK68_RUNTIME
                ),

            "Block67_closure":
                sha256_file(
                    BLOCK67
                ),
        },

        "regression": {
            "tests":
                tests,

            "status":
                "PASS",
        },

        "next":
            (
                "Controlled pre-outcome repair: "
                "align Block6.8 numerical metric "
                "protocol/runtime with the earlier "
                "Block6.6 preregistration before "
                "development-bound evaluation."
            ),
    }

    write_json(
        REPORT,
        result,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART 2/2 — METRIC AUTHORITY AUDIT FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block6.6 preregistration = FOUND"
    )

    print(
        "Block6.8 metric freeze   = FOUND"
    )

    print(
        "development outcomes     = NOT READ"
    )

    print(
        "formal outcomes          = NOT READ"
    )

    print(
        "semantic conflicts       = FOUND"
    )

    print(
        "silent override allowed  = NO"
    )

    print(
        "authority                = "
        "EARLIEST PRE-OUTCOME PREREGISTRATION"
    )

    print(
        "development evaluation   = STILL BLOCKED"
    )

    print(
        "Stage6 regression        =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "BLOCKED_PREOUTCOME_METRIC_RECONCILIATION_REQUIRED"
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
        "BLOCK 6.8 METRIC AUTHORITY AUDIT = BLOCKED"
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

    print(
        "development outcomes read = NO"
    )

    print(
        "formal outcomes read      = NO"
    )

    print(
        "formal evaluation         = NO"
    )

    print(
        "policy tuning             = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
