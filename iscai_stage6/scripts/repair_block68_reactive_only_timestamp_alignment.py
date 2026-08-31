from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

import numpy as np

from iscai_stage4.data.real_pipeline import (
    read_training_scenario,
)


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"
S0 = ROOT / "iscai_stage0"

RUNNER = (
    S6
    / "scripts/"
      "run_block68_reactive_only_ni_delta_freeze.py"
)

COHORT = (
    S6
    / "artifacts/block66/"
      "block66_class_aware_development_cohort_120.jsonl"
)

PREOUTCOME_CONTRACT = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_scoring_contract.json"
)

PROVENANCE_BINDING = (
    S6
    / "configs/"
      "stage6_future_gt_evaluator_provenance_binding.json"
)

STAGE0_TIMING_TEST = (
    S0
    / "tests/"
      "test_womd_schema_and_causality.py"
)

RAW_VALUES = (
    S6
    / "artifacts/block68/"
      "block68_reactive_only_ni_values.jsonl"
)

DELTA_FREEZE = (
    S6
    / "configs/"
      "stage6_exact_noninferiority_deltas.json"
)

ACCEPTANCE_POLICY = (
    S6
    / "configs/"
      "stage6_preformal_primary_acceptance_policy.json"
)

FINAL_REPORT = (
    S6
    / "reports/"
      "block68_reactive_only_ni_delta_freeze.json"
)

AMENDMENT = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_timestamp_alignment_amendment.json"
)

PATCH_REPORT = (
    S6
    / "reports/"
      "block68_reactive_future_truth_timestamp_alignment_repair.json"
)

BACKUP = (
    S6
    / "artifacts/block68/"
      "timestamp_alignment_repair/"
      "run_block68_reactive_only_ni_delta_freeze.pre_timestamp_patch.py"
)


EXPECTED_RUNNER_PRE_SHA = (
    "35fedb5ac974140626ec64ab95ff1423"
    "28366a77bffeb2815bce16e4953249dd"
)

EXPECTED_PREOUTCOME_SHA = (
    "8364c7cdfa4b66d820823c804d45905a"
    "207f77f66b53f06a6383e7722b8a9970"
)

EXPECTED_PROVENANCE_BINDING_SHA = (
    "24119c2a029eaf20dc33c5baa66693c3"
    "1b6a41f61966304ce02f9c5793ae50b7"
)

EXPECTED_TESTS = 224
EXPECTED_SCENARIOS = 120

NOMINAL_STEP_S = 0.1
PER_STEP_TOLERANCE_S = 1.0e-3

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

FUTURE_INDEX_OFFSETS = (
    1,
    3,
    5,
    10,
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
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


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line_number, line in enumerate(
            stream,
            start=1,
        ):

            if not line.strip():
                continue

            value = json.loads(line)

            require(
                isinstance(value, dict),
                (
                    "JSONL row is not object: "
                    f"{path}:{line_number}"
                ),
            )

            rows.append(value)

    return rows


def canonical_json_bytes(value):
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


def atomic_write(path: Path, payload: bytes):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)

    os.replace(
        temporary,
        path,
    )


def write_once(path: Path, value):
    payload = canonical_json_bytes(value)

    if path.exists():

        require(
            path.read_bytes()
            ==
            payload,
            (
                "Existing write-once artifact "
                f"differs: {path}"
            ),
        )

        return

    atomic_write(
        path,
        payload,
    )


def run_command(command):
    process = subprocess.run(
        command,
        cwd=str(S6),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return (
        process.returncode,
        process.stdout,
    )


def parse_test_count(output):
    for line in output.splitlines():

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            return int(match.group(1))

    return None


def regression():
    rc, output = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ]
    )

    count = parse_test_count(output)

    require(
        rc == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                output.splitlines()[-100:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"got {count}."
        ),
    )

    return count


# ============================================================
# Stage-0 semantic authority
# ============================================================

def prove_stage0_timing_semantics():
    require(
        STAGE0_TIMING_TEST.is_file(),
        (
            "Stage0 timing test missing: "
            f"{STAGE0_TIMING_TEST}"
        ),
    )

    source = STAGE0_TIMING_TEST.read_text(
        encoding="utf-8"
    )

    compact = re.sub(
        r"\s+",
        " ",
        source,
    )

    require(
        (
            "abs(dt - 0.1) < 1e-3 "
            "for dt in deltas"
        )
        in
        compact,
        (
            "Could not prove the frozen Stage0 "
            "per-step 100-ms timing tolerance."
        ),
    )

    require(
        (
            "t1 > t0 "
            "for t0, t1 in zip("
            "timestamps[:-1], timestamps[1:])"
        )
        in
        compact,
        (
            "Could not prove frozen Stage0 "
            "timestamp monotonicity check."
        ),
    )

    return {
        "path":
            str(STAGE0_TIMING_TEST),

        "sha256":
            file_sha(STAGE0_TIMING_TEST),

        "nominal_adjacent_step_s":
            NOMINAL_STEP_S,

        "per_adjacent_step_absolute_tolerance_s":
            PER_STEP_TOLERANCE_S,

        "strict_inequality":
            True,

        "semantics":
            (
                "for every adjacent WOMD timestamp pair: "
                "timestamps strictly increase and "
                "abs(dt-0.1) < 1e-3"
            ),
    }


# ============================================================
# Development-cohort timestamp-only audit
# ============================================================

def audit_timestamp_alignment():
    """
    Timestamp-only evaluator audit.

    No actor future-state geometry, no illumination values,
    no metric values and no predictive/P_occ content.
    """

    rows = read_jsonl(COHORT)

    require(
        len(rows) == EXPECTED_SCENARIOS,
        (
            f"Expected {EXPECTED_SCENARIOS} "
            f"development scenarios; got {len(rows)}."
        ),
    )

    maximum_step_error = 0.0

    maximum_cumulative_error = {
        str(horizon):
            0.0
        for horizon in HORIZONS_S
    }

    maximum_cumulative_example = {
        str(horizon):
            None
        for horizon in HORIZONS_S
    }

    checked_unique_steps = 0

    for index, row in enumerate(rows):

        scenario = read_training_scenario(
            row
        )

        sid = str(
            scenario.scenario_id
        )

        require(
            sid
            ==
            str(
                row[
                    "scenario_id"
                ]
            ),
            (
                "Scenario ID mismatch "
                f"at cohort row {index}."
            ),
        )

        anchor = int(
            scenario.current_time_index
        )

        if "current_time_index" in row:

            require(
                anchor
                ==
                int(
                    row[
                        "current_time_index"
                    ]
                ),
                (
                    "Anchor mismatch for "
                    f"{sid}."
                ),
            )

        timestamps = np.asarray(
            scenario.timestamps_seconds,
            dtype=np.float64,
        )

        require(
            timestamps.ndim == 1,
            (
                "timestamps_seconds is not 1-D: "
                f"{sid}"
            ),
        )

        require(
            np.all(
                np.isfinite(
                    timestamps
                )
            ),
            (
                "Non-finite timestamp: "
                f"{sid}"
            ),
        )

        last_index = (
            anchor
            +
            max(
                FUTURE_INDEX_OFFSETS
            )
        )

        require(
            last_index
            <
            len(timestamps),
            (
                "Insufficient future timestamp "
                f"coverage: {sid}"
            ),
        )

        # Audit the 10 unique adjacent samples once.
        window = timestamps[
            anchor:
            last_index + 1
        ]

        step_dts = np.diff(
            window
        )

        require(
            len(step_dts)
            ==
            max(
                FUTURE_INDEX_OFFSETS
            ),
            (
                "Unexpected timestamp-window "
                f"length for {sid}."
            ),
        )

        require(
            np.all(
                step_dts > 0.0
            ),
            (
                "Non-monotonic future timestamps: "
                f"{sid}"
            ),
        )

        step_errors = np.abs(
            step_dts
            -
            NOMINAL_STEP_S
        )

        require(
            np.all(
                step_errors
                <
                PER_STEP_TOLERANCE_S
            ),
            (
                "Stage0 per-step timestamp gate "
                f"fails for scenario={sid}; "
                f"max_error="
                f"{float(np.max(step_errors))}"
            ),
        )

        maximum_step_error = max(
            maximum_step_error,
            float(
                np.max(
                    step_errors
                )
            ),
        )

        checked_unique_steps += len(
            step_dts
        )

        anchor_time = float(
            timestamps[
                anchor
            ]
        )

        # Cumulative deviation is diagnostic only.
        for horizon, offset in zip(
            HORIZONS_S,
            FUTURE_INDEX_OFFSETS,
            strict=True,
        ):

            actual = (
                float(
                    timestamps[
                        anchor + offset
                    ]
                )
                -
                anchor_time
            )

            error = abs(
                actual
                -
                float(
                    horizon
                )
            )

            key = str(
                horizon
            )

            if (
                error
                >
                maximum_cumulative_error[
                    key
                ]
            ):

                maximum_cumulative_error[
                    key
                ] = error

                maximum_cumulative_example[
                    key
                ] = {
                    "scenario_id":
                        sid,

                    "nominal_horizon_s":
                        float(
                            horizon
                        ),

                    "actual_elapsed_s":
                        actual,

                    "absolute_cumulative_error_s":
                        error,
                }

    return {
        "scenario_count":
            len(rows),

        "unique_adjacent_steps_checked":
            checked_unique_steps,

        "per_step_gate":
            "PASS",

        "maximum_adjacent_step_error_s":
            maximum_step_error,

        "cumulative_error_role":
            "DIAGNOSTIC_ONLY",

        "maximum_cumulative_error_s":
            maximum_cumulative_error,

        "maximum_cumulative_examples":
            maximum_cumulative_example,

        "performance_metric_values_read":
            False,

        "actor_future_state_values_read":
            False,

        "P_occ_opened":
            False,

        "predictive_performance_read":
            False,
    }


# ============================================================
# Exact AST patch
# ============================================================

def call_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr

    return None


def contains_string(node, text):
    return any(
        isinstance(child, ast.Constant)
        and
        isinstance(child.value, str)
        and
        text in child.value
        for child in ast.walk(node)
    )


def locate_cumulative_timestamp_require(
    source,
):
    tree = ast.parse(
        source,
        filename=str(RUNNER),
    )

    matches = []

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.Expr,
        ):
            continue

        call = node.value

        if not isinstance(
            call,
            ast.Call,
        ):
            continue

        if call_name(
            call.func
        ) != "require":
            continue

        if contains_string(
            node,
            "WOMD horizon timestamp mismatch",
        ):

            matches.append(node)

    require(
        len(matches) == 1,
        (
            "Expected exactly one cumulative "
            "WOMD horizon timestamp require(); "
            f"found {len(matches)}."
        ),
    )

    require(
        matches[0].end_lineno
        is not None,
        (
            "AST end_lineno unavailable."
        ),
    )

    return matches[0]


def patch_timestamp_gate(source):
    node = locate_cumulative_timestamp_require(
        source
    )

    lines = source.splitlines(
        keepends=True
    )

    start = int(
        node.lineno
    )

    end = int(
        node.end_lineno
    )

    original_statement = "".join(
        lines[
            start - 1:
            end
        ]
    )

    indentation = re.match(
        r"\s*",
        lines[
            start - 1
        ],
    ).group(0)

    body = [
        "# STAGE0_PER_STEP_TIMESTAMP_ALIGNMENT",
        "timestamp_window = np.asarray(",
        "    scenario.timestamps_seconds[",
        "        anchor_index:",
        "        future_index + 1",
        "    ],",
        "    dtype=np.float64,",
        ")",
        "",
        "require(",
        "    len(timestamp_window)",
        "    ==",
        "    offset + 1,",
        "    (",
        '        "Unexpected WOMD timestamp-window length "',
        '        f"sid={sid} h={horizon_s}"',
        "    ),",
        ")",
        "",
        "step_dts = np.diff(",
        "    timestamp_window",
        ")",
        "",
        "require(",
        "    np.all(",
        "        np.isfinite(",
        "            step_dts",
        "        )",
        "    ),",
        "    (",
        '        "Non-finite WOMD timestamp delta "',
        '        f"sid={sid} h={horizon_s}"',
        "    ),",
        ")",
        "",
        "require(",
        "    np.all(",
        "        step_dts > 0.0",
        "    ),",
        "    (",
        '        "Non-monotonic WOMD timestamps "',
        '        f"sid={sid} h={horizon_s}"',
        "    ),",
        ")",
        "",
        "# Exact upstream Stage0 semantic:",
        "# each adjacent timestep, not cumulative horizon.",
        "step_errors = np.abs(",
        "    step_dts",
        "    -",
        "    0.1",
        ")",
        "",
        "require(",
        "    np.all(",
        "        step_errors",
        "        <",
        "        TIMESTAMP_TOLERANCE_S",
        "    ),",
        "    (",
        '        "WOMD per-step timestamp mismatch "',
        '        f"sid={sid} h={horizon_s} "',
        '        f"max_step_error={float(np.max(step_errors))} "',
        '        f"actual_cumulative_dt={actual_dt}"',
        "    ),",
        ")",
    ]

    replacement = "".join(
        indentation
        +
        line
        +
        "\n"
        for line in body
    )

    patched = "".join(
        lines[
            :start - 1
        ]
        +
        [
            replacement
        ]
        +
        lines[
            end:
        ]
    )

    compile(
        patched,
        str(RUNNER),
        "exec",
    )

    require(
        (
            "STAGE0_PER_STEP_TIMESTAMP_ALIGNMENT"
            in
            patched
        ),
        (
            "Patched semantic marker missing."
        ),
    )

    require(
        (
            "WOMD horizon timestamp mismatch"
            not in
            patched
        ),
        (
            "Old cumulative timestamp gate "
            "still exists."
        ),
    )

    require(
        (
            "WOMD per-step timestamp mismatch"
            in
            patched
        ),
        (
            "New per-step timestamp gate missing."
        ),
    )

    return (
        patched,
        {
            "start_line":
                start,

            "end_line":
                end,

            "old_statement_sha256":
                sha256(
                    original_statement.encode(
                        "utf-8"
                    )
                ).hexdigest(),
        },
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART2C"
    )
    print(
        "WOMD TIMESTAMP ALIGNMENT SEMANTIC REPAIR"
    )
    print(
        "UPSTREAM STAGE0 PER-STEP RULE / NO METRIC TUNING"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Idempotent repair readback
    # --------------------------------------------------------

    if PATCH_REPORT.is_file():

        report = read_json(
            PATCH_REPORT
        )

        if (
            report.get(
                "status"
            )
            ==
            "PASS_REACTIVE_TIMESTAMP_ALIGNMENT_REPAIRED"
        ):

            require(
                file_sha(
                    AMENDMENT
                )
                ==
                report[
                    "amendment"
                ][
                    "sha256"
                ],
                (
                    "Timestamp amendment "
                    "changed after freeze."
                ),
            )

            require(
                file_sha(
                    RUNNER
                )
                ==
                report[
                    "runner"
                ][
                    "post_patch_sha256"
                ],
                (
                    "Timestamp-patched runner "
                    "changed after freeze."
                ),
            )

            tests = regression()

            print(
                "existing timestamp repair = EXACT PASS"
            )

            print(
                "Stage6 regression         =",
                f"{tests} / {tests} PASS",
            )

            print(
                "STATUS = "
                "PASS_REACTIVE_TIMESTAMP_ALIGNMENT_REPAIRED"
            )

            print(
                "terminal remains open = YES"
            )

            return

    # --------------------------------------------------------
    # A. Exact failure-state boundary
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT FAILURE-STATE BOUNDARY ====="
    )

    require(
        file_sha(
            RUNNER
        )
        ==
        EXPECTED_RUNNER_PRE_SHA,
        (
            "Reactive-only runner differs "
            "from exact provenance-repaired state."
        ),
    )

    require(
        file_sha(
            PREOUTCOME_CONTRACT
        )
        ==
        EXPECTED_PREOUTCOME_SHA,
        (
            "Preoutcome scoring contract changed."
        ),
    )

    require(
        file_sha(
            PROVENANCE_BINDING
        )
        ==
        EXPECTED_PROVENANCE_BINDING_SHA,
        (
            "Future-GT provenance binding changed."
        ),
    )

    for path, label in (
        (
            RAW_VALUES,
            "reactive raw metrics",
        ),
        (
            DELTA_FREEZE,
            "NI delta freeze",
        ),
        (
            ACCEPTANCE_POLICY,
            "acceptance policy",
        ),
        (
            FINAL_REPORT,
            "final reactive-only report",
        ),
    ):

        require(
            not path.exists(),
            (
                f"{label} unexpectedly exists: "
                f"{path}"
            ),
        )

    print(
        "runner provenance repair = EXACT PASS"
    )

    print(
        "preoutcome contract       = EXACT PASS"
    )

    print(
        "metric artifacts          = ABSENT"
    )

    print(
        "NI deltas                 = NOT COMPUTED"
    )

    print(
        "predictive performance    = NOT COMPUTED"
    )

    print(
        "P_occ content             = NOT OPENED"
    )

    print(
        "future GT                 = OPENED, EVALUATOR ONLY"
    )

    # --------------------------------------------------------
    # B. Prove upstream Stage0 semantics
    # --------------------------------------------------------

    print()
    print(
        "===== B. UPSTREAM STAGE0 TIMESTAMP AUTHORITY ====="
    )

    stage0_authority = (
        prove_stage0_timing_semantics()
    )

    print(
        "nominal adjacent dt      = 0.1 s"
    )

    print(
        "per-step tolerance       = < 0.001 s"
    )

    print(
        "timestamp monotonicity   = REQUIRED"
    )

    print(
        "cumulative 1-ms gate     = NOT UPSTREAM SEMANTIC"
    )

    print(
        "Stage0 timing test SHA   =",
        stage0_authority[
            "sha256"
        ],
    )

    # --------------------------------------------------------
    # C. Freeze semantic amendment BEFORE cohort audit
    # --------------------------------------------------------

    print()
    print(
        "===== C. TIMESTAMP SEMANTIC AMENDMENT FREEZE ====="
    )

    amendment = {
        "stage":
            6,

        "block":
            "6.8_Part2C_timestamp_alignment_repair",

        "status":
            "FROZEN_PREMETRIC_TIMESTAMP_ALIGNMENT_AMENDMENT",

        "reason":
            (
                "Part2C incorrectly interpreted the "
                "existing 1e-3 Stage0 adjacent-step "
                "tolerance as an absolute cumulative-"
                "horizon tolerance."
            ),

        "original_preoutcome_contract": {
            "path":
                str(
                    PREOUTCOME_CONTRACT
                ),

            "sha256":
                EXPECTED_PREOUTCOME_SHA,

            "modified":
                False,
        },

        "authority":
            stage0_authority,

        "authoritative_runtime_rule": {
            "nominal_adjacent_step_s":
                NOMINAL_STEP_S,

            "per_adjacent_step_tolerance_s":
                PER_STEP_TOLERANCE_S,

            "comparison":
                "abs(dt_adjacent - 0.1) < 1e-3",

            "strict_less_than":
                True,

            "monotonic_required":
                True,

            "future_index_offsets":
                list(
                    FUTURE_INDEX_OFFSETS
                ),

            "nominal_horizons_s":
                list(
                    HORIZONS_S
                ),

            "interpolation":
                False,

            "nearest_timestamp_search":
                False,

            "cumulative_elapsed_time":
                "DIAGNOSTIC_ONLY",

            "cumulative_absolute_1ms_gate":
                False,
        },

        "scientific_change": {
            "future_index_selection":
                False,

            "future_state_selection":
                False,

            "trajectory_geometry":
                False,

            "rasterization":
                False,

            "metric_formula":
                False,

            "NI_rule":
                False,

            "controller":
                False,

            "policy":
                False,
        },

        "outcome_boundary": {
            "future_GT_already_opened":
                True,

            "future_GT_role":
                "evaluator_only",

            "metric_values_computed_before_amendment":
                False,

            "reactive_metric_artifact_written":
                False,

            "predictive_performance_seen":
                False,

            "NI_deltas_seen":
                False,

            "formal_evaluation":
                False,
        },
    }

    write_once(
        AMENDMENT,
        amendment,
    )

    amendment_sha = file_sha(
        AMENDMENT
    )

    print(
        "amendment SHA256 =",
        amendment_sha,
    )

    # --------------------------------------------------------
    # D. Timestamp-only audit of all 120
    # --------------------------------------------------------

    print()
    print(
        "===== D. 120-SCENARIO TIMESTAMP-ONLY AUDIT ====="
    )

    timestamp_audit = (
        audit_timestamp_alignment()
    )

    print(
        "development scenarios      =",
        timestamp_audit[
            "scenario_count"
        ],
        "/ 120 PASS",
    )

    print(
        "unique adjacent steps      =",
        timestamp_audit[
            "unique_adjacent_steps_checked"
        ],
    )

    print(
        "per-step Stage0 gate       = PASS"
    )

    print(
        "max adjacent-step error s  =",
        timestamp_audit[
            "maximum_adjacent_step_error_s"
        ],
    )

    print(
        "cumulative errors           = DIAGNOSTIC ONLY"
    )

    for horizon in HORIZONS_S:

        key = str(horizon)

        print(
            f"  horizon {horizon:0.1f}s "
            "max cumulative error =",
            timestamp_audit[
                "maximum_cumulative_error_s"
            ][
                key
            ],
        )

    print(
        "actor future states read   = NO"
    )

    print(
        "performance values read    = NO"
    )

    # --------------------------------------------------------
    # E. Minimal exact runner patch
    # --------------------------------------------------------

    print()
    print(
        "===== E. PATCH CUMULATIVE GATE ONLY ====="
    )

    original_bytes = RUNNER.read_bytes()

    original_source = (
        original_bytes.decode(
            "utf-8"
        )
    )

    (
        patched_source,
        patch_detail,
    ) = patch_timestamp_gate(
        original_source
    )

    BACKUP.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BACKUP.exists():

        require(
            BACKUP.read_bytes()
            ==
            original_bytes,
            (
                "Existing timestamp-repair "
                "backup differs."
            ),
        )

    else:

        BACKUP.write_bytes(
            original_bytes
        )

    require(
        file_sha(
            BACKUP
        )
        ==
        EXPECTED_RUNNER_PRE_SHA,
        (
            "Timestamp repair backup "
            "SHA mismatch."
        ),
    )

    print(
        "runner pre-patch SHA =",
        EXPECTED_RUNNER_PRE_SHA,
    )

    print(
        "patched old lines    =",
        patch_detail[
            "start_line"
        ],
        "..",
        patch_detail[
            "end_line"
        ],
    )

    print(
        "patch scope           = TIMESTAMP REQUIRE ONLY"
    )

    # --------------------------------------------------------
    # F. Transactional write / regression
    # --------------------------------------------------------

    print()
    print(
        "===== F. TRANSACTIONAL PATCH ====="
    )

    applied = False

    try:

        atomic_write(
            RUNNER,
            patched_source.encode(
                "utf-8"
            ),
        )

        applied = True

        rc_compile, output_compile = (
            run_command(
                [
                    sys.executable,
                    "-m",
                    "py_compile",
                    str(
                        RUNNER
                    ),
                ]
            )
        )

        require(
            rc_compile == 0,
            (
                "Timestamp-patched runner "
                "compile failed:\n"
                +
                output_compile
            ),
        )

        tests = regression()

        print(
            "patched runner compile = PASS"
        )

        print(
            "Stage6 regression      =",
            f"{tests} / {tests} PASS",
        )

    except BaseException:

        if applied:

            print()
            print(
                "PATCH FAILURE -> EXACT RUNNER ROLLBACK"
            )

            atomic_write(
                RUNNER,
                original_bytes,
            )

            require(
                file_sha(
                    RUNNER
                )
                ==
                EXPECTED_RUNNER_PRE_SHA,
                (
                    "Timestamp repair rollback "
                    "SHA mismatch."
                ),
            )

            print(
                "runner rollback = EXACT PASS"
            )

        raise

    post_runner_sha = file_sha(
        RUNNER
    )

    # --------------------------------------------------------
    # G. Report
    # --------------------------------------------------------

    result = {
        "stage":
            6,

        "block":
            "6.8_Part2C_timestamp_alignment_repair",

        "status":
            "PASS_REACTIVE_TIMESTAMP_ALIGNMENT_REPAIRED",

        "diagnosis": {
            "dataset_timestamp_failure":
                False,

            "future_index_failure":
                False,

            "metric_failure":
                False,

            "failure":
                (
                    "per-step Stage0 tolerance was "
                    "incorrectly applied to cumulative "
                    "elapsed horizon time"
                ),

            "example_observed_before_repair":
                {
                    "scenario_id":
                        "f72a05a05035feb8",

                    "nominal_horizon_s":
                        0.5,

                    "actual_elapsed_s":
                        0.50118,

                    "absolute_cumulative_error_s":
                        0.00118,

                    "used_to_select_new_tolerance":
                        False,
                },
        },

        "amendment": {
            "path":
                str(
                    AMENDMENT
                ),

            "sha256":
                amendment_sha,
        },

        "upstream_timing_authority":
            stage0_authority,

        "timestamp_audit":
            timestamp_audit,

        "runner": {
            "path":
                str(
                    RUNNER
                ),

            "pre_patch_sha256":
                EXPECTED_RUNNER_PRE_SHA,

            "post_patch_sha256":
                post_runner_sha,

            "backup_path":
                str(
                    BACKUP
                ),

            "backup_sha256":
                file_sha(
                    BACKUP
                ),

            "patch_scope":
                "single cumulative timestamp require statement",
        },

        "scientific_boundary": {
            "future_GT_opened":
                True,

            "future_GT_role":
                "evaluator_only",

            "reactive_metric_values_computed_before_repair":
                False,

            "predictive_performance_computed":
                False,

            "P_occ_opened":
                False,

            "NI_deltas_computed":
                False,

            "formal_evaluation":
                False,

            "controller_policy_tuning":
                False,
        },

        "regression": {
            "tests":
                tests,

            "status":
                "PASS",
        },

        "next":
            (
                "rerun unchanged reactive-only NI delta "
                "freeze using fixed offsets and upstream "
                "Stage0 per-adjacent-step timing gate"
            ),
    }

    atomic_write(
        PATCH_REPORT,
        canonical_json_bytes(
            result
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 TIMESTAMP ALIGNMENT REPAIR — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Stage0 timing authority      = PER-STEP EXACT PASS"
    )

    print(
        "nominal adjacent timestep    = 0.1 s"
    )

    print(
        "adjacent-step tolerance      = < 0.001 s"
    )

    print(
        "cumulative 1-ms gate         = REMOVED"
    )

    print(
        "future index offsets         = UNCHANGED [1,3,5,10]"
    )

    print(
        "interpolation                = NO"
    )

    print(
        "metric/rasterization change  = NO"
    )

    print(
        "preoutcome contract          = BYTE-UNCHANGED"
    )

    print(
        "timestamp amendment          = FROZEN"
    )

    print(
        "120-scenario timestamp audit = PASS"
    )

    print(
        "future GT                    = OPENED, EVALUATOR ONLY"
    )

    print(
        "reactive metrics before fix  = ZERO"
    )

    print(
        "predictive performance       = NOT COMPUTED"
    )

    print(
        "P_occ content                = NOT OPENED"
    )

    print(
        "NI deltas                    = NOT COMPUTED"
    )

    print(
        "formal evaluation            = NO"
    )

    print(
        "Stage6 regression            =",
        f"{tests} / {tests} PASS",
    )

    print(
        "patched runner SHA256        =",
        post_runner_sha,
    )

    print(
        "STATUS = "
        "PASS_REACTIVE_TIMESTAMP_ALIGNMENT_REPAIRED"
    )

    print(
        "report =",
        PATCH_REPORT,
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
        "BLOCK 6.8 TIMESTAMP ALIGNMENT REPAIR = BLOCKED"
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
        "future GT                = OPENED, EVALUATOR ONLY"
    )

    print(
        "reactive metric freeze   = ABSENT"
    )

    print(
        "predictive performance   = NOT COMPUTED"
    )

    print(
        "P_occ content            = NOT OPENED"
    )

    print(
        "NI deltas                = NOT COMPUTED"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "Do not loosen timestamp tolerances manually."
    )

    print(
        "Do not run predictive/formal evaluation."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
