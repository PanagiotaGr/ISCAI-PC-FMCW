from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
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

COHORT = (
    S6
    / "artifacts/block66/"
      "block66_class_aware_development_cohort_120.jsonl"
)

RUNNER = (
    S6
    / "scripts/"
      "run_block68_reactive_only_ni_delta_freeze.py"
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

FAILED_TIMESTAMP_AMENDMENT = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_timestamp_alignment_amendment.json"
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

FINAL_REACTIVE_REPORT = (
    S6
    / "reports/"
      "block68_reactive_only_ni_delta_freeze.json"
)

OUTPUT = (
    S6
    / "reports/"
      "block68_timestamp_structure_diagnostic.json"
)


EXPECTED_PREOUTCOME_SHA = (
    "8364c7cdfa4b66d820823c804d45905a"
    "207f77f66b53f06a6383e7722b8a9970"
)

EXPECTED_PROVENANCE_SHA = (
    "24119c2a029eaf20dc33c5baa66693c3"
    "1b6a41f61966304ce02f9c5793ae50b7"
)

EXPECTED_FAILED_AMENDMENT_SHA = (
    "986174594522f735646eb9c9bccbfce7"
    "c167d861c0ae6d13baae53b89786b525"
)

EXPECTED_TESTS = 224

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

NOMINAL_OFFSETS = {
    0.1: 1,
    0.3: 3,
    0.5: 5,
    1.0: 10,
}

SCENARIO_COUNT = 120


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


def read_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line_number, line in enumerate(
            stream,
            1,
        ):
            if not line.strip():
                continue

            value = json.loads(line)

            require(
                isinstance(value, dict),
                (
                    "Non-object JSONL row: "
                    f"{path}:{line_number}"
                ),
            )

            rows.append(value)

    return rows


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
        cwd=str(S6),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    count = None

    for line in process.stdout.splitlines():

        words = line.strip().split()

        if (
            len(words) >= 2
            and
            words[0] == "Ran"
        ):
            try:
                count = int(words[1])
            except Exception:
                pass

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                process.stdout.splitlines()[-100:]
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


def q(values, fraction):
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        return None

    return float(
        np.quantile(
            array,
            fraction,
        )
    )


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "DEVELOPMENT TIMESTAMP STRUCTURE DIAGNOSTIC"
    )
    print(
        "TIMESTAMPS ONLY / ZERO PERFORMANCE METRICS"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # A. Exact scientific boundary
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT PRE-METRIC BOUNDARY ====="
    )

    require(
        file_sha(
            PREOUTCOME_CONTRACT
        )
        ==
        EXPECTED_PREOUTCOME_SHA,
        "Preoutcome contract changed.",
    )

    require(
        file_sha(
            PROVENANCE_BINDING
        )
        ==
        EXPECTED_PROVENANCE_SHA,
        "Provenance binding changed.",
    )

    require(
        file_sha(
            FAILED_TIMESTAMP_AMENDMENT
        )
        ==
        EXPECTED_FAILED_AMENDMENT_SHA,
        (
            "Failed timestamp amendment "
            "changed unexpectedly."
        ),
    )

    for path, label in (
        (RAW_VALUES, "raw metric values"),
        (DELTA_FREEZE, "NI delta freeze"),
        (ACCEPTANCE_POLICY, "acceptance policy"),
        (FINAL_REACTIVE_REPORT, "final reactive report"),
    ):

        require(
            not path.exists(),
            (
                f"{label} unexpectedly exists: "
                f"{path}"
            ),
        )

    print(
        "preoutcome contract          = EXACT PASS"
    )
    print(
        "provenance binding           = EXACT PASS"
    )
    print(
        "failed timing amendment      = PRESERVED EXACT"
    )
    print(
        "reactive metric artifacts    = ABSENT"
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

    # --------------------------------------------------------
    # B. Show exact Stage0 timing assertion context
    # --------------------------------------------------------

    print()
    print(
        "===== B. STAGE0 TIMING ASSERTION CONTEXT ====="
    )

    require(
        STAGE0_TIMING_TEST.is_file(),
        (
            "Stage0 timing test missing."
        ),
    )

    source_lines = STAGE0_TIMING_TEST.read_text(
        encoding="utf-8"
    ).splitlines()

    timing_context = []

    for index, line in enumerate(
        source_lines,
        start=1,
    ):

        if (
            "abs(dt - 0.1)"
            in line
            or
            "timestamps[:-1]"
            in line
            or
            "timestamps[1:]"
            in line
        ):

            start = max(
                1,
                index - 8,
            )

            end = min(
                len(source_lines),
                index + 8,
            )

            excerpt = [
                {
                    "line":
                        line_number,

                    "text":
                        source_lines[
                            line_number - 1
                        ],
                }

                for line_number
                in range(
                    start,
                    end + 1,
                )
            ]

            timing_context.append(
                {
                    "match_line":
                        index,

                    "excerpt":
                        excerpt,
                }
            )

    require(
        timing_context,
        (
            "Could not locate Stage0 "
            "timing assertion context."
        ),
    )

    print(
        "Stage0 timing test SHA256 =",
        file_sha(
            STAGE0_TIMING_TEST
        ),
    )

    for block in timing_context:

        print()
        print(
            "context around line",
            block[
                "match_line"
            ],
        )

        for item in block[
            "excerpt"
        ]:

            print(
                f"  L{item['line']:04d}: "
                f"{item['text']}"
            )

    # --------------------------------------------------------
    # C. Timestamp-only cohort scan
    # --------------------------------------------------------

    print()
    print(
        "===== C. 120-SCENARIO TIMESTAMP STRUCTURE ====="
    )

    cohort = read_jsonl(
        COHORT
    )

    require(
        len(cohort)
        ==
        SCENARIO_COUNT,
        (
            "Development cohort is not 120."
        ),
    )

    scenario_summaries = []

    adjacent_dts = []
    adjacent_errors_from_100ms = []

    horizon_nearest_errors = defaultdict(list)
    horizon_nearest_offsets = defaultdict(list)
    horizon_nominal_offset_errors = defaultdict(list)

    scenarios_with_non_100ms_step = set()
    scenarios_with_nominal_offset_mismatch = set()
    scenarios_with_any_large_nearest_error = set()

    worst_adjacent = None

    worst_nearest = {
        horizon:
            None
        for horizon
        in HORIZONS_S
    }

    for cohort_index, row in enumerate(
        cohort
    ):

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
                "Scenario ID mismatch at "
                f"cohort index {cohort_index}."
            ),
        )

        anchor = int(
            scenario.current_time_index
        )

        timestamps = np.asarray(
            scenario.timestamps_seconds,
            dtype=np.float64,
        )

        require(
            timestamps.ndim == 1,
            f"{sid}: timestamps not 1-D.",
        )

        require(
            np.all(
                np.isfinite(
                    timestamps
                )
            ),
            f"{sid}: non-finite timestamp.",
        )

        require(
            0
            <=
            anchor
            <
            len(timestamps),
            f"{sid}: invalid anchor.",
        )

        # Need enough timestamp coverage to inspect beyond 1 s.
        future_indices = np.arange(
            anchor + 1,
            len(timestamps),
            dtype=np.int64,
        )

        require(
            len(future_indices) > 0,
            f"{sid}: no future timestamps.",
        )

        future_times = timestamps[
            future_indices
        ]

        require(
            np.all(
                np.diff(
                    timestamps
                )
                >
                0.0
            ),
            (
                f"{sid}: timestamps "
                "not strictly monotonic."
            ),
        )

        # Inspect the first region that covers at least
        # nominal 1.2 s after anchor where possible.
        anchor_time = float(
            timestamps[
                anchor
            ]
        )

        relative_future = (
            future_times
            -
            anchor_time
        )

        inspection_mask = (
            relative_future
            <=
            1.25
        )

        inspection_indices = future_indices[
            inspection_mask
        ]

        if len(
            inspection_indices
        ) == 0:

            inspection_indices = future_indices[
                :min(
                    15,
                    len(
                        future_indices
                    ),
                )
            ]

        inspection_end = int(
            inspection_indices[-1]
        )

        window = timestamps[
            anchor:
            inspection_end + 1
        ]

        dts = np.diff(
            window
        )

        for local_index, dt in enumerate(
            dts,
            start=1,
        ):

            dt_value = float(dt)

            error = abs(
                dt_value
                -
                0.1
            )

            adjacent_dts.append(
                dt_value
            )

            adjacent_errors_from_100ms.append(
                error
            )

            if error >= 0.001:

                scenarios_with_non_100ms_step.add(
                    sid
                )

            if (
                worst_adjacent is None
                or
                error
                >
                worst_adjacent[
                    "abs_error_from_0p1_s"
                ]
            ):

                worst_adjacent = {
                    "scenario_id":
                        sid,

                    "anchor_index":
                        anchor,

                    "absolute_index_before":
                        anchor
                        +
                        local_index
                        -
                        1,

                    "absolute_index_after":
                        anchor
                        +
                        local_index,

                    "dt_s":
                        dt_value,

                    "abs_error_from_0p1_s":
                        error,
                }

        horizon_rows = []

        for horizon in HORIZONS_S:

            target_time = (
                anchor_time
                +
                horizon
            )

            differences = np.abs(
                future_times
                -
                target_time
            )

            nearest_position = int(
                np.argmin(
                    differences
                )
            )

            nearest_index = int(
                future_indices[
                    nearest_position
                ]
            )

            nearest_offset = (
                nearest_index
                -
                anchor
            )

            nearest_time = float(
                timestamps[
                    nearest_index
                ]
            )

            nearest_elapsed = (
                nearest_time
                -
                anchor_time
            )

            nearest_error = abs(
                nearest_elapsed
                -
                horizon
            )

            nominal_offset = (
                NOMINAL_OFFSETS[
                    horizon
                ]
            )

            nominal_index = (
                anchor
                +
                nominal_offset
            )

            nominal_available = (
                nominal_index
                <
                len(
                    timestamps
                )
            )

            if nominal_available:

                nominal_elapsed = (
                    float(
                        timestamps[
                            nominal_index
                        ]
                    )
                    -
                    anchor_time
                )

                nominal_error = abs(
                    nominal_elapsed
                    -
                    horizon
                )

            else:

                nominal_elapsed = None
                nominal_error = None

            # Bracket nominal target by timestamps.
            lower_candidates = np.flatnonzero(
                timestamps
                <=
                target_time
            )

            upper_candidates = np.flatnonzero(
                timestamps
                >=
                target_time
            )

            lower_index = (
                int(
                    lower_candidates[-1]
                )
                if len(
                    lower_candidates
                )
                else None
            )

            upper_index = (
                int(
                    upper_candidates[0]
                )
                if len(
                    upper_candidates
                )
                else None
            )

            bracketed = (
                lower_index
                is not None
                and
                upper_index
                is not None
                and
                lower_index
                <=
                upper_index
            )

            bracket_width = None

            if bracketed:

                bracket_width = (
                    float(
                        timestamps[
                            upper_index
                        ]
                    )
                    -
                    float(
                        timestamps[
                            lower_index
                        ]
                    )
                )

            horizon_nearest_errors[
                horizon
            ].append(
                nearest_error
            )

            horizon_nearest_offsets[
                horizon
            ].append(
                nearest_offset
            )

            if nominal_error is not None:

                horizon_nominal_offset_errors[
                    horizon
                ].append(
                    nominal_error
                )

            if (
                nearest_offset
                !=
                nominal_offset
            ):

                scenarios_with_nominal_offset_mismatch.add(
                    sid
                )

            # Diagnostic marker only; not a gate/tolerance.
            if nearest_error > 0.01:

                scenarios_with_any_large_nearest_error.add(
                    sid
                )

            current_worst = (
                worst_nearest[
                    horizon
                ]
            )

            if (
                current_worst is None
                or
                nearest_error
                >
                current_worst[
                    "absolute_error_s"
                ]
            ):

                worst_nearest[
                    horizon
                ] = {
                    "scenario_id":
                        sid,

                    "anchor_index":
                        anchor,

                    "nominal_horizon_s":
                        horizon,

                    "nominal_offset":
                        nominal_offset,

                    "nearest_offset":
                        nearest_offset,

                    "nearest_elapsed_s":
                        nearest_elapsed,

                    "absolute_error_s":
                        nearest_error,

                    "nominal_offset_elapsed_s":
                        nominal_elapsed,

                    "nominal_offset_error_s":
                        nominal_error,

                    "bracketed":
                        bracketed,

                    "lower_index":
                        lower_index,

                    "upper_index":
                        upper_index,

                    "bracket_width_s":
                        bracket_width,
                }

            horizon_rows.append(
                {
                    "nominal_horizon_s":
                        horizon,

                    "target_absolute_time_s":
                        target_time,

                    "nominal_offset":
                        nominal_offset,

                    "nominal_offset_elapsed_s":
                        nominal_elapsed,

                    "nominal_offset_absolute_error_s":
                        nominal_error,

                    "nearest_absolute_index":
                        nearest_index,

                    "nearest_offset":
                        nearest_offset,

                    "nearest_elapsed_s":
                        nearest_elapsed,

                    "nearest_absolute_error_s":
                        nearest_error,

                    "target_bracketed":
                        bracketed,

                    "lower_index":
                        lower_index,

                    "upper_index":
                        upper_index,

                    "bracket_width_s":
                        bracket_width,
                }
            )

        scenario_summaries.append(
            {
                "cohort_index":
                    cohort_index,

                "scenario_id":
                    sid,

                "anchor_index":
                    anchor,

                "anchor_timestamp_s":
                    anchor_time,

                "inspection_timestamps_s":
                    [
                        float(value)
                        for value in window
                    ],

                "inspection_adjacent_dts_s":
                    [
                        float(value)
                        for value in dts
                    ],

                "horizons":
                    horizon_rows,
            }
        )

    print(
        "development scenarios        = 120 / 120"
    )

    print(
        "timestamp arrays monotonic   = PASS"
    )

    print(
        "scenarios with >=1ms "
        "adjacent-step deviation      =",
        len(
            scenarios_with_non_100ms_step
        ),
    )

    print(
        "scenarios where nearest "
        "offset differs from nominal =",
        len(
            scenarios_with_nominal_offset_mismatch
        ),
    )

    print(
        "scenarios with nearest error "
        ">10ms (diagnostic only)      =",
        len(
            scenarios_with_any_large_nearest_error
        ),
    )

    print()
    print(
        "worst adjacent interval:"
    )
    print(
        json.dumps(
            worst_adjacent,
            indent=2,
            sort_keys=True,
        )
    )

    # --------------------------------------------------------
    # D. Horizon summary
    # --------------------------------------------------------

    print()
    print(
        "===== D. NOMINAL-HORIZON TIMESTAMP RESOLUTION ====="
    )

    horizon_summary = {}

    for horizon in HORIZONS_S:

        errors = horizon_nearest_errors[
            horizon
        ]

        nominal_errors = (
            horizon_nominal_offset_errors[
                horizon
            ]
        )

        offset_counts = Counter(
            horizon_nearest_offsets[
                horizon
            ]
        )

        summary = {
            "nominal_horizon_s":
                horizon,

            "nominal_fixed_offset":
                NOMINAL_OFFSETS[
                    horizon
                ],

            "nearest_offset_distribution":
                {
                    str(
                        offset
                    ):
                        int(
                            count
                        )
                    for offset, count
                    in sorted(
                        offset_counts.items()
                    )
                },

            "nearest_abs_error_s": {
                "min":
                    float(
                        np.min(
                            errors
                        )
                    ),

                "median":
                    float(
                        np.median(
                            errors
                        )
                    ),

                "p90":
                    q(
                        errors,
                        0.90,
                    ),

                "p95":
                    q(
                        errors,
                        0.95,
                    ),

                "p99":
                    q(
                        errors,
                        0.99,
                    ),

                "max":
                    float(
                        np.max(
                            errors
                        )
                    ),
            },

            "fixed_nominal_offset_abs_error_s": {
                "median":
                    (
                        float(
                            np.median(
                                nominal_errors
                            )
                        )
                        if nominal_errors
                        else None
                    ),

                "max":
                    (
                        float(
                            np.max(
                                nominal_errors
                            )
                        )
                        if nominal_errors
                        else None
                    ),
            },

            "worst_nearest_case":
                worst_nearest[
                    horizon
                ],
        }

        horizon_summary[
            str(
                horizon
            )
        ] = summary

        print()
        print(
            f"horizon = {horizon:.1f}s"
        )

        print(
            "  nearest offset distribution =",
            summary[
                "nearest_offset_distribution"
            ],
        )

        print(
            "  nearest abs error median =",
            summary[
                "nearest_abs_error_s"
            ][
                "median"
            ],
        )

        print(
            "  nearest abs error p95    =",
            summary[
                "nearest_abs_error_s"
            ][
                "p95"
            ],
        )

        print(
            "  nearest abs error max    =",
            summary[
                "nearest_abs_error_s"
            ][
                "max"
            ],
        )

        print(
            "  fixed-offset error max   =",
            summary[
                "fixed_nominal_offset_abs_error_s"
            ][
                "max"
            ],
        )

        print(
            "  worst nearest case       =",
            json.dumps(
                summary[
                    "worst_nearest_case"
                ],
                sort_keys=True,
            ),
        )

    # --------------------------------------------------------
    # E. Explicit failing scenario
    # --------------------------------------------------------

    print()
    print(
        "===== E. PREVIOUS FAILING SCENARIO b66168b7cf95bf84 ====="
    )

    failing = [
        item
        for item in scenario_summaries
        if (
            item[
                "scenario_id"
            ]
            ==
            "b66168b7cf95bf84"
        )
    ]

    require(
        len(failing) == 1,
        (
            "Previous failing scenario "
            "not found exactly once."
        ),
    )

    print(
        json.dumps(
            failing[0],
            indent=2,
            sort_keys=True,
        )
    )

    # --------------------------------------------------------
    # F. Scientific boundary / regression
    # --------------------------------------------------------

    print()
    print(
        "===== F. SCIENTIFIC BOUNDARY ====="
    )

    print(
        "actor future states          = NOT READ"
    )
    print(
        "future box geometry          = NOT READ"
    )
    print(
        "reactive illumination values = NOT READ"
    )
    print(
        "performance metrics          = NOT COMPUTED"
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
        "timestamp policy changed     = NO"
    )

    tests = run_regression()

    print(
        "Stage6 regression            =",
        f"{tests} / {tests} PASS",
    )

    # --------------------------------------------------------
    # G. Report
    # --------------------------------------------------------

    result = {
        "stage":
            6,

        "block":
            "6.8_timestamp_structure_diagnostic",

        "status":
            "PASS_TIMESTAMP_STRUCTURE_DIAGNOSTIC_ONLY",

        "purpose":
            (
                "determine actual timestamp/index "
                "relationship before choosing a "
                "replacement evaluator alignment rule"
            ),

        "prior_failed_amendment": {
            "path":
                str(
                    FAILED_TIMESTAMP_AMENDMENT
                ),

            "sha256":
                EXPECTED_FAILED_AMENDMENT_SHA,

            "preserved":
                True,

            "authoritative_for_next_run":
                False,

            "reason":
                (
                    "120-scenario audit demonstrated "
                    "that strict adjacent 100-ms spacing "
                    "does not hold for entire development "
                    "cohort"
                ),
        },

        "stage0_assertion_context":
            timing_context,

        "cohort_timestamp_structure": {
            "scenario_count":
                SCENARIO_COUNT,

            "scenarios_with_ge_1ms_adjacent_error":
                len(
                    scenarios_with_non_100ms_step
                ),

            "scenarios_with_nearest_offset_different_from_nominal":
                len(
                    scenarios_with_nominal_offset_mismatch
                ),

            "scenarios_with_nearest_error_gt_10ms_diagnostic":
                len(
                    scenarios_with_any_large_nearest_error
                ),

            "worst_adjacent_interval":
                worst_adjacent,

            "adjacent_dt_summary_s": {
                "min":
                    float(
                        np.min(
                            adjacent_dts
                        )
                    ),

                "median":
                    float(
                        np.median(
                            adjacent_dts
                        )
                    ),

                "max":
                    float(
                        np.max(
                            adjacent_dts
                        )
                    ),
            },

            "horizon_summary":
                horizon_summary,
        },

        "previous_failing_scenario":
            failing[0],

        "all_scenario_timestamp_diagnostics":
            scenario_summaries,

        "scientific_boundary": {
            "actor_future_states_read":
                False,

            "performance_metrics_computed":
                False,

            "reactive_illumination_values_read":
                False,

            "predictive_performance_computed":
                False,

            "P_occ_opened":
                False,

            "NI_deltas_computed":
                False,

            "formal_evaluation":
                False,

            "policy_or_controller_modified":
                False,

            "timestamp_alignment_rule_modified":
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
                "freeze a superseding timestamp alignment "
                "rule using only this timestamp-structure "
                "evidence; do not use performance outcomes"
            ),
    }

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 TIMESTAMP STRUCTURE DIAGNOSTIC — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "development scenarios     = 120 / 120"
    )
    print(
        "timestamp-only access     = YES"
    )
    print(
        "actor future states       = NOT READ"
    )
    print(
        "performance metrics       = NOT COMPUTED"
    )
    print(
        "predictive performance    = NOT COMPUTED"
    )
    print(
        "P_occ content             = NOT OPENED"
    )
    print(
        "NI deltas                 = NOT COMPUTED"
    )
    print(
        "formal evaluation         = NO"
    )
    print(
        "alignment policy selected = NO"
    )
    print(
        "Stage6 regression         =",
        f"{tests} / {tests} PASS",
    )
    print(
        "STATUS = "
        "PASS_TIMESTAMP_STRUCTURE_DIAGNOSTIC_ONLY"
    )
    print(
        "report =",
        OUTPUT,
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
        "BLOCK 6.8 TIMESTAMP STRUCTURE DIAGNOSTIC = BLOCKED"
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
        "performance metrics    = NOT COMPUTED"
    )
    print(
        "predictive performance = NOT COMPUTED"
    )
    print(
        "P_occ content          = NOT OPENED"
    )
    print(
        "NI deltas              = NOT COMPUTED"
    )
    print(
        "formal evaluation      = NO"
    )
    print(
        "Do not choose a new timestamp "
        "tolerance or interpolation rule yet."
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
