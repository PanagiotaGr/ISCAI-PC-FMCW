from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
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

PREOUTCOME = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_scoring_contract.json"
)

PROVENANCE = (
    S6
    / "configs/"
      "stage6_future_gt_evaluator_provenance_binding.json"
)

FAILED_AMENDMENT = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_timestamp_alignment_amendment.json"
)

DIAGNOSTIC = (
    S6
    / "reports/"
      "block68_timestamp_structure_diagnostic.json"
)

SUPERSEDING = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_timestamp_alignment_superseding.json"
)

PATCH_REPORT = (
    S6
    / "reports/"
      "block68_timestamp_alignment_superseding_repair.json"
)

BACKUP = (
    S6
    / "artifacts/block68/"
      "timestamp_alignment_superseding/"
      "run_block68_reactive_only_ni_delta_freeze.pre_superseding_patch.py"
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


EXPECTED_RUNNER_SHA = (
    "35fedb5ac974140626ec64ab95ff1423"
    "28366a77bffeb2815bce16e4953249dd"
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
EXPECTED_SCENARIOS = 120

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


# ============================================================
# Helpers
# ============================================================

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
                isinstance(
                    value,
                    dict,
                ),
                (
                    f"Invalid JSONL object "
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
            path.read_bytes() == payload,
            (
                "Existing frozen artifact differs: "
                f"{path}"
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

    count = None

    for line in output.splitlines():

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(match.group(1))

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
# Timestamp-only proof
# ============================================================

def resolve_nearest(
    timestamps,
    *,
    anchor,
    horizon_s,
):
    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    anchor_time = float(
        timestamps[
            anchor
        ]
    )

    candidate_indices = np.arange(
        anchor + 1,
        len(timestamps),
        dtype=np.int64,
    )

    require(
        len(candidate_indices) > 0,
        "No future timestamp candidates.",
    )

    target_time = (
        anchor_time
        +
        float(horizon_s)
    )

    candidate_errors = np.abs(
        timestamps[
            candidate_indices
        ]
        -
        target_time
    )

    require(
        np.all(
            np.isfinite(
                candidate_errors
            )
        ),
        "Non-finite timestamp error.",
    )

    nearest_position = int(
        np.argmin(
            candidate_errors
        )
    )

    selected_index = int(
        candidate_indices[
            nearest_position
        ]
    )

    selected_error = float(
        candidate_errors[
            nearest_position
        ]
    )

    selected_elapsed = (
        float(
            timestamps[
                selected_index
            ]
        )
        -
        anchor_time
    )

    # Deterministic tie semantics:
    # np.argmin chooses the first/smallest future index.
    minimum_error = float(
        np.min(
            candidate_errors
        )
    )

    tie_positions = np.flatnonzero(
        np.isclose(
            candidate_errors,
            minimum_error,
            rtol=0.0,
            atol=1.0e-12,
        )
    )

    tie_count = int(
        len(
            tie_positions
        )
    )

    require(
        tie_count >= 1,
        "Nearest timestamp tie accounting failed.",
    )

    earliest_tied_index = int(
        candidate_indices[
            tie_positions[
                0
            ]
        ]
    )

    require(
        selected_index
        ==
        earliest_tied_index,
        (
            "Nearest timestamp tie-break "
            "is not earliest-index deterministic."
        ),
    )

    lower = np.flatnonzero(
        timestamps
        <=
        target_time
    )

    upper = np.flatnonzero(
        timestamps
        >=
        target_time
    )

    bracketed = (
        len(lower) > 0
        and
        len(upper) > 0
        and
        int(lower[-1])
        <=
        int(upper[0])
    )

    require(
        bracketed,
        (
            "Nominal horizon is not bracketed "
            "by observed timestamps."
        ),
    )

    return {
        "selected_index":
            selected_index,

        "selected_offset":
            int(
                selected_index
                -
                anchor
            ),

        "selected_elapsed_s":
            selected_elapsed,

        "absolute_error_s":
            selected_error,

        "tie_count":
            tie_count,

        "bracketed":
            True,

        "lower_index":
            int(
                lower[-1]
            ),

        "upper_index":
            int(
                upper[0]
            ),
    }


def timestamp_only_revalidation():

    cohort = read_jsonl(
        COHORT
    )

    require(
        len(cohort)
        ==
        EXPECTED_SCENARIOS,
        (
            "Development cohort is not 120."
        ),
    )

    distributions = {
        horizon:
            Counter()
        for horizon in HORIZONS_S
    }

    maximum_errors = {
        horizon:
            0.0
        for horizon in HORIZONS_S
    }

    tie_counts = {
        horizon:
            0
        for horizon in HORIZONS_S
    }

    different_from_nominal_scenarios = set()

    scenario_records = []

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
            np.all(
                np.diff(
                    timestamps
                )
                >
                0.0
            ),
            f"{sid}: timestamps not monotonic.",
        )

        selected_indices = []

        horizon_records = []

        for horizon in HORIZONS_S:

            resolved = resolve_nearest(
                timestamps,
                anchor=anchor,
                horizon_s=horizon,
            )

            selected_indices.append(
                resolved[
                    "selected_index"
                ]
            )

            distributions[
                horizon
            ][
                resolved[
                    "selected_offset"
                ]
            ] += 1

            maximum_errors[
                horizon
            ] = max(
                maximum_errors[
                    horizon
                ],
                resolved[
                    "absolute_error_s"
                ],
            )

            if (
                resolved[
                    "tie_count"
                ]
                >
                1
            ):

                tie_counts[
                    horizon
                ] += 1

            if (
                resolved[
                    "selected_offset"
                ]
                !=
                NOMINAL_OFFSETS[
                    horizon
                ]
            ):

                different_from_nominal_scenarios.add(
                    sid
                )

            horizon_records.append(
                {
                    "horizon_s":
                        horizon,

                    **resolved,
                }
            )

        require(
            all(
                later > earlier
                for earlier, later
                in zip(
                    selected_indices[:-1],
                    selected_indices[1:],
                )
            ),
            (
                "Resolved future indices are "
                f"not strictly increasing: {sid}"
            ),
        )

        scenario_records.append(
            {
                "scenario_id":
                    sid,

                "anchor_index":
                    anchor,

                "resolved":
                    horizon_records,
            }
        )

    expected_distributions = {
        0.1:
            {1: 120},

        0.3:
            {3: 120},

        0.5:
            {5: 120},

        1.0:
            {
                9: 2,
                10: 118,
            },
    }

    for horizon in HORIZONS_S:

        require(
            dict(
                distributions[
                    horizon
                ]
            )
            ==
            expected_distributions[
                horizon
            ],
            (
                "Timestamp resolution differs "
                "from frozen diagnostic for "
                f"h={horizon}: "
                f"{dict(distributions[horizon])}"
            ),
        )

    require(
        len(
            different_from_nominal_scenarios
        )
        ==
        2,
        (
            "Expected exactly two scenarios "
            "requiring non-nominal index offset."
        ),
    )

    return {
        "scenario_count":
            EXPECTED_SCENARIOS,

        "selection":
            (
                "argmin over i>anchor of "
                "|(timestamp[i]-timestamp[anchor])-horizon|"
            ),

        "tie_break":
            "smallest/earliest absolute future index",

        "interpolation":
            False,

        "absolute_error_acceptance_threshold":
            None,

        "target_bracketing_required":
            True,

        "resolved_indices_strictly_increasing":
            True,

        "nearest_offset_distribution": {
            str(horizon):
                {
                    str(offset):
                        int(count)
                    for offset, count
                    in sorted(
                        distributions[
                            horizon
                        ].items()
                    )
                }
            for horizon
            in HORIZONS_S
        },

        "maximum_observed_nearest_error_s": {
            str(horizon):
                float(
                    maximum_errors[
                        horizon
                    ]
                )
            for horizon
            in HORIZONS_S
        },

        "nearest_tie_scenario_counts": {
            str(horizon):
                int(
                    tie_counts[
                        horizon
                    ]
                )
            for horizon
            in HORIZONS_S
        },

        "scenarios_with_non_nominal_offset":
            sorted(
                different_from_nominal_scenarios
            ),

        "all_scenarios":
            scenario_records,
    }


# ============================================================
# Exact source patch
# ============================================================

OLD_FUTURE_INDEX = '''                future_index = (
                    anchor_index
                    +
                    offset
                )
'''


NEW_FUTURE_INDEX_TEMPLATE = '''                # SUPERSEDING_TIMESTAMP_ALIGNMENT
                # Nominal offset is diagnostic only.
                # Actual future state is selected from
                # observed timestamps nearest to the
                # requested physical horizon.

                _timestamps = np.asarray(
                    scenario.timestamps_seconds,
                    dtype=np.float64,
                )

                _target_timestamp = (
                    anchor_timestamp
                    +
                    float(
                        horizon_s
                    )
                )

                _candidate_indices = np.arange(
                    anchor_index + 1,
                    len(
                        _timestamps
                    ),
                    dtype=np.int64,
                )

                require(
                    len(
                        _candidate_indices
                    )
                    >
                    0,
                    (
                        "No future timestamp candidates "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                _timestamp_errors = np.abs(
                    _timestamps[
                        _candidate_indices
                    ]
                    -
                    _target_timestamp
                )

                require(
                    np.all(
                        np.isfinite(
                            _timestamp_errors
                        )
                    ),
                    (
                        "Non-finite timestamp resolution "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                _nearest_position = int(
                    np.argmin(
                        _timestamp_errors
                    )
                )

                future_index = int(
                    _candidate_indices[
                        _nearest_position
                    ]
                )

                _minimum_error = float(
                    np.min(
                        _timestamp_errors
                    )
                )

                _tie_positions = np.flatnonzero(
                    np.isclose(
                        _timestamp_errors,
                        _minimum_error,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )

                require(
                    len(
                        _tie_positions
                    )
                    >=
                    1,
                    (
                        "Nearest timestamp resolution "
                        f"failed sid={sid} h={horizon_s}"
                    ),
                )

                _earliest_tied_index = int(
                    _candidate_indices[
                        _tie_positions[
                            0
                        ]
                    ]
                )

                require(
                    future_index
                    ==
                    _earliest_tied_index,
                    (
                        "Timestamp tie-break changed "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                _lower = np.flatnonzero(
                    _timestamps
                    <=
                    _target_timestamp
                )

                _upper = np.flatnonzero(
                    _timestamps
                    >=
                    _target_timestamp
                )

                require(
                    (
                        len(
                            _lower
                        )
                        >
                        0
                        and
                        len(
                            _upper
                        )
                        >
                        0
                    ),
                    (
                        "Nominal horizon not bracketed "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                _resolved_offset = (
                    future_index
                    -
                    anchor_index
                )
'''


OLD_TIMESTAMP_GATE = '''                require(
                    abs(
                        actual_dt
                        -
                        horizon_s
                    )
                    <=
                    TIMESTAMP_TOLERANCE_S,
                    (
                        "WOMD horizon timestamp mismatch "
                        f"sid={sid} "
                        f"h={horizon_s} "
                        f"actual={actual_dt}"
                    ),
                )
'''


NEW_TIMESTAMP_GATE = '''                require(
                    math.isfinite(
                        actual_dt
                    ),
                    (
                        "Resolved future elapsed time "
                        "is non-finite "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                # Diagnostic only. There is deliberately no
                # post-hoc numerical tolerance gate.
                _resolved_timestamp_error_s = abs(
                    actual_dt
                    -
                    float(
                        horizon_s
                    )
                )
'''


OLD_PRINT = (
    '"future index offsets     = [1, 3, 5, 10]"'
)

NEW_PRINT = (
    '"nominal offset labels    = [1, 3, 5, 10] (diagnostic only)"'
)


def patch_runner(
    source,
    *,
    alignment_sha,
):

    require(
        source.count(
            OLD_FUTURE_INDEX
        )
        ==
        1,
        (
            "Expected exactly one direct "
            "anchor+offset future-index assignment."
        ),
    )

    require(
        source.count(
            OLD_TIMESTAMP_GATE
        )
        ==
        1,
        (
            "Expected exactly one cumulative "
            "timestamp tolerance gate."
        ),
    )

    require(
        source.count(
            OLD_PRINT
        )
        ==
        1,
        (
            "Expected exactly one fixed-offset "
            "status print."
        ),
    )

    patched = source.replace(
        OLD_FUTURE_INDEX,
        NEW_FUTURE_INDEX_TEMPLATE,
        1,
    )

    patched = patched.replace(
        OLD_TIMESTAMP_GATE,
        NEW_TIMESTAMP_GATE,
        1,
    )

    patched = patched.replace(
        OLD_PRINT,
        NEW_PRINT,
        1,
    )

    marker = '''    # --------------------------------------------------------
    # E. FIRST scientific outcome access
'''

    require(
        patched.count(
            marker
        )
        ==
        1,
        (
            "Could not locate Section-E "
            "scientific-access boundary."
        ),
    )

    alignment_gate = f'''    # --------------------------------------------------------
    # Superseding timestamp-alignment authority
    # --------------------------------------------------------

    _alignment_path = (
        S6
        / "configs/"
          "stage6_reactive_future_truth_timestamp_alignment_superseding.json"
    )

    require(
        _alignment_path.is_file(),
        (
            "Superseding timestamp alignment "
            f"missing: {{_alignment_path}}"
        ),
    )

    require(
        file_sha(
            _alignment_path
        )
        ==
        "{alignment_sha}",
        (
            "Superseding timestamp-alignment "
            "SHA changed."
        ),
    )

    _alignment_contract = read_json(
        _alignment_path
    )

    require(
        _alignment_contract.get(
            "status"
        )
        ==
        "FROZEN_SUPERSEDING_TIMESTAMP_ALIGNMENT_RULE",
        (
            "Superseding timestamp-alignment "
            "status mismatch."
        ),
    )

    print(
        "timestamp alignment       = "
        "NEAREST OBSERVED FUTURE SAMPLE"
    )

    print(
        "interpolation             = NO"
    )

    print(
        "timestamp tolerance gate  = NONE"
    )

'''

    patched = patched.replace(
        marker,
        alignment_gate
        +
        marker,
        1,
    )

    # Add the superseding alignment seal to the final report.
    old_frozen_inputs = '''        "frozen_inputs":
            preoutcome_contract[
                "frozen_input_hashes"
            ],
'''

    new_frozen_inputs = f'''        "frozen_inputs": {{
            **preoutcome_contract[
                "frozen_input_hashes"
            ],

            "superseding_timestamp_alignment":
                "{alignment_sha}",
        }},
'''

    require(
        patched.count(
            old_frozen_inputs
        )
        ==
        1,
        (
            "Could not locate final frozen-input "
            "report payload."
        ),
    )

    patched = patched.replace(
        old_frozen_inputs,
        new_frozen_inputs,
        1,
    )

    compile(
        patched,
        str(
            RUNNER
        ),
        "exec",
    )

    require(
        "SUPERSEDING_TIMESTAMP_ALIGNMENT"
        in
        patched,
        (
            "Nearest-sample runtime marker missing."
        ),
    )

    require(
        "WOMD horizon timestamp mismatch"
        not in
        patched,
        (
            "Old cumulative tolerance gate "
            "remains active."
        ),
    )

    require(
        alignment_sha
        in
        patched,
        (
            "Patched runner does not seal "
            "superseding alignment SHA."
        ),
    )

    return patched


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "SUPERSEDING TIMESTAMP ALIGNMENT FREEZE"
    )
    print(
        "NEAREST OBSERVED SAMPLE / NO INTERPOLATION"
    )
    print(
        "NO PERFORMANCE-BASED SELECTION"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # A. Exact boundary
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT PRE-METRIC BOUNDARY ====="
    )

    require(
        file_sha(
            RUNNER
        )
        ==
        EXPECTED_RUNNER_SHA,
        (
            "Reactive-only runner differs "
            "from exact provenance-repaired SHA."
        ),
    )

    require(
        file_sha(
            PREOUTCOME
        )
        ==
        EXPECTED_PREOUTCOME_SHA,
        (
            "Original preoutcome contract changed."
        ),
    )

    require(
        file_sha(
            PROVENANCE
        )
        ==
        EXPECTED_PROVENANCE_SHA,
        (
            "Future-GT provenance binding changed."
        ),
    )

    require(
        file_sha(
            FAILED_AMENDMENT
        )
        ==
        EXPECTED_FAILED_AMENDMENT_SHA,
        (
            "Failed timestamp amendment changed."
        ),
    )

    require(
        DIAGNOSTIC.is_file(),
        (
            "Timestamp diagnostic report missing."
        ),
    )

    diagnostic = read_json(
        DIAGNOSTIC
    )

    require(
        diagnostic.get(
            "status"
        )
        ==
        "PASS_TIMESTAMP_STRUCTURE_DIAGNOSTIC_ONLY",
        (
            "Timestamp diagnostic is not PASS."
        ),
    )

    for path, label in (
        (
            RAW_VALUES,
            "reactive metric artifact",
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
            "reactive final report",
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
        "runner                       = EXACT PASS"
    )
    print(
        "preoutcome contract          = EXACT PASS"
    )
    print(
        "provenance binding           = EXACT PASS"
    )
    print(
        "failed amendment             = PRESERVED"
    )
    print(
        "timestamp diagnostic         = PASS"
    )
    print(
        "metric artifacts             = ABSENT"
    )
    print(
        "predictive performance       = NOT COMPUTED"
    )
    print(
        "NI deltas                    = NOT COMPUTED"
    )

    # --------------------------------------------------------
    # B. Freeze diagnostic facts
    # --------------------------------------------------------

    print()
    print(
        "===== B. DIAGNOSTIC FACT READBACK ====="
    )

    structure = diagnostic[
        "cohort_timestamp_structure"
    ]

    require(
        int(
            structure[
                "scenario_count"
            ]
        )
        ==
        120,
        "Diagnostic scenario count != 120.",
    )

    require(
        int(
            structure[
                "scenarios_with_ge_1ms_adjacent_error"
            ]
        )
        ==
        2,
        (
            "Expected two scenarios with "
            "large adjacent-step deviation."
        ),
    )

    require(
        int(
            structure[
                "scenarios_with_nearest_offset_different_from_nominal"
            ]
        )
        ==
        2,
        (
            "Expected two scenarios with "
            "nearest-offset difference."
        ),
    )

    require(
        int(
            structure[
                "scenarios_with_nearest_error_gt_10ms_diagnostic"
            ]
        )
        ==
        0,
        (
            "Diagnostic nearest error >10 ms "
            "count unexpectedly changed."
        ),
    )

    expected_distributions = {
        "0.1":
            {
                "1":
                    120,
            },

        "0.3":
            {
                "3":
                    120,
            },

        "0.5":
            {
                "5":
                    120,
            },

        "1.0":
            {
                "9":
                    2,
                "10":
                    118,
            },
    }

    for horizon, expected in (
        expected_distributions.items()
    ):

        actual = (
            structure[
                "horizon_summary"
            ][
                horizon
            ][
                "nearest_offset_distribution"
            ]
        )

        require(
            actual == expected,
            (
                "Nearest-offset diagnostic changed "
                f"for horizon={horizon}: {actual}"
            ),
        )

    print(
        "0.1 s nearest offsets = {1:120}"
    )
    print(
        "0.3 s nearest offsets = {3:120}"
    )
    print(
        "0.5 s nearest offsets = {5:120}"
    )
    print(
        "1.0 s nearest offsets = {9:2, 10:118}"
    )
    print(
        "nearest error >10 ms  = 0 / 120"
    )

    # --------------------------------------------------------
    # C. Independent timestamp-only revalidation
    # --------------------------------------------------------

    print()
    print(
        "===== C. TIMESTAMP-ONLY ALIGNMENT REVALIDATION ====="
    )

    resolved = (
        timestamp_only_revalidation()
    )

    print(
        "development scenarios   = 120 / 120 PASS"
    )
    print(
        "targets bracketed        = 120 / 120 × 4 PASS"
    )
    print(
        "selected indices         = STRICTLY INCREASING PASS"
    )
    print(
        "interpolation required   = NO"
    )
    print(
        "numeric tolerance needed = NO"
    )

    print()
    print(
        "maximum observed nearest errors "
        "(DIAGNOSTIC ONLY):"
    )

    for horizon in HORIZONS_S:

        print(
            f"  {horizon:.1f}s =",
            resolved[
                "maximum_observed_nearest_error_s"
            ][
                str(
                    horizon
                )
            ],
        )

    # --------------------------------------------------------
    # D. Freeze superseding alignment rule
    # --------------------------------------------------------

    print()
    print(
        "===== D. SUPERSEDING ALIGNMENT RULE FREEZE ====="
    )

    diagnostic_sha = file_sha(
        DIAGNOSTIC
    )

    superseding = {
        "stage":
            6,

        "block":
            "6.8_timestamp_alignment_superseding",

        "status":
            "FROZEN_SUPERSEDING_TIMESTAMP_ALIGNMENT_RULE",

        "freeze_timing":
            (
                "before_any_reactive_metric_value "
                "or_NI_delta"
            ),

        "reason":
            (
                "development timestamp-only diagnostic "
                "showed missing/irregular timestamp slots "
                "in two scenarios; fixed positional offsets "
                "do not universally represent the requested "
                "physical horizon"
            ),

        "authority_chain": {
            "original_preoutcome_contract": {
                "path":
                    str(
                        PREOUTCOME
                    ),

                "sha256":
                    EXPECTED_PREOUTCOME_SHA,

                "modified":
                    False,
            },

            "failed_per_step_amendment": {
                "path":
                    str(
                        FAILED_AMENDMENT
                    ),

                "sha256":
                    EXPECTED_FAILED_AMENDMENT_SHA,

                "preserved":
                    True,

                "authoritative":
                    False,

                "status":
                    "SUPERSEDED_AFTER_TIMESTAMP_ONLY_AUDIT",
            },

            "timestamp_structure_diagnostic": {
                "path":
                    str(
                        DIAGNOSTIC
                    ),

                "sha256":
                    diagnostic_sha,

                "performance_metrics_used":
                    False,
            },
        },

        "authoritative_rule": {
            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "candidate_indices":
                "all timestamp indices strictly after anchor",

            "selection":
                (
                    "argmin_i "
                    "|(timestamp[i]-timestamp[anchor]) "
                    "- nominal_horizon|"
                ),

            "tie_break":
                (
                    "smallest/earliest absolute "
                    "future index"
                ),

            "target_bracketing_required":
                True,

            "selected_indices_must_increase_with_horizon":
                True,

            "interpolation":
                False,

            "nearest_sample_error_threshold":
                None,

            "fixed_positional_offset_selector":
                False,

            "nominal_offsets": {
                "0.1":
                    1,

                "0.3":
                    3,

                "0.5":
                    5,

                "1.0":
                    10,
            },

            "nominal_offsets_role":
                "DIAGNOSTIC_ONLY",
        },

        "timestamp_only_validation":
            resolved,

        "scientific_boundary": {
            "actor_future_state_values_used_to_select_rule":
                False,

            "illumination_values_used_to_select_rule":
                False,

            "reactive_performance_used_to_select_rule":
                False,

            "predictive_performance_used_to_select_rule":
                False,

            "P_occ_used_to_select_rule":
                False,

            "NI_delta_used_to_select_rule":
                False,

            "formal_outcome_used_to_select_rule":
                False,

            "controller_or_policy_tuning":
                False,
        },

        "supersedes_only": [
            (
                "original preoutcome fixed "
                "WOMD index-offset interpretation"
            ),
            (
                "failed per-adjacent-step "
                "timestamp amendment"
            ),
        ],

        "unchanged": [
            "nominal horizons",
            "actor population",
            "future truth evaluator-only role",
            "future box geometry",
            "rasterization",
            "reactive t0 action",
            "metric semantics",
            "NI rule",
            "controller",
            "policy",
        ],
    }

    write_once(
        SUPERSEDING,
        superseding,
    )

    alignment_sha = file_sha(
        SUPERSEDING
    )

    print(
        "superseding rule SHA256 =",
        alignment_sha,
    )
    print(
        "alignment selector       = NEAREST TIMESTAMP"
    )
    print(
        "interpolation            = NO"
    )
    print(
        "error threshold          = NONE"
    )

    # --------------------------------------------------------
    # E. Exact runner repair
    # --------------------------------------------------------

    print()
    print(
        "===== E. PATCH REACTIVE-ONLY RUNNER ====="
    )

    original_bytes = RUNNER.read_bytes()

    require(
        file_sha(
            RUNNER
        )
        ==
        EXPECTED_RUNNER_SHA,
        (
            "Runner changed before patch."
        ),
    )

    original_source = original_bytes.decode(
        "utf-8"
    )

    patched_source = patch_runner(
        original_source,
        alignment_sha=alignment_sha,
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
                "Existing superseding-patch "
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
        EXPECTED_RUNNER_SHA,
        (
            "Backup SHA mismatch."
        ),
    )

    print(
        "runner pre-patch SHA =",
        EXPECTED_RUNNER_SHA,
    )
    print(
        "patch scope          = TIMESTAMP SELECTION ONLY"
    )
    print(
        "future box code      = UNCHANGED"
    )
    print(
        "metric code          = UNCHANGED"
    )

    # --------------------------------------------------------
    # F. Transactional write
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

        rc_compile, compile_output = run_command(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(
                    RUNNER
                ),
            ]
        )

        require(
            rc_compile == 0,
            (
                "Patched runner compile failed:\n"
                +
                compile_output
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
                "PATCH FAILURE -> EXACT ROLLBACK"
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
                EXPECTED_RUNNER_SHA,
                (
                    "Runner rollback SHA mismatch."
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
    # G. Final readback
    # --------------------------------------------------------

    print()
    print(
        "===== G. POST-PATCH READBACK ====="
    )

    post_source = RUNNER.read_text(
        encoding="utf-8"
    )

    require(
        "SUPERSEDING_TIMESTAMP_ALIGNMENT"
        in
        post_source,
        (
            "Superseding selector marker "
            "missing after patch."
        ),
    )

    require(
        alignment_sha
        in
        post_source,
        (
            "Superseding alignment SHA "
            "not sealed in runner."
        ),
    )

    require(
        "WOMD horizon timestamp mismatch"
        not in
        post_source,
        (
            "Old cumulative timestamp "
            "gate remains active."
        ),
    )

    require(
        not RAW_VALUES.exists(),
        (
            "Patch unexpectedly generated metrics."
        ),
    )

    require(
        not DELTA_FREEZE.exists(),
        (
            "Patch unexpectedly generated NI deltas."
        ),
    )

    print(
        "nearest timestamp selector = ACTIVE"
    )
    print(
        "fixed direct offset         = DISABLED"
    )
    print(
        "cumulative tolerance gate   = DISABLED"
    )
    print(
        "superseding SHA seal        = PASS"
    )

    # --------------------------------------------------------
    # H. Patch report
    # --------------------------------------------------------

    result = {
        "stage":
            6,

        "block":
            "6.8_timestamp_alignment_superseding_repair",

        "status":
            "PASS_SUPERSEDING_TIMESTAMP_ALIGNMENT_FROZEN",

        "diagnosis": {
            "original_fixed_offsets_universally_valid":
                False,

            "affected_scenarios":
                2,

            "interpolation_required":
                False,

            "timestamp_tolerance_relaxation_required":
                False,

            "nearest_observation_sufficient":
                True,
        },

        "superseding_alignment": {
            "path":
                str(
                    SUPERSEDING
                ),

            "sha256":
                alignment_sha,

            "selection":
                (
                    "nearest observed timestamp "
                    "to each nominal physical horizon"
                ),

            "tie_break":
                "earliest index",

            "interpolation":
                False,

            "error_threshold":
                None,
        },

        "runner": {
            "path":
                str(
                    RUNNER
                ),

            "pre_patch_sha256":
                EXPECTED_RUNNER_SHA,

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
                "timestamp resolution only",
        },

        "scientific_boundary": {
            "future_GT_already_opened":
                True,

            "future_GT_role":
                "evaluator_only",

            "reactive_metric_values_computed":
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
                "rerun same reactive-only evaluator "
                "to compute exactly the three "
                "reactive NI-bound metrics"
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
        "BLOCK 6.8 SUPERSEDING TIMESTAMP ALIGNMENT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "timestamp diagnostic       = 120 / 120 PASS"
    )
    print(
        "fixed offset selector      = SUPERSEDED"
    )
    print(
        "nearest timestamp selector = FROZEN"
    )
    print(
        "tie break                  = EARLIEST INDEX"
    )
    print(
        "interpolation              = NO"
    )
    print(
        "numeric tolerance gate     = NONE"
    )
    print(
        "0.1s selected offsets      = {1:120}"
    )
    print(
        "0.3s selected offsets      = {3:120}"
    )
    print(
        "0.5s selected offsets      = {5:120}"
    )
    print(
        "1.0s selected offsets      = {9:2, 10:118}"
    )
    print(
        "preoutcome contract        = BYTE-UNCHANGED"
    )
    print(
        "failed amendment           = PRESERVED + SUPERSEDED"
    )
    print(
        "future GT                  = EVALUATOR ONLY"
    )
    print(
        "reactive metrics           = NOT COMPUTED"
    )
    print(
        "predictive performance     = NOT COMPUTED"
    )
    print(
        "P_occ content              = NOT OPENED"
    )
    print(
        "NI deltas                  = NOT COMPUTED"
    )
    print(
        "formal evaluation          = NO"
    )
    print(
        "Stage6 regression          =",
        f"{tests} / {tests} PASS",
    )
    print(
        "patched runner SHA256      =",
        post_runner_sha,
    )
    print(
        "superseding rule SHA256    =",
        alignment_sha,
    )
    print(
        "STATUS = "
        "PASS_SUPERSEDING_TIMESTAMP_ALIGNMENT_FROZEN"
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
        "BLOCK 6.8 SUPERSEDING TIMESTAMP ALIGNMENT = BLOCKED"
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
        "reactive metric freeze = ABSENT"
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
        "Do not alter timestamp tolerances manually."
    )
    print(
        "Do not run predictive/formal evaluation."
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
