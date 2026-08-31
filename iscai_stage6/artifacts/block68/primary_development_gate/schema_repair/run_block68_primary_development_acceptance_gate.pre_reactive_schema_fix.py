from __future__ import annotations

from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
import re
import subprocess
import sys
import traceback


S6 = Path("/home/agni/waymo/iscai_stage6")

# ============================================================
# FROZEN AUTHORITIES
# ============================================================

STAGE4_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage4_temporal_rate_selection_freeze.json"
)

STAGE4_REPORT = (
    S6
    / "reports/"
      "block68_stage4_temporal_rate_selection.json"
)

PREDICTIVE_WINNER = (
    S6
    / "artifacts/block68/stage4_temporal_rate_selection/"
      "block68_stage4_winner_scenario_metrics.jsonl"
)

ACCEPTANCE_POLICY = (
    S6
    / "configs/"
      "stage6_preformal_primary_acceptance_policy.json"
)

OUTPUT = (
    S6
    / "reports/"
      "block68_primary_development_acceptance_gate.json"
)

DETAIL = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "block68_primary_development_acceptance_detail.json"
)


EXPECTED_SHA = {
    "stage4_freeze":
        "780e134a66153c52783bc18d55f607c891b9a6983f8cfc8ac78e2e7367874885",

    "stage4_report":
        "f2c952bb2c4029e5c142e1607c46dc784d195ed4e09f5780ac3d5472219aee7b",

    "predictive_winner":
        "51ddaa6b14187bfb61e8fa338dbbad042b3df56aec043ee55562cf163e7626bc",

    "acceptance_policy":
        "e103466b1a6eb62be4e6746029147cbda9671b05130acba5210a3ab7328e6b81",
}

# Exact frozen reactive raw development values artifact.
REACTIVE_RAW_SHA = (
    "3a81d6d39b29a2d30247f30f06f0e02"
    "bedd47ecc754a0eb550611947f6e9b0d5"
)

EXPECTED_TESTS = 224

# Frozen before predictive outcomes.
OVERMASK_DELTA_MAX = 0.02
PEDESTRIAN_VISIBILITY_DELTA_MIN = -0.05
CYCLIST_VISIBILITY_DELTA_MIN = -0.05

METRICS = (
    "vehicle_shadow_zone_violation",
    "over_masking_area",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
)

EXPECTED_PREDICTIVE = {
    "vehicle_shadow_zone_violation":
        0.013116180708780553,

    "over_masking_area":
        0.26521748861081823,

    "pedestrian_visibility_proxy":
        0.8992267609505494,

    "cyclist_visibility_proxy":
        0.8869782482483739,
}

PARITY_TOL = 1.0e-12


# ============================================================
# HELPERS
# ============================================================

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def exact(path: Path, expected: str, label: str):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = file_sha(path)

    require(
        actual == expected,
        (
            f"{label} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        ),
    )

    return actual


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

        for line_number, raw in enumerate(
            stream,
            start=1,
        ):
            if not raw.strip():
                continue

            try:
                value = json.loads(raw)

            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL {path}:{line_number}"
                ) from exc

            require(
                isinstance(value, dict),
                (
                    "Non-object JSONL row at "
                    f"{path}:{line_number}"
                ),
            )

            rows.append(value)

    return rows


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_once(path: Path, value):
    payload = canonical_bytes(value)

    if path.exists():
        require(
            path.read_bytes() == payload,
            (
                "Existing write-once result differs: "
                f"{path}"
            ),
        )

        return

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)
    temporary.replace(path)


def is_numeric(value):
    return (
        isinstance(value, Real)
        and
        not isinstance(value, bool)
    )


def metric_values_in_row(value, metric):
    """
    Find numeric leaves whose dictionary key is exactly the
    authoritative metric name.

    Duplicate identical representations inside one row are
    tolerated; conflicting duplicates are forbidden.
    """
    found = []

    def recurse(item):
        if isinstance(item, dict):

            for key, child in item.items():

                if (
                    str(key) == metric
                    and
                    is_numeric(child)
                ):
                    number = float(child)

                    if math.isfinite(number):
                        found.append(number)

                recurse(child)

        elif isinstance(item, list):

            for child in item:
                recurse(child)

    recurse(value)

    if not found:
        return None

    unique = []

    for value in found:

        if not any(
            abs(value - previous)
            <=
            1.0e-15
            for previous in unique
        ):
            unique.append(value)

    require(
        len(unique) == 1,
        (
            f"Conflicting {metric} values "
            f"inside one scenario row: {unique}"
        ),
    )

    return unique[0]


def macro_metric(rows, metric):
    values = []

    for row in rows:
        value = metric_values_in_row(
            row,
            metric,
        )

        if value is not None:
            values.append(value)

    require(
        values,
        f"No finite values found for {metric}.",
    )

    return (
        float(
            math.fsum(values)
            /
            len(values)
        ),
        len(values),
    )


def scenario_ids(rows):
    ids = []

    for row in rows:
        value = row.get(
            "scenario_id"
        )

        if value is not None:
            ids.append(str(value))

    return ids


def locate_reactive_raw():
    """
    Locate the immutable reactive metric JSONL by SHA,
    never by guessed filename.
    """
    matches = []

    roots = (
        S6 / "artifacts/block68",
        S6 / "artifacts/block66",
    )

    for root in roots:

        if not root.is_dir():
            continue

        for path in root.rglob("*.jsonl"):

            # Formal data must not be read.
            lowered = str(path).lower().replace(
                "preformal",
                "",
            )

            if "formal" in lowered:
                continue

            try:
                digest = file_sha(path)

            except Exception:
                continue

            if digest == REACTIVE_RAW_SHA:
                matches.append(path)

    require(
        len(matches) == 1,
        (
            "Expected exactly one frozen reactive "
            "raw-values JSONL matching SHA "
            f"{REACTIVE_RAW_SHA}; found {matches}"
        ),
    )

    return matches[0]


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
        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

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
            f"observed {count}."
        ),
    )

    return count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("STAGE 6 — BLOCK 6.8")
    print("FROZEN PRIMARY DEVELOPMENT ACCEPTANCE GATE")
    print("NO TUNING / NO FORMAL / SCIENTIFIC PASS OR FAIL ONLY")
    print("=" * 78)

    # --------------------------------------------------------
    # A. Exact immutable policy boundary
    # --------------------------------------------------------

    print()
    print("===== A. EXACT IMMUTABLE BOUNDARY =====")

    seals = {}

    for key, path in (
        ("stage4_freeze", STAGE4_FREEZE),
        ("stage4_report", STAGE4_REPORT),
        ("predictive_winner", PREDICTIVE_WINNER),
        ("acceptance_policy", ACCEPTANCE_POLICY),
    ):

        seals[key] = exact(
            path,
            EXPECTED_SHA[key],
            key,
        )

        print(
            f"{key:22s} = EXACT PASS"
        )

    stage4 = read_json(
        STAGE4_FREEZE
    )

    require(
        stage4.get("status")
        ==
        (
            "FROZEN_STAGE4_TEMPORAL_RATE_"
            "BEFORE_PRIMARY_ACCEPTANCE_GATE"
        ),
        "Unexpected Stage-4 freeze status.",
    )

    require(
        stage4[
            "scientific_boundary"
        ][
            "primary_acceptance_tested"
        ]
        is False,
        "Primary acceptance was already tested.",
    )

    require(
        stage4[
            "scientific_boundary"
        ][
            "formal_evaluation"
        ]
        is False,
        "Formal evaluation unexpectedly opened.",
    )

    print(
        "Stage-1 gamma       = IMMUTABLE"
    )

    print(
        "Stage-2 margins     = IMMUTABLE"
    )

    print(
        "Stage-3 floors      = IMMUTABLE"
    )

    print(
        "Stage-4 temporal/rate = IMMUTABLE"
    )

    print(
        "post-outcome tuning = FORBIDDEN"
    )

    print(
        "formal evaluation   = NO"
    )

    # --------------------------------------------------------
    # B. Frozen acceptance rule
    # --------------------------------------------------------

    print()
    print("===== B. FROZEN PRIMARY ACCEPTANCE RULE =====")

    policy_text = ACCEPTANCE_POLICY.read_text(
        encoding="utf-8"
    )

    # The exact policy file is already SHA-sealed.
    # These assertions ensure this runner does not silently
    # apply different numeric tolerances.
    require(
        "0.02" in policy_text,
        "Frozen 0.02 overmask bound not present.",
    )

    require(
        "0.05" in policy_text,
        "Frozen 0.05 VRU bound not present.",
    )

    print(
        "vehicle criterion = predictive < reactive STRICT"
    )

    print(
        "overmask criterion = predictive - reactive <= 0.02"
    )

    print(
        "ped visibility criterion = predictive - reactive >= -0.05"
    )

    print(
        "cyclist visibility criterion = predictive - reactive >= -0.05"
    )

    print(
        "all four criteria required = YES"
    )

    # --------------------------------------------------------
    # C. Locate exact reactive raw comparator
    # --------------------------------------------------------

    print()
    print("===== C. FROZEN REACTIVE COMPARATOR =====")

    reactive_path = locate_reactive_raw()

    require(
        file_sha(reactive_path)
        ==
        REACTIVE_RAW_SHA,
        "Reactive raw SHA changed.",
    )

    reactive_rows = read_jsonl(
        reactive_path
    )

    require(
        len(reactive_rows) == 120,
        (
            "Expected 120 reactive development rows; "
            f"observed {len(reactive_rows)}."
        ),
    )

    print(
        "reactive raw artifact =",
        reactive_path,
    )

    print(
        "reactive raw SHA      =",
        REACTIVE_RAW_SHA,
    )

    print(
        "reactive scenarios    = 120 / 120"
    )

    # --------------------------------------------------------
    # D. Predictive winner readback
    # --------------------------------------------------------

    print()
    print("===== D. FROZEN PREDICTIVE WINNER =====")

    predictive_rows = read_jsonl(
        PREDICTIVE_WINNER
    )

    require(
        len(predictive_rows) == 120,
        (
            "Expected 120 predictive winner rows; "
            f"observed {len(predictive_rows)}."
        ),
    )

    reactive_ids = scenario_ids(
        reactive_rows
    )

    predictive_ids = scenario_ids(
        predictive_rows
    )

    if (
        len(reactive_ids) == 120
        and
        len(predictive_ids) == 120
    ):

        require(
            set(reactive_ids)
            ==
            set(predictive_ids),
            (
                "Reactive/predictive scenario populations "
                "are not identical."
            ),
        )

        print(
            "scenario identity join = 120 / 120 EXACT"
        )

    else:

        print(
            "scenario identity join = unavailable in one raw schema"
        )

        print(
            "population cardinality = 120 / 120 EXACT"
        )

    # --------------------------------------------------------
    # E. Exact macro recomputation
    # --------------------------------------------------------

    print()
    print("===== E. EXACT MACRO RECOMPUTATION =====")

    reactive = {}
    predictive = {}
    support = {}

    for metric in METRICS:

        reactive_mean, reactive_n = macro_metric(
            reactive_rows,
            metric,
        )

        predictive_mean, predictive_n = macro_metric(
            predictive_rows,
            metric,
        )

        reactive[metric] = reactive_mean
        predictive[metric] = predictive_mean

        support[metric] = {
            "reactive_n":
                reactive_n,

            "predictive_n":
                predictive_n,
        }

        expected_predictive = EXPECTED_PREDICTIVE[
            metric
        ]

        delta = abs(
            predictive_mean
            -
            expected_predictive
        )

        require(
            delta <= PARITY_TOL,
            (
                "Predictive winner replay mismatch\n"
                f"metric={metric}\n"
                f"computed={predictive_mean!r}\n"
                f"frozen={expected_predictive!r}\n"
                f"delta={delta!r}"
            ),
        )

        print()
        print(metric)
        print(
            "  reactive mean  =",
            repr(
                reactive_mean
            ),
        )

        print(
            "  predictive mean=",
            repr(
                predictive_mean
            ),
        )

        print(
            "  delta          =",
            repr(
                predictive_mean
                -
                reactive_mean
            ),
        )

        print(
            "  support R/P    =",
            reactive_n,
            "/",
            predictive_n,
        )

    # --------------------------------------------------------
    # F. Apply immutable primary gate
    # --------------------------------------------------------

    print()
    print("===== F. FROZEN PRIMARY DEVELOPMENT GATE =====")

    vehicle_delta = (
        predictive[
            "vehicle_shadow_zone_violation"
        ]
        -
        reactive[
            "vehicle_shadow_zone_violation"
        ]
    )

    overmask_delta = (
        predictive[
            "over_masking_area"
        ]
        -
        reactive[
            "over_masking_area"
        ]
    )

    pedestrian_delta = (
        predictive[
            "pedestrian_visibility_proxy"
        ]
        -
        reactive[
            "pedestrian_visibility_proxy"
        ]
    )

    cyclist_delta = (
        predictive[
            "cyclist_visibility_proxy"
        ]
        -
        reactive[
            "cyclist_visibility_proxy"
        ]
    )

    vehicle_pass = (
        predictive[
            "vehicle_shadow_zone_violation"
        ]
        <
        reactive[
            "vehicle_shadow_zone_violation"
        ]
    )

    overmask_pass = (
        overmask_delta
        <=
        OVERMASK_DELTA_MAX
    )

    pedestrian_pass = (
        pedestrian_delta
        >=
        PEDESTRIAN_VISIBILITY_DELTA_MIN
    )

    cyclist_pass = (
        cyclist_delta
        >=
        CYCLIST_VISIBILITY_DELTA_MIN
    )

    gates = {
        "vehicle_strict_improvement": {
            "pass":
                bool(
                    vehicle_pass
                ),

            "reactive":
                reactive[
                    "vehicle_shadow_zone_violation"
                ],

            "predictive":
                predictive[
                    "vehicle_shadow_zone_violation"
                ],

            "delta_predictive_minus_reactive":
                vehicle_delta,

            "rule":
                "predictive < reactive",
        },

        "overmask_noninferiority": {
            "pass":
                bool(
                    overmask_pass
                ),

            "reactive":
                reactive[
                    "over_masking_area"
                ],

            "predictive":
                predictive[
                    "over_masking_area"
                ],

            "delta_predictive_minus_reactive":
                overmask_delta,

            "maximum_allowed_delta":
                OVERMASK_DELTA_MAX,

            "rule":
                "predictive - reactive <= 0.02",
        },

        "pedestrian_visibility_noninferiority": {
            "pass":
                bool(
                    pedestrian_pass
                ),

            "reactive":
                reactive[
                    "pedestrian_visibility_proxy"
                ],

            "predictive":
                predictive[
                    "pedestrian_visibility_proxy"
                ],

            "delta_predictive_minus_reactive":
                pedestrian_delta,

            "minimum_allowed_delta":
                PEDESTRIAN_VISIBILITY_DELTA_MIN,

            "rule":
                "predictive - reactive >= -0.05",
        },

        "cyclist_visibility_noninferiority": {
            "pass":
                bool(
                    cyclist_pass
                ),

            "reactive":
                reactive[
                    "cyclist_visibility_proxy"
                ],

            "predictive":
                predictive[
                    "cyclist_visibility_proxy"
                ],

            "delta_predictive_minus_reactive":
                cyclist_delta,

            "minimum_allowed_delta":
                CYCLIST_VISIBILITY_DELTA_MIN,

            "rule":
                "predictive - reactive >= -0.05",
        },
    }

    overall_pass = all(
        item["pass"]
        for item in gates.values()
    )

    for name, item in gates.items():

        print()
        print(
            name,
            "=",
            "PASS"
            if item["pass"]
            else "FAIL",
        )

        print(
            "  reactive   =",
            repr(
                item["reactive"]
            ),
        )

        print(
            "  predictive =",
            repr(
                item["predictive"]
            ),
        )

        print(
            "  delta      =",
            repr(
                item[
                    "delta_predictive_minus_reactive"
                ]
            ),
        )

        print(
            "  rule       =",
            item["rule"],
        )

    print()
    print(
        "OVERALL PRIMARY DEVELOPMENT ACCEPTANCE =",
        (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    )

    # --------------------------------------------------------
    # G. Regression
    # --------------------------------------------------------

    print()
    print("===== G. POST-GATE REGRESSION =====")

    tests = run_regression()

    print(
        "Stage6 regression =",
        f"{tests} / {tests} PASS",
    )

    # --------------------------------------------------------
    # H. Freeze scientific result
    # --------------------------------------------------------

    print()
    print("===== H. WRITE-ONCE PRIMARY GATE RESULT =====")

    status = (
        "PASS_PRIMARY_DEVELOPMENT_ACCEPTANCE_FROZEN"
        if overall_pass
        else
        "FAIL_PRIMARY_DEVELOPMENT_ACCEPTANCE_FROZEN"
    )

    result = {
        "stage":
            6,

        "block":
            "6.8_primary_development_acceptance",

        "status":
            status,

        "overall_pass":
            bool(
                overall_pass
            ),

        "reactive_comparator": {
            "artifact":
                str(
                    reactive_path
                ),

            "sha256":
                REACTIVE_RAW_SHA,

            "scenario_rows":
                120,
        },

        "predictive_policy": {
            "Stage1_gamma":
                "IMMUTABLE",

            "Stage2_margins":
                "IMMUTABLE",

            "Stage3_floors":
                "IMMUTABLE",

            "Stage4_temporal_rate":
                {
                    "time_constant_ms":
                        200,

                    "rho_dim_per_s":
                        1,

                    "rho_bright_per_s":
                        1,
                },

            "stage4_freeze_sha256":
                EXPECTED_SHA[
                    "stage4_freeze"
                ],
        },

        "metrics": {
            "reactive":
                reactive,

            "predictive":
                predictive,

            "support":
                support,
        },

        "gates":
            gates,

        "scientific_boundary": {
            "post_outcome_retuning":
                False,

            "Stage1_modified":
                False,

            "Stage2_modified":
                False,

            "Stage3_modified":
                False,

            "Stage4_modified":
                False,

            "NI_bounds_modified":
                False,

            "primary_acceptance_tested":
                True,

            "formal_evaluation":
                False,

            "formal_evaluation_allowed":
                bool(
                    overall_pass
                ),
        },

        "required_action": (
            "PROCEED_WITH_FROZEN_DEVELOPMENT_POLICY"
            if overall_pass
            else
            "STOP_NO_RETUNING_REPORT_DEVELOPMENT_ACCEPTANCE_FAILURE"
        ),

        "Stage6_regression":
            tests,

        "upstream_seals":
            seals,
    }

    write_once(
        DETAIL,
        result,
    )

    detail_sha = file_sha(
        DETAIL
    )

    report = {
        "stage":
            6,

        "block":
            "6.8_primary_development_acceptance",

        "status":
            status,

        "overall_pass":
            bool(
                overall_pass
            ),

        "gates":
            {
                key:
                    bool(
                        value[
                            "pass"
                        ]
                    )
                for key, value in
                gates.items()
            },

        "detail": {
            "path":
                str(
                    DETAIL
                ),

            "sha256":
                detail_sha,
        },

        "Stage6_regression":
            tests,

        "formal_evaluation":
            False,

        "post_outcome_retuning":
            False,

        "next": (
            "COMPLETE_DEVELOPMENT_EVALUATION_WITH_FROZEN_POLICY"
            if overall_pass
            else
            "DO_NOT_TUNE_DO_NOT_START_FORMAL_REPORT_GATE_FAILURE"
        ),
    }

    write_once(
        OUTPUT,
        report,
    )

    report_sha = file_sha(
        OUTPUT
    )

    # --------------------------------------------------------
    # I. Final
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("BLOCK 6.8 PRIMARY DEVELOPMENT ACCEPTANCE — FINAL")
    print("=" * 78)

    print(
        "vehicle strict improvement =",
        "PASS"
        if vehicle_pass
        else "FAIL",
    )

    print(
        "overmask <= +0.02          =",
        "PASS"
        if overmask_pass
        else "FAIL",
    )

    print(
        "ped visibility >= -0.05    =",
        "PASS"
        if pedestrian_pass
        else "FAIL",
    )

    print(
        "cyclist visibility >= -0.05=",
        "PASS"
        if cyclist_pass
        else "FAIL",
    )

    print(
        "overall primary gate       =",
        "PASS"
        if overall_pass
        else "FAIL",
    )

    print(
        "Stage1/2/3/4 policy        = IMMUTABLE"
    )

    print(
        "post-outcome retuning      = FORBIDDEN"
    )

    print(
        "NI bounds                  = IMMUTABLE"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "Stage6 regression          =",
        f"{tests} / {tests} PASS",
    )

    print(
        "detail SHA                 =",
        detail_sha,
    )

    print(
        "report SHA                 =",
        report_sha,
    )

    print(
        "STATUS =",
        status,
    )

    if overall_pass:

        print(
            "NEXT = FROZEN DEVELOPMENT COMPLETION EVALUATION"
        )

    else:

        print(
            "NEXT = STOP / NO RETUNING / "
            "REPORT DEVELOPMENT ACCEPTANCE FAILURE"
        )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print("=" * 78)
    print(
        "BLOCK 6.8 PRIMARY DEVELOPMENT ACCEPTANCE = BLOCKED"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "Stage1/2/3/4 policy   = STILL IMMUTABLE"
    )

    print(
        "post-outcome retuning = FORBIDDEN"
    )

    print(
        "formal evaluation     = NO"
    )

    print()
    print(
        "A runtime/schema BLOCKED state is not "
        "permission to alter the frozen policy or bounds."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
