from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

PREREG = (
    S6 / "configs/"
    "block66_part3b_class_aware_policy_preregistration.json"
)

STAGE1 = (
    S6 / "configs/"
    "stage6_development_stage1_class_gamma_freeze.json"
)

STAGE2 = (
    S6 / "configs/"
    "stage6_development_stage2_class_margin_freeze.json"
)

STAGE3_EXECUTION = (
    S6 / "configs/"
    "stage6_development_stage3_floor_execution_binding.json"
)

POCC_MANIFEST = (
    S6 / "artifacts/block66/"
    "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

PREDICTIVE_MATCHES = (
    S6 / "artifacts/block66/"
    "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

COHORT = (
    S6 / "artifacts/block66/"
    "block66_class_aware_development_cohort_120.jsonl"
)

METRIC_RUNTIME = (
    S6 / "src/iscai_stage6/adb/"
    "metric_semantics.py"
)

OUTPUT = (
    S6 / "configs/"
    "stage6_development_stage3_floor_scoring_semantics.json"
)

REPORT = (
    S6 / "reports/"
    "block68_stage3_floor_scoring_semantics_freeze.json"
)


EXPECTED_SHA = {
    "prereg":
        "5704494a89b4b3c7a3c19b0df8176c550c00795ed17cf2906c343f340461429e",

    "stage1":
        "a55f21589f55a75afcb660a8a7278e802dd9e102f4c75ca7863697359bbb2a87",

    "stage2":
        "b061541f0563181cd07a91107a211b3519c6fdcc2e08243171db5b3bd9393ae8",

    "stage3_execution":
        "28cc6e986d89ac449cce4b5180f3879567e0f6a8bd77a2bd6054fcb262cba091",

    "pocc_manifest":
        "c361ed640b91850831d7e7177d89733fedd3e171f5f3111bcd93698c74a4a300",

    "predictive_matches":
        "d7ce8769deee08286974b0327cdfd55c08ad7ea326cf9d6450a48a82b3e251b6",

    "metric_runtime":
        "bf8358b5a7edfe40ffac2718ca7cd5af6e7b5fdb3ff8a2823caad6d0f40f057a",
}

EXPECTED_TESTS = 224

EXPECTED_CLASS_COUNTS = {
    "TYPE_VEHICLE": 719,
    "TYPE_PEDESTRIAN": 110,
    "TYPE_CYCLIST": 48,
}

EXPECTED_RISK_VECTOR = [
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "1-pedestrian_visibility_proxy",
    "1-cyclist_visibility_proxy",
    "1-road_illumination_retention",
]

EXPECTED_LEXICOGRAPHIC = [
    "minimize maximum element of normalized risk vector",
    "minimize arithmetic mean of normalized risk vector",
    "minimize over_masking_area",
    "minimize false_dimming",
    "canonical floor tuple (f_vehicle,f_pedestrian,f_cyclist)",
]


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


def exact(path: Path, expected: str, label: str) -> str:
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
        for line_number, line in enumerate(
            stream,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc

    return rows


def canonical_bytes(value) -> bytes:
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


def write_once(path: Path, value):
    payload = canonical_bytes(value)

    if path.exists():
        require(
            path.read_bytes() == payload,
            (
                "Existing frozen file differs: "
                f"{path}"
            ),
        )
        return

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(
        payload
    )

    temporary.replace(
        path
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
            f"got {count}."
        ),
    )

    return count


def main():

    print("=" * 72)
    print("STAGE 6 — BLOCK 6.8")
    print("STAGE-3 FLOOR SCORING SEMANTICS FREEZE")
    print("PRE-OUTCOME / NO FLOOR SWEEP / NO FORMAL")
    print("=" * 72)

    # --------------------------------------------------------
    # A. Exact upstream
    # --------------------------------------------------------

    print()
    print("===== A. EXACT PRE-OUTCOME BOUNDARY =====")

    seals = {}

    for key, path in (
        ("prereg", PREREG),
        ("stage1", STAGE1),
        ("stage2", STAGE2),
        ("stage3_execution", STAGE3_EXECUTION),
        ("pocc_manifest", POCC_MANIFEST),
        ("predictive_matches", PREDICTIVE_MATCHES),
        ("metric_runtime", METRIC_RUNTIME),
    ):
        seals[key] = exact(
            path,
            EXPECTED_SHA[key],
            key,
        )

        print(
            f"{key:22s} = EXACT PASS"
        )

    prereg = read_json(
        PREREG
    )

    stage3_execution = read_json(
        STAGE3_EXECUTION
    )

    require(
        stage3_execution.get(
            "status"
        )
        ==
        "FROZEN_STAGE3_FLOOR_EXECUTION_BEFORE_FLOOR_OUTCOMES",
        "Unexpected Stage-3 execution binding status.",
    )

    print("Stage-1 gamma            = IMMUTABLE")
    print("Stage-2 margins          = IMMUTABLE")
    print("Stage-3 floor outcomes   = NOT COMPUTED")
    print("formal evaluation        = NO")

    # --------------------------------------------------------
    # B. Exact prereg Stage-3 semantics
    # --------------------------------------------------------

    print()
    print("===== B. PREREGISTERED STAGE-3 OBJECTIVE =====")

    protocol = (
        prereg[
            "development_selection_protocol"
        ][
            "stage_3_class_floors"
        ]
    )

    require(
        protocol[
            "normalized_risk_vector"
        ]
        ==
        EXPECTED_RISK_VECTOR,
        "Stage-3 risk vector changed.",
    )

    require(
        protocol[
            "lexicographic_selection"
        ]
        ==
        EXPECTED_LEXICOGRAPHIC,
        "Stage-3 lexicographic order changed.",
    )

    require(
        protocol[
            "arbitrary_weighted_sum"
        ]
        is False,
        "Weighted sum became permitted.",
    )

    require(
        protocol[
            "risk_vector_all_components_unit_interval"
        ]
        is True,
        "Risk-vector unit-interval contract changed.",
    )

    require(
        protocol[
            "gamma_and_margin"
        ]
        ==
        "fixed from stages 1 and 2",
        "Stage1/2 immutability contract changed.",
    )

    for index, name in enumerate(
        EXPECTED_RISK_VECTOR,
        start=1,
    ):
        print(
            f"risk[{index}] = {name}"
        )

    print("weighted sum = FORBIDDEN")
    print("observed min-max normalization = FORBIDDEN")

    # --------------------------------------------------------
    # C. Exact Part3B selection population
    # --------------------------------------------------------

    print()
    print("===== C. STAGE-3 PARAMETER-SELECTION POPULATION =====")

    prereg_counts = (
        prereg[
            "development_population"
        ][
            "class_counts"
        ]
    )

    for actor_class, expected in (
        EXPECTED_CLASS_COUNTS.items()
    ):
        require(
            int(
                prereg_counts[
                    actor_class
                ]
            )
            ==
            expected,
            (
                f"Preregistered class count changed "
                f"for {actor_class}."
            ),
        )

    pocc_rows = read_jsonl(
        POCC_MANIFEST
    )

    require(
        len(pocc_rows) == 877,
        (
            "Expected exact predictive population "
            f"N=877; got {len(pocc_rows)}."
        ),
    )

    pocc_counts = Counter(
        str(
            row[
                "object_type"
            ]
        )
        for row in pocc_rows
    )

    require(
        dict(pocc_counts)
        ==
        EXPECTED_CLASS_COUNTS,
        (
            "P_occ class population differs from "
            f"preregistration: {dict(pocc_counts)}"
        ),
    )

    # We intentionally inspect only JSON metadata.
    # No occupancy_counts_path is dereferenced here.
    for row in pocc_rows:
        require(
            "occupancy_counts_path" in row,
            "P_occ manifest row missing payload path.",
        )

    cohort_rows = read_jsonl(
        COHORT
    )

    require(
        len(cohort_rows) == 120,
        (
            "Expected 120 development scenarios; "
            f"got {len(cohort_rows)}."
        ),
    )

    scenario_ids = [
        str(
            row[
                "scenario_id"
            ]
        )
        for row in cohort_rows
    ]

    require(
        len(
            set(
                scenario_ids
            )
        )
        ==
        120,
        "Development scenario IDs are not unique.",
    )

    for row in cohort_rows:
        require(
            row.get(
                "selection_uses_future"
            )
            is False,
            "Development cohort used future information.",
        )

        require(
            row.get(
                "tracks_to_predict_used"
            )
            is False,
            "tracks_to_predict used in cohort selection.",
        )

        require(
            row.get(
                "objects_of_interest_used"
            )
            is False,
            "objects_of_interest used in cohort selection.",
        )

    print("development scenarios     = 120 EXACT")
    print("predictive actors          = 877 EXACT")
    print("  TYPE_VEHICLE             = 719")
    print("  TYPE_PEDESTRIAN          = 110")
    print("  TYPE_CYCLIST             = 48")
    print("future used as selector    = NO")

    # --------------------------------------------------------
    # D. Reactive fallback role
    # --------------------------------------------------------

    print()
    print("===== D. REACTIVE-FALLBACK FLOOR ROLE =====")

    fallback_contract = (
        prereg[
            "reactive_fallback_inside_class_aware_controller"
        ]
    )

    require(
        fallback_contract[
            "class_floor_applied"
        ]
        is True,
        (
            "Preregistered downstream fallback floor "
            "application changed."
        ),
    )

    print(
        "Stage-3 selection population = "
        "PREDICTIVE DEVELOPMENT ACTORS ONLY"
    )

    print(
        "reactive fallback in floor objective = NO"
    )

    print(
        "selected class floor applied to fallback downstream = YES"
    )

    print(
        "reason = Part3B development_population defines "
        "the staged predictive policy-selection population; "
        "fallback is a separate integrated-controller branch"
    )

    # --------------------------------------------------------
    # E. Freeze aggregation semantics
    # --------------------------------------------------------

    print()
    print("===== E. PRE-OUTCOME METRIC AGGREGATION FREEZE =====")

    aggregation = {
        "evaluation_unit":
            "development scenario",

        "controller_schedule":
            (
                "one raw class-aware illumination schedule "
                "of shape [4,501,301] per scenario and "
                "candidate floor triplet"
            ),

        "terminal_illumination":
            "ClassAwareComposition.raw_class_aware_illumination",

        "temporal_smoothing":
            False,

        "actuation_rate_limit":
            False,

        "metric_evaluation":
            (
                "call the authoritative frozen metric semantics "
                "on the complete four-horizon scenario schedule; "
                "do not manually reweight horizons"
            ),

        "scenario_aggregation":
            (
                "arithmetic macro mean over finite "
                "scenario-level metric values"
            ),

        "scenario_weighting":
            "equal scenario weight",

        "actor_count_weighting":
            False,

        "horizon_reweighting":
            False,

        "future_validity_as_selection_filter":
            False,

        "empty_support":
            (
                "inherit authoritative metric runtime exactly; "
                "if it returns NaN/NA, exclude that scenario "
                "only from that metric aggregate; do not impute"
            ),

        "normalization":
            (
                "no outcome-dependent normalization; all five "
                "risk components are already frozen in [0,1]"
            ),

        "risk_vector_after_scenario_macro": [
            "vehicle_shadow_zone_violation",
            "glare_risk_exposure",
            "1-pedestrian_visibility_proxy",
            "1-cyclist_visibility_proxy",
            "1-road_illumination_retention",
        ],

        "secondary_metrics_after_scenario_macro": [
            "over_masking_area",
            "false_dimming",
        ],
    }

    for key, value in (
        aggregation.items()
    ):
        print(
            f"{key} = {value}"
        )

    # --------------------------------------------------------
    # F. Freeze exact selection comparator
    # --------------------------------------------------------

    print()
    print("===== F. EXACT LEXICOGRAPHIC COMPARATOR =====")

    comparator = {
        "candidate_key":
            (
                "("
                "max(risk_vector), "
                "mean(risk_vector), "
                "over_masking_area, "
                "false_dimming, "
                "j_vehicle, "
                "j_pedestrian, "
                "j_cyclist"
                ")"
            ),

        "comparison":
            "ascending Python/IEEE-754 float64 lexicographic order",

        "epsilon_tolerance":
            None,

        "rounding_before_comparison":
            False,

        "post_outcome_tie_discretion":
            False,

        "canonical_integer_tuple_final_tie_break":
            True,
    }

    print(
        "selection key =",
        comparator[
            "candidate_key"
        ],
    )

    print("epsilon tolerance = NONE")
    print("rounding           = NONE")
    print("posthoc tie choice = FORBIDDEN")

    # --------------------------------------------------------
    # G. Write-once freeze
    # --------------------------------------------------------

    print()
    print("===== G. WRITE-ONCE SCORING SEMANTICS FREEZE =====")

    frozen = {
        "stage":
            6,

        "block":
            "6.8_stage3_floor_scoring_semantics",

        "status":
            (
                "FROZEN_STAGE3_SCORING_SEMANTICS_"
                "BEFORE_FLOOR_OUTCOMES"
            ),

        "authority": {
            "preregistered_candidate_space_and_objective":
                str(
                    PREREG
                ),

            "preoutcome_execution_binding":
                str(
                    STAGE3_EXECUTION
                ),

            "metric_runtime":
                str(
                    METRIC_RUNTIME
                ),
        },

        "selection_population": {
            "development_scenarios":
                120,

            "predictive_actors":
                877,

            "class_counts":
                EXPECTED_CLASS_COUNTS,

            "reactive_fallback_actors_used_for_stage3_selection":
                False,

            "reactive_fallback_receives_selected_floor_downstream":
                True,

            "future_validity_used_as_selector":
                False,
        },

        "aggregation":
            aggregation,

        "selection_comparator":
            comparator,

        "frozen_risk_vector":
            EXPECTED_RISK_VECTOR,

        "frozen_lexicographic_order":
            EXPECTED_LEXICOGRAPHIC,

        "candidate_floor_triplets":
            3270,

        "scientific_boundary": {
            "P_occ_array_payloads_dereferenced":
                False,

            "future_truth_opened_by_this_run":
                False,

            "Stage3_floor_metric_values_computed":
                False,

            "Stage3_floor_sweep_executed":
                False,

            "Stage3_floor_tuple_selected":
                False,

            "Stage1_modified":
                False,

            "Stage2_modified":
                False,

            "Stage4_temporal_or_rate_selected":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,
        },

        "seals":
            seals,

        "cohort": {
            "path":
                str(
                    COHORT
                ),

            "sha256":
                file_sha(
                    COHORT
                ),

            "scenario_count":
                120,
        },
    }

    write_once(
        OUTPUT,
        frozen,
    )

    output_sha = file_sha(
        OUTPUT
    )

    print(
        "scoring freeze SHA256 =",
        output_sha,
    )

    # --------------------------------------------------------
    # H. Regression
    # --------------------------------------------------------

    print()
    print("===== H. POST-FREEZE REGRESSION =====")

    tests = run_regression()

    print(
        "Stage6 regression =",
        f"{tests} / {tests} PASS",
    )

    report = {
        "stage":
            6,

        "block":
            "6.8_stage3_floor_scoring_semantics",

        "status":
            "PASS_STAGE3_SCORING_SEMANTICS_FROZEN",

        "scoring_freeze": {
            "path":
                str(
                    OUTPUT
                ),

            "sha256":
                output_sha,
        },

        "selection_population": {
            "scenarios":
                120,

            "predictive_actors":
                877,

            "class_counts":
                EXPECTED_CLASS_COUNTS,
        },

        "aggregation":
            (
                "equal-weight macro mean of finite "
                "scenario-level authoritative metric values"
            ),

        "candidate_floor_triplets":
            3270,

        "Stage6_regression":
            tests,

        "next":
            (
                "execute the exact 3270-triplet "
                "development-only Stage-3 floor sweep "
                "under this immutable scoring contract"
            ),
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BLOCK 6.8 STAGE-3 SCORING SEMANTICS — FINAL")
    print("=" * 72)

    print("Stage-1 gamma             = IMMUTABLE")
    print("Stage-2 margins           = IMMUTABLE")
    print("selection scenarios       = 120")
    print("selection actors          = 877")
    print("reactive fallback scoring = EXCLUDED")
    print("fallback floor downstream = YES")
    print("aggregation               = SCENARIO MACRO")
    print("actor weighting           = NO")
    print("horizon reweighting       = NO")
    print("outcome normalization     = NO")
    print("epsilon tie tolerance     = NONE")
    print("candidate triplets        = 3270")
    print("Stage-3 outcomes          = NOT COMPUTED")
    print("Stage-4 temporal/rate     = NOT SELECTED")
    print("primary acceptance        = NOT TESTED")
    print("formal evaluation         = NO")

    print(
        "Stage6 regression         =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_STAGE3_SCORING_SEMANTICS_FROZEN"
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
    print("=" * 72)
    print(
        "BLOCK 6.8 STAGE-3 SCORING SEMANTICS = BLOCKED"
    )
    print("=" * 72)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print("Stage-1 gamma           = STILL FROZEN")
    print("Stage-2 margins         = STILL FROZEN")
    print("Stage-3 floor sweep     = NOT EXECUTED")
    print("Stage-3 floor selected  = NO")
    print("Stage-4 temporal/rate   = NOT SELECTED")
    print("primary acceptance      = NOT TESTED")
    print("formal evaluation       = NO")

    print()
    print(
        "Do not execute the 3270-triplet sweep "
        "unless this scoring freeze passes."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().
