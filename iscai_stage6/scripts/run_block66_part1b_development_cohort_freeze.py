from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import traceback
from typing import Any


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

BLOCK65_HANDOFF = (
    S6
    / "artifacts/block65/"
      "block65_to_block66_handoff.json"
)

PART0B_REPORT = (
    S6
    / "reports/"
      "block66_part0b_development_partition_resolution.json"
)

PART1A_REPORT = (
    S6
    / "reports/"
      "block66_part1a_forward_headlamp_census.json"
)

FULL_CENSUS = (
    S6
    / "artifacts/block66/"
      "block66_part1a_headlamp_census.partial.jsonl"
)

ELIGIBLE = (
    S6
    / "artifacts/block66/"
      "block66_part1a_headlamp_relevant_scenarios.jsonl"
)

PREREG = (
    S6
    / "configs/"
      "block66_development_cohort_selection_preregistration.json"
)

COHORT = (
    S6
    / "artifacts/block66/"
      "block66_class_aware_development_cohort_120.jsonl"
)

REPORT = (
    S6
    / "reports/"
      "block66_part1b_development_cohort_freeze.json"
)

FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_development_cohort_freeze_manifest.json"
)

HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part1b_to_part2_handoff.json"
)


EXPECTED = {
    "block65_handoff":
        (
            "124aaedfb39867109a5ddbf5dc7ef243"
            "188df798e5ac19b4b289418ec5ffe700"
        ),

    "part0b_report":
        (
            "1e7860b9589d7ce62791c5c48802fb06"
            "f90e2ccfca5d90b8ec6b528e5e3bb032"
        ),

    "part1a_report":
        (
            "553188a6d01a890e9a6c2f11e9bbfd2"
            "d8426665065c4abcc770fa7e30ca84228"
        ),

    "full_census":
        (
            "a8b6a1f084583552434af9e00c762e36"
            "3c09f3912b4720573acb9cd4ce34e4cb"
        ),

    "eligible":
        (
            "b34340afd615fb92b2cf40bb2795b555"
            "c1acaa25b5778d8af7dd5bca8efd4efa"
        ),
}


STRATA = (
    "cyclist",
    "pedestrian_no_cyclist",
    "vehicle_only",
)

PER_STRATUM = 40
TOTAL = 120

CORE_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)


class ControlledBlock(
    RuntimeError
):
    pass


def require(
    condition: Any,
    message: str,
) -> None:
    if not bool(condition):
        raise ControlledBlock(
            message
        )


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_bytes(
    payload: Any,
) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def canonical_line(
    payload: Any,
) -> str:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        +
        "\n"
    )


def write_json_exact_or_create(
    path: Path,
    payload: Any,
) -> str:
    data = canonical_json_bytes(
        payload
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing JSON differs from "
                f"deterministic rerun: {path}"
            ),
        )

    else:
        path.write_bytes(
            data
        )

    return sha256_file(
        path
    )


def write_jsonl_exact_or_create(
    path: Path,
    rows,
) -> str:
    data = "".join(
        canonical_line(
            row
        )
        for row in rows
    ).encode(
        "utf-8"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing JSONL differs from "
                f"deterministic rerun: {path}"
            ),
        )

    else:
        path.write_bytes(
            data
        )

    return sha256_file(
        path
    )


def load_jsonl(
    path: Path,
):
    return [
        json.loads(
            line
        )
        for line in
        path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def scenario_order_key(
    row,
):
    return (
        str(
            row[
                "selection_hash"
            ]
        ),
        str(
            row[
                "scenario_id"
            ]
        ),
    )


def select_cohort(
    rows,
):
    selected = []

    for stratum in STRATA:
        candidates = [
            row
            for row in rows
            if (
                str(
                    row[
                        "headlamp_stratum"
                    ]
                )
                ==
                stratum
            )
        ]

        require(
            len(
                candidates
            )
            >=
            PER_STRATUM,
            (
                f"Insufficient {stratum} "
                f"support: {len(candidates)}"
            ),
        )

        candidates = sorted(
            candidates,
            key=scenario_order_key,
        )

        for stratum_rank, source in enumerate(
            candidates[
                :PER_STRATUM
            ]
        ):
            item = dict(
                source
            )

            item[
                "cohort_stratum"
            ] = stratum

            item[
                "cohort_stratum_rank"
            ] = int(
                stratum_rank
            )

            selected.append(
                item
            )

    # Canonical cohort order:
    # cyclist 0..39,
    # pedestrian_no_cyclist 0..39,
    # vehicle_only 0..39.
    for cohort_rank, row in enumerate(
        selected
    ):
        row[
            "cohort_rank"
        ] = int(
            cohort_rank
        )

    return selected


def run_regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            S6
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    return {
        "returncode":
            int(
                process.returncode
            ),

        "tests":
            (
                int(
                    match.group(1)
                )
                if match
                else None
            ),

        "tail":
            "\n".join(
                process.stdout
                .splitlines()[
                    -30:
                ]
            ),
    }


try:
    print("=" * 72)
    print(
        "BLOCK 6.6 PART1B — DETERMINISTIC "
        "CLASS-COVERED DEVELOPMENT COHORT FREEZE"
    )
    print("=" * 72)

    # ========================================================
    # A. Immutable Part1A evidence
    # ========================================================

    print()
    print(
        "===== A. IMMUTABLE HEADLAMP-CENSUS SEAL ====="
    )

    for path, expected, label in (
        (
            BLOCK65_HANDOFF,
            EXPECTED[
                "block65_handoff"
            ],
            "Block6.5 handoff",
        ),
        (
            PART0B_REPORT,
            EXPECTED[
                "part0b_report"
            ],
            "Part0B report",
        ),
        (
            PART1A_REPORT,
            EXPECTED[
                "part1a_report"
            ],
            "Part1A report",
        ),
        (
            FULL_CENSUS,
            EXPECTED[
                "full_census"
            ],
            "full headlamp census",
        ),
        (
            ELIGIBLE,
            EXPECTED[
                "eligible"
            ],
            "eligible development manifest",
        ),
    ):
        require(
            path.is_file(),
            f"Missing {label}: {path}",
        )

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            expected,
            (
                f"{label} SHA changed: "
                f"{actual}"
            ),
        )

        print(
            f"{label:30s} = EXACT PASS"
        )

    census_report = json.loads(
        PART1A_REPORT.read_text(
            encoding="utf-8"
        )
    )

    require(
        census_report[
            "status"
        ]
        ==
        (
            "PASS_CAUSAL_FORWARD_HEADLAMP_"
            "DEVELOPMENT_CENSUS"
        ),
        (
            "Part1A status changed."
        ),
    )

    strata_counts = (
        census_report[
            "census"
        ][
            "mutually_exclusive_strata"
        ]
    )

    print(
        "available strata =",
        strata_counts,
    )

    for stratum in STRATA:
        require(
            int(
                strata_counts[
                    stratum
                ]
            )
            >=
            PER_STRATUM,
            (
                "Insufficient development "
                f"support for {stratum}."
            ),
        )

    # ========================================================
    # B. Preregister cohort rule before member selection
    # ========================================================

    print()
    print(
        "===== B. COHORT-SELECTION PREREGISTRATION ====="
    )

    prereg_payload = {
        "stage":
            6,

        "block":
            "6.6-Part1B",

        "status":
            (
                "PREREGISTERED_BEFORE_"
                "COHORT_MEMBER_SELECTION"
            ),

        "source":
            (
                "frozen_Stage4_development_"
                "headlamp_relevant_manifest"
            ),

        "cohort_size":
            TOTAL,

        "strata": {
            "cyclist":
                PER_STRATUM,

            "pedestrian_no_cyclist":
                PER_STRATUM,

            "vehicle_only":
                PER_STRATUM,
        },

        "stratum_definition":
            (
                "mutually_exclusive_current_causal_"
                "headlamp_relevance_priority_"
                "cyclist_then_pedestrian_then_vehicle"
            ),

        "member_ordering_rule":
            (
                "ascending_frozen_selection_hash_"
                "then_scenario_id"
            ),

        "selection_hash_semantics":
            "SHA256(scenario_id)",

        "design_rationale": {
            "reuse_existing_40_40_40_"
            "scenario_stratum_structure":
                True,

            "PDF_literal_requirement":
                False,

            "purpose":
                (
                    "balanced_development_class_"
                    "coverage_before_policy_preregistration"
                ),

            "performance_used":
                False,

            "model_output_used":
                False,

            "future_used":
                False,

            "formal_outcome_used":
                False,
        },

        "important_boundary": {
            "scenario_balanced":
                True,

            "actor_count_balanced":
                False,

            "class_policy_parameter_selection":
                False,

            "class_policy_numeric_freeze":
                False,

            "new_model_forward":
                False,
        },
    }

    prereg_sha = (
        write_json_exact_or_create(
            PREREG,
            prereg_payload,
        )
    )

    print(
        "preregistration =",
        PREREG,
    )

    print(
        "preregistration SHA256 =",
        prereg_sha,
    )

    print(
        "cohort size = 120"
    )

    print(
        "strata = 40 / 40 / 40"
    )

    print(
        "member ordering = "
        "ASCENDING FROZEN SELECTION HASH"
    )

    print(
        "model/policy outcomes used = NO"
    )

    # ========================================================
    # C. Load eligible source and select
    # ========================================================

    print()
    print(
        "===== C. DETERMINISTIC MEMBER SELECTION ====="
    )

    eligible = load_jsonl(
        ELIGIBLE
    )

    require(
        len(
            eligible
        )
        ==
        6922,
        (
            "Eligible count changed: "
            f"{len(eligible)}"
        ),
    )

    require(
        len({
            str(
                row[
                    "scenario_id"
                ]
            )
            for row in eligible
        })
        ==
        len(
            eligible
        ),
        (
            "Eligible manifest has duplicate "
            "scenario IDs."
        ),
    )

    for row in eligible:
        require(
            str(
                row[
                    "headlamp_stratum"
                ]
            )
            in
            STRATA,
            (
                "Eligible manifest contains "
                "unexpected stratum."
            ),
        )

        expected_hash = sha256(
            str(
                row[
                    "scenario_id"
                ]
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        require(
            str(
                row[
                    "selection_hash"
                ]
            )
            ==
            expected_hash,
            (
                "Frozen scenario selection "
                "hash mismatch."
            ),
        )

        require(
            row[
                "selection_uses_full_box"
            ]
            is True,
            (
                "Cohort source lacks full-box "
                "selection provenance."
            ),
        )

        require(
            row[
                "selection_uses_centroid"
            ]
            is False,
            (
                "Centroid selection entered "
                "cohort source."
            ),
        )

        require(
            row[
                "selection_uses_future"
            ]
            is False,
            (
                "Future entered cohort source."
            ),
        )

        require(
            row[
                "tracks_to_predict_used"
            ]
            is False,
            (
                "tracks_to_predict entered "
                "cohort source."
            ),
        )

        require(
            row[
                "objects_of_interest_used"
            ]
            is False,
            (
                "objects_of_interest entered "
                "cohort source."
            ),
        )

    selected = select_cohort(
        eligible
    )

    require(
        len(
            selected
        )
        ==
        TOTAL,
        (
            "Selected cohort size mismatch."
        ),
    )

    require(
        len({
            str(
                row[
                    "scenario_id"
                ]
            )
            for row in selected
        })
        ==
        TOTAL,
        (
            "Selected cohort contains "
            "duplicate scenario IDs."
        ),
    )

    selected_strata = Counter(
        str(
            row[
                "cohort_stratum"
            ]
        )
        for row in selected
    )

    require(
        {
            stratum:
                int(
                    selected_strata[
                        stratum
                    ]
                )
            for stratum in STRATA
        }
        ==
        {
            stratum:
                PER_STRATUM
            for stratum in STRATA
        },
        (
            "Selected stratum counts differ "
            "from preregistration."
        ),
    )

    print(
        "selected scenarios =",
        len(
            selected
        ),
    )

    print(
        "selected strata =",
        dict(
            selected_strata
        ),
    )

    # ========================================================
    # D. Input-order invariance
    # ========================================================

    print()
    print(
        "===== D. INPUT-ORDER INVARIANCE ====="
    )

    shuffled = list(
        eligible
    )

    random.Random(
        20260822
    ).shuffle(
        shuffled
    )

    selected_shuffled = (
        select_cohort(
            shuffled
        )
    )

    canonical_ids = tuple(
        str(
            row[
                "scenario_id"
            ]
        )
        for row in selected
    )

    shuffled_ids = tuple(
        str(
            row[
                "scenario_id"
            ]
        )
        for row in
        selected_shuffled
    )

    require(
        canonical_ids
        ==
        shuffled_ids,
        (
            "Cohort selection depends "
            "on source-row ordering."
        ),
    )

    print(
        "shuffled-source selection = EXACT PASS"
    )

    print(
        "input-order invariant      = YES"
    )

    # ========================================================
    # E. Development class support inside selected cohort
    # ========================================================

    print()
    print(
        "===== E. SELECTED COHORT CLASS SUPPORT ====="
    )

    scenes_with_class = Counter()

    actor_totals = Counter()

    for row in selected:
        counts = row[
            "headlamp_relevant_counts"
        ]

        for actor_class in CORE_CLASSES:
            value = int(
                counts[
                    actor_class
                ]
            )

            actor_totals[
                actor_class
            ] += value

            if value > 0:
                scenes_with_class[
                    actor_class
                ] += 1

    class_scene_support = {
        actor_class:
            int(
                scenes_with_class[
                    actor_class
                ]
            )
        for actor_class in CORE_CLASSES
    }

    class_actor_support = {
        actor_class:
            int(
                actor_totals[
                    actor_class
                ]
            )
        for actor_class in CORE_CLASSES
    }

    print(
        "headlamp scenes/class =",
        class_scene_support,
    )

    print(
        "headlamp actors/class =",
        class_actor_support,
    )

    for actor_class in CORE_CLASSES:
        require(
            class_scene_support[
                actor_class
            ]
            >=
            PER_STRATUM,
            (
                "Selected cohort has fewer "
                f"than 40 scenes containing "
                f"{actor_class}."
            ),
        )

        require(
            class_actor_support[
                actor_class
            ]
            >
            0,
            (
                "Selected cohort has zero "
                f"{actor_class} actors."
            ),
        )

    # Structural stratum checks.
    for row in selected:
        stratum = str(
            row[
                "cohort_stratum"
            ]
        )

        counts = row[
            "headlamp_relevant_counts"
        ]

        if stratum == "cyclist":
            require(
                int(
                    counts[
                        "TYPE_CYCLIST"
                    ]
                )
                >
                0,
                (
                    "Cyclist stratum row lacks "
                    "headlamp-relevant cyclist."
                ),
            )

        elif (
            stratum
            ==
            "pedestrian_no_cyclist"
        ):
            require(
                int(
                    counts[
                        "TYPE_CYCLIST"
                    ]
                )
                ==
                0
                and
                int(
                    counts[
                        "TYPE_PEDESTRIAN"
                    ]
                )
                >
                0,
                (
                    "Pedestrian stratum semantics "
                    "violated."
                ),
            )

        elif stratum == "vehicle_only":
            require(
                int(
                    counts[
                        "TYPE_CYCLIST"
                    ]
                )
                ==
                0
                and
                int(
                    counts[
                        "TYPE_PEDESTRIAN"
                    ]
                )
                ==
                0
                and
                int(
                    counts[
                        "TYPE_VEHICLE"
                    ]
                )
                >
                0,
                (
                    "Vehicle-only stratum "
                    "semantics violated."
                ),
            )

    print(
        "stratum semantic checks = PASS"
    )

    # ========================================================
    # F. Materialize frozen cohort
    # ========================================================

    print()
    print(
        "===== F. MATERIALIZE DEVELOPMENT COHORT ====="
    )

    cohort_sha = (
        write_jsonl_exact_or_create(
            COHORT,
            selected,
        )
    )

    print(
        "cohort =",
        COHORT,
    )

    print(
        "cohort SHA256 =",
        cohort_sha,
    )

    # ========================================================
    # G. Fresh Stage6 regression
    # ========================================================

    print()
    print(
        "===== G. FRESH STAGE6 REGRESSION ====="
    )

    regression = run_regression()

    print(
        regression[
            "tail"
        ]
    )

    require(
        regression[
            "returncode"
        ]
        ==
        0,
        (
            "Stage6 regression failed."
        ),
    )

    require(
        regression[
            "tests"
        ]
        ==
        145,
        (
            "Expected 145 Stage6 tests, got "
            f"{regression['tests']}."
        ),
    )

    print(
        "fresh regression = PASS | 145"
    )

    # ========================================================
    # H. Freeze report
    # ========================================================

    print()
    print(
        "===== H. BLOCK6.6 PART1B FREEZE REPORT ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.6-Part1B",

        "status":
            (
                "PASS_CLASS_COVERED_DEVELOPMENT_"
                "COHORT_FROZEN"
            ),

        "source": {
            "eligible_manifest":
                str(
                    ELIGIBLE
                ),

            "eligible_manifest_sha256":
                EXPECTED[
                    "eligible"
                ],

            "eligible_scenarios":
                6922,

            "formal_overlap":
                0,
        },

        "preregistration": {
            "path":
                str(
                    PREREG
                ),

            "sha256":
                prereg_sha,

            "member_selection_after_preregistration":
                True,
        },

        "cohort": {
            "path":
                str(
                    COHORT
                ),

            "sha256":
                cohort_sha,

            "scenario_count":
                TOTAL,

            "strata":
                {
                    stratum:
                        PER_STRATUM
                    for stratum in STRATA
                },

            "class_scene_support":
                class_scene_support,

            "class_actor_support":
                class_actor_support,

            "selection_rule":
                (
                    "ascending_selection_hash_"
                    "then_scenario_id_within_stratum"
                ),

            "input_order_invariant":
                True,
        },

        "selection_safety": {
            "current_causal_only":
                True,

            "full_box_headlamp_relevance":
                True,

            "centroid_only":
                False,

            "future":
                False,

            "tracks_to_predict":
                False,

            "objects_of_interest":
                False,

            "model_output":
                False,

            "performance":
                False,

            "formal_outcomes":
                False,
        },

        "interpretation": {
            "40_40_40_is_PDF_literal":
                False,

            "40_40_40_role":
                (
                    "implementation_design_continuity_"
                    "and_balanced_scenario_level_"
                    "development_coverage"
                ),

            "actor_counts_balanced":
                False,
        },

        "still_not_frozen": {
            "class_specific_gamma":
                True,

            "vehicle_margin":
                True,

            "pedestrian_margin":
                True,

            "cyclist_margin":
                True,

            "vehicle_dimming_floor":
                True,

            "pedestrian_dimming_floor":
                True,

            "cyclist_dimming_floor":
                True,

            "temporal_smoothing":
                True,

            "actuation_rate_limits":
                True,

            "class_priority":
                True,
        },

        "scientific_execution": {
            "new_model_forward":
                False,

            "new_MC_sampling":
                False,

            "training":
                False,

            "recalibration":
                False,

            "formal_evaluation":
                False,
        },

        "regression": {
            "status":
                "PASS",

            "tests":
                145,
        },

        "next":
            (
                "BLOCK6.6_PART2_FROZEN_MODEL_"
                "POSTERIOR_AND_CAUSAL_ASSOCIATION_"
                "ON_DEVELOPMENT_COHORT"
            ),
    }

    report_sha = (
        write_json_exact_or_create(
            REPORT,
            report_payload,
        )
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    # ========================================================
    # I. Freeze manifest + handoff
    # ========================================================

    print()
    print(
        "===== I. FREEZE MANIFEST + HANDOFF ====="
    )

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.6-Part1B",

        "status":
            (
                "BLOCK66_DEVELOPMENT_COHORT_FROZEN"
            ),

        "immutable_inputs": {
            str(
                PART1A_REPORT
            ):
                EXPECTED[
                    "part1a_report"
                ],

            str(
                FULL_CENSUS
            ):
                EXPECTED[
                    "full_census"
                ],

            str(
                ELIGIBLE
            ):
                EXPECTED[
                    "eligible"
                ],
        },

        "preregistration": {
            "path":
                str(
                    PREREG
                ),

            "sha256":
                prereg_sha,
        },

        "cohort": {
            "path":
                str(
                    COHORT
                ),

            "sha256":
                cohort_sha,

            "N":
                TOTAL,

            "strata":
                {
                    stratum:
                        PER_STRATUM
                    for stratum in STRATA
                },
        },

        "report": {
            "path":
                str(
                    REPORT
                ),

            "sha256":
                report_sha,
        },

        "scientific_lock": {
            "development_cohort_membership":
                "FROZEN",

            "class_policy_numerics":
                "NOT_FROZEN",

            "formal_tuning":
                "FORBIDDEN",
        },
    }

    freeze_sha = (
        write_json_exact_or_create(
            FREEZE,
            freeze_payload,
        )
    )

    handoff_payload = {
        "stage":
            6,

        "from":
            "6.6-Part1B",

        "to":
            "6.6-Part2",

        "status":
            (
                "PASS_READY_FOR_FROZEN_MODEL_"
                "COHORT_POSTERIOR_MATERIALIZATION"
            ),

        "development_cohort": {
            "path":
                str(
                    COHORT
                ),

            "sha256":
                cohort_sha,

            "N":
                TOTAL,

            "strata":
                {
                    stratum:
                        PER_STRATUM
                    for stratum in STRATA
                },

            "class_scene_support":
                class_scene_support,

            "class_actor_support":
                class_actor_support,
        },

        "required_next": [
            (
                "run_frozen_calibrated_Gaussian_"
                "predictor_only_on_cohort"
            ),
            (
                "retain_identity_safe_prediction_id_"
                "per_causal_track_history"
            ),
            (
                "apply_frozen_reciprocal_unique_"
                "nearest_anchor_association"
            ),
            (
                "record_predictive_matches_and_"
                "original_reactive_fallbacks"
            ),
            (
                "materialize_frozen_N8192_full_box_"
                "P_occ_for_matched_ADB_actors"
            ),
        ],

        "forbidden_next": [
            "future_GT_controller_input",
            "tracks_to_predict_actor_selection",
            "objects_of_interest_actor_selection",
            "perfect_track_ID_association",
            "formal_outcome_read",
            "class_policy_parameter_tuning_before_development_evidence",
        ],

        "upstream_freeze": {
            "prereg_sha256":
                prereg_sha,

            "cohort_sha256":
                cohort_sha,

            "report_sha256":
                report_sha,

            "freeze_sha256":
                freeze_sha,
        },
    }

    handoff_sha = (
        write_json_exact_or_create(
            HANDOFF,
            handoff_payload,
        )
    )

    print(
        "freeze manifest =",
        FREEZE,
    )

    print(
        "freeze SHA256 =",
        freeze_sha,
    )

    print(
        "handoff =",
        HANDOFF,
    )

    print(
        "handoff SHA256 =",
        handoff_sha,
    )

    # ========================================================
    # J. Final
    # ========================================================

    print()
    print("=" * 72)
    print(
        "BLOCK 6.6 PART1B — FINAL"
    )
    print("=" * 72)

    print(
        "STATUS = "
        "PASS_BLOCK66_CLASS_COVERED_DEVELOPMENT_COHORT_FROZEN"
    )

    print(
        "development cohort N      = 120"
    )

    print(
        "scenario strata           = "
        "40 cyclist / 40 pedestrian-no-cyclist / 40 vehicle-only"
    )

    print(
        "member ranking            = "
        "ASCENDING SHA256(scenario_id)"
    )

    print(
        "input-order invariant     = YES"
    )

    print(
        "headlamp scenes/class     =",
        class_scene_support,
    )

    print(
        "headlamp actors/class     =",
        class_actor_support,
    )

    print(
        "future selection          = NO"
    )

    print(
        "model/performance selection = NO"
    )

    print(
        "formal outcomes read      = NO"
    )

    print(
        "class policy frozen       = NO"
    )

    print(
        "new model forward         = NO"
    )

    print(
        "new MC sampling           = NO"
    )

    print(
        "fresh Stage6 regression   = PASS | 145"
    )

    print(
        "prereg SHA256             =",
        prereg_sha,
    )

    print(
        "cohort SHA256             =",
        cohort_sha,
    )

    print(
        "report SHA256             =",
        report_sha,
    )

    print(
        "freeze SHA256             =",
        freeze_sha,
    )

    print(
        "handoff SHA256            =",
        handoff_sha,
    )

    print(
        "NEXT = BLOCK6.6 PART2 "
        "FROZEN MODEL POSTERIOR + CAUSAL ASSOCIATION"
    )

    print(
        "terminal remains open = YES"
    )

    print("=" * 72)


except BaseException as exc:
    print()
    print("=" * 72)
    print(
        "BLOCK6.6 PART1B — CONTROLLED BLOCK"
    )
    print("=" * 72)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc(
        limit=18
    )

    print()
    print(
        "development cohort freeze = NO/INCOMPLETE"
    )

    print(
        "class policy frozen       = NO"
    )

    print(
        "model forward             = NO"
    )

    print(
        "formal outcomes read      = NO"
    )

    print(
        "terminal remains open     = YES"
    )

# Deliberately no sys.exit().
