from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage3.evaluation import (
    build_womd_truth_sidecar,
)

from iscai_stage3.validation import (
    build_compact_motion_offset_index,
    read_motion_scenario,
    read_validation_manifest,
    sha256_file,
)


ROOT = Path("/home/agni/waymo")
STAGE3 = ROOT / "iscai_stage3"

PAIRED = (
    ROOT
    / "data/paired_womd_lidar_v1_3_0"
)

BASE_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
      "selected_validation.jsonl"
)

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

CHECKPOINT = (
    STAGE3
    / "artifacts/block38f/"
      "formal_per_scenario.jsonl"
)

REPORT = (
    STAGE3
    / "reports/"
      "block38f_formal_evaluation.json"
)

EXPECTED_FORMAL_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

HORIZONS = (
    0.1,
    0.3,
    0.5,
    1.0,
)

MISS_THRESHOLD_M = 2.0


def canonical_sha(value):
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def strip_runtime(value):
    if isinstance(value, dict):
        return {
            k: strip_runtime(v)
            for k, v in value.items()
            if "runtime" not in k.lower()
        }

    if isinstance(value, list):
        return [
            strip_runtime(v)
            for v in value
        ]

    return value


def percentile95(values):
    if not values:
        return None

    values = sorted(
        float(x)
        for x in values
    )

    return values[
        max(
            0,
            math.ceil(
                0.95 * len(values)
            ) - 1,
        )
    ]


def mean(values):
    if not values:
        return None

    return (
        sum(values)
        /
        len(values)
    )


def rmse(values):
    if not values:
        return None

    return math.sqrt(
        sum(
            x * x
            for x in values
        )
        /
        len(values)
    )


def empty_class():
    return {
        "truth_tracks": 0,
        "matched_tracks": 0,
        "unmatched_truth": 0,

        "ade_planar_errors": [],
        "ade_3d_errors": [],

        "fde_planar_errors": [],
        "fde_3d_errors": [],
        "fde_unavailable_matched_tracks": 0,

        "horizon_errors":
            defaultdict(list),

        "horizon_errors_3d":
            defaultdict(list),

        "horizon_eligible":
            Counter(),

        "horizon_misses":
            Counter(),
    }


def empty_aggregate():
    return {
        "prediction_tracks": 0,
        "truth_tracks": 0,
        "matched_tracks": 0,
        "false_predictions": 0,
        "unmatched_truth": 0,

        "ade_planar_errors": [],
        "ade_3d_errors": [],

        "fde_planar_errors": [],
        "fde_3d_errors": [],
        "fde_unavailable_matched_tracks": 0,

        "anchor_errors": [],

        "horizon_errors":
            defaultdict(list),

        "horizon_errors_3d":
            defaultdict(list),

        "horizon_eligible":
            Counter(),

        "horizon_misses":
            Counter(),

        "runtime_ms": [],

        "classes":
            defaultdict(
                empty_class
            ),
    }


def ingest_report(
    agg,
    report,
    truth_meta,
):
    agg["prediction_tracks"] += (
        report["prediction_track_count"]
    )

    agg["truth_tracks"] += (
        report["truth_track_count"]
    )

    agg["matched_tracks"] += (
        report["matched_track_count"]
    )

    agg["false_predictions"] += (
        report["false_prediction_count"]
    )

    agg["unmatched_truth"] += (
        report["unmatched_truth_count"]
    )

    agg["runtime_ms"].append(
        report["runtime_ms"]
    )

    for horizon in (
        report["horizon_aggregates"]
    ):
        h = float(
            horizon["horizon_s"]
        )

        agg["horizon_eligible"][h] += (
            horizon[
                "eligible_truth_tracks"
            ]
        )

        agg["horizon_misses"][h] += (
            horizon[
                "endpoint_miss_count"
            ]
        )

    track_by_truth = {
        str(x["truth_id"]): x
        for x in report[
            "track_evaluations"
        ]
    }

    matched_truth_ids = {
        str(x["truth_id"])
        for x in report[
            "assignment"
        ]["matches"]
    }

    for track in report[
        "track_evaluations"
    ]:
        actor_class = (
            track["actor_class"]
        )

        c = agg["classes"][
            actor_class
        ]

        fde = track["fde_m"]
        fde3 = track["fde_3d_m"]

        if (
            (fde is None)
            !=
            (fde3 is None)
        ):
            raise RuntimeError(
                "Planar/3D FDE availability "
                "mismatch."
            )

        if fde is None:
            agg[
                "fde_unavailable_matched_tracks"
            ] += 1

            c[
                "fde_unavailable_matched_tracks"
            ] += 1

        else:
            agg[
                "fde_planar_errors"
            ].append(
                float(fde)
            )

            agg[
                "fde_3d_errors"
            ].append(
                float(fde3)
            )

            c[
                "fde_planar_errors"
            ].append(
                float(fde)
            )

            c[
                "fde_3d_errors"
            ].append(
                float(fde3)
            )

        agg["anchor_errors"].append(
            float(
                track[
                    "anchor_assignment_error_m"
                ]
            )
        )

        for point in track[
            "horizon_errors"
        ]:
            h = float(
                point["horizon_s"]
            )

            e = float(
                point[
                    "planar_error_m"
                ]
            )

            e3 = float(
                point[
                    "error_3d_m"
                ]
            )

            agg[
                "ade_planar_errors"
            ].append(e)

            agg[
                "ade_3d_errors"
            ].append(e3)

            agg[
                "horizon_errors"
            ][h].append(e)

            agg[
                "horizon_errors_3d"
            ][h].append(e3)

            c[
                "ade_planar_errors"
            ].append(e)

            c[
                "ade_3d_errors"
            ].append(e3)

            c[
                "horizon_errors"
            ][h].append(e)

            c[
                "horizon_errors_3d"
            ][h].append(e3)

    for truth in truth_meta:
        truth_id = str(
            truth["truth_id"]
        )

        actor_class = (
            truth["actor_class"]
        )

        c = agg["classes"][
            actor_class
        ]

        c["truth_tracks"] += 1

        if truth_id in (
            matched_truth_ids
        ):
            c[
                "matched_tracks"
            ] += 1
        else:
            c[
                "unmatched_truth"
            ] += 1

        evaluation = (
            track_by_truth.get(
                truth_id
            )
        )

        errors = {}

        if evaluation is not None:
            errors = {
                float(p["horizon_s"]):
                    float(
                        p[
                            "planar_error_m"
                        ]
                    )
                for p in evaluation[
                    "horizon_errors"
                ]
            }

        for h in truth[
            "available_horizons_s"
        ]:
            h = float(h)

            c[
                "horizon_eligible"
            ][h] += 1

            error = errors.get(h)

            if (
                error is None
                or error
                >
                MISS_THRESHOLD_M
            ):
                c[
                    "horizon_misses"
                ][h] += 1


def finalize(agg):
    tp = agg["matched_tracks"]
    fp = agg["false_predictions"]
    fn = agg["unmatched_truth"]

    precision = (
        tp / (tp + fp)
        if tp + fp
        else None
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else None
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if (
            precision is not None
            and recall is not None
            and precision + recall
        )
        else None
    )

    horizons = {}

    for h in HORIZONS:
        errors = agg[
            "horizon_errors"
        ][h]

        errors3 = agg[
            "horizon_errors_3d"
        ][h]

        eligible = agg[
            "horizon_eligible"
        ][h]

        misses = agg[
            "horizon_misses"
        ][h]

        horizons[str(h)] = {
            "eligible_truth_tracks":
                eligible,

            "available_predictions":
                len(errors),

            "endpoint_miss_count":
                misses,

            "endpoint_miss_rate":
                (
                    misses / eligible
                    if eligible
                    else None
                ),

            "mean_planar_error_m":
                mean(errors),

            "rmse_planar_error_m":
                rmse(errors),

            "mean_3d_error_m":
                mean(errors3),

            "rmse_3d_error_m":
                rmse(errors3),
        }

    classes = {}

    for actor_class, c in sorted(
        agg["classes"].items()
    ):
        class_horizons = {}

        for h in HORIZONS:
            eligible = (
                c[
                    "horizon_eligible"
                ][h]
            )

            misses = (
                c[
                    "horizon_misses"
                ][h]
            )

            errors = (
                c[
                    "horizon_errors"
                ][h]
            )

            class_horizons[
                str(h)
            ] = {
                "eligible_truth_tracks":
                    eligible,

                "available_predictions":
                    len(errors),

                "endpoint_miss_count":
                    misses,

                "endpoint_miss_rate":
                    (
                        misses / eligible
                        if eligible
                        else None
                    ),

                "mean_planar_error_m":
                    mean(errors),

                "rmse_planar_error_m":
                    rmse(errors),
            }

        classes[
            actor_class
        ] = {
            "truth_tracks":
                c["truth_tracks"],

            "matched_tracks":
                c["matched_tracks"],

            "unmatched_truth":
                c["unmatched_truth"],

            "recall":
                (
                    c["matched_tracks"]
                    /
                    c["truth_tracks"]
                    if c["truth_tracks"]
                    else None
                ),

            "ade_m":
                mean(
                    c[
                        "ade_planar_errors"
                    ]
                ),

            "ade_3d_m":
                mean(
                    c[
                        "ade_3d_errors"
                    ]
                ),

            "fde_m":
                mean(
                    c[
                        "fde_planar_errors"
                    ]
                ),

            "fde_3d_m":
                mean(
                    c[
                        "fde_3d_errors"
                    ]
                ),

            "fde_track_count":
                len(
                    c[
                        "fde_planar_errors"
                    ]
                ),

            "fde_unavailable_matched_track_count":
                c[
                    "fde_unavailable_matched_tracks"
                ],

            "horizons":
                class_horizons,
        }

    return {
        "prediction_track_count":
            agg["prediction_tracks"],

        "truth_track_count":
            agg["truth_tracks"],

        "matched_track_count":
            tp,

        "false_prediction_count":
            fp,

        "unmatched_truth_count":
            fn,

        "reconstruction_precision":
            precision,

        "reconstruction_recall":
            recall,

        "reconstruction_f1":
            f1,

        "ade_m":
            mean(
                agg[
                    "ade_planar_errors"
                ]
            ),

        "ade_3d_m":
            mean(
                agg[
                    "ade_3d_errors"
                ]
            ),

        "ade_point_count":
            len(
                agg[
                    "ade_planar_errors"
                ]
            ),

        "fde_m":
            mean(
                agg[
                    "fde_planar_errors"
                ]
            ),

        "fde_3d_m":
            mean(
                agg[
                    "fde_3d_errors"
                ]
            ),

        "fde_track_count":
            len(
                agg[
                    "fde_planar_errors"
                ]
            ),

        "fde_unavailable_matched_track_count":
            agg[
                "fde_unavailable_matched_tracks"
            ],

        "anchor_assignment_error_mean_m":
            mean(
                agg["anchor_errors"]
            ),

        "horizons":
            horizons,

        "classes":
            classes,

        "runtime_ms": {
            "total":
                sum(
                    agg["runtime_ms"]
                ),

            "median":
                statistics.median(
                    agg["runtime_ms"]
                ),

            "p95":
                percentile95(
                    agg["runtime_ms"]
                ),
        },
    }


def current_valid_indices(
    scenario,
):
    anchor = int(
        scenario.current_time_index
    )

    return tuple(
        i
        for i, track
        in enumerate(
            scenario.tracks
        )
        if (
            i
            != int(
                scenario.sdc_track_index
            )
            and
            track.states[
                anchor
            ].valid
        )
    )


def truth_meta(truth):
    return [
        {
            "truth_id":
                str(track.truth_id),

            "actor_class":
                track.actor_class,

            "available_horizons_s": [
                float(
                    point.horizon_s
                )
                for point
                in track.points
            ],
        }
        for track in truth.tracks
    ]


print(
    "===== Block 3.8F "
    "checkpoint recovery ====="
)

if (
    sha256_file(
        FORMAL_MANIFEST
    )
    !=
    EXPECTED_FORMAL_SHA
):
    raise RuntimeError(
        "Formal manifest SHA changed."
    )

formal_rows = [
    json.loads(line)
    for line in (
        FORMAL_MANIFEST
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )
    if line.strip()
]

records = [
    json.loads(line)
    for line in (
        CHECKPOINT
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )
    if line.strip()
]

if (
    len(formal_rows) != 120
    or len(records) != 120
):
    raise RuntimeError(
        "Expected complete 120-scenario "
        "checkpoint."
    )

formal_ids = [
    row["scenario_id"]
    for row in formal_rows
]

checkpoint_ids = [
    row["scenario_id"]
    for row in records
]

if checkpoint_ids != formal_ids:
    raise RuntimeError(
        "Checkpoint scenario order "
        "does not equal frozen manifest."
    )

for expected_rank, record in enumerate(
    records,
    start=1,
):
    if record[
        "formal_rank"
    ] != expected_rank:
        raise RuntimeError(
            "Checkpoint rank mismatch."
        )

    stored = record[
        "deterministic_sha256"
    ]

    unhashed = dict(record)

    unhashed.pop(
        "deterministic_sha256"
    )

    actual = canonical_sha(
        strip_runtime(
            unhashed
        )
    )

    if stored != actual:
        raise RuntimeError(
            f"Checkpoint deterministic "
            f"hash mismatch: "
            f"{record['scenario_id']}"
        )


methods = tuple(
    records[0][
        "primary_evaluation"
    ].keys()
)

primary = {
    method: empty_aggregate()
    for method in methods
}

supplementary = {
    method: empty_aggregate()
    for method in methods
}


base_rows = (
    read_validation_manifest(
        BASE_MANIFEST
    )
)

row_by_id = {
    row.scenario_id: row
    for row in base_rows
}

offsets = (
    build_compact_motion_offset_index(
        base_rows
    )
)

scenario_hashes = []

for number, record in enumerate(
    records,
    start=1,
):
    sid = record[
        "scenario_id"
    ]

    primary_meta = record[
        "primary_truth"
    ]

    scenario = read_motion_scenario(
        row_by_id[sid],
        paired_root=PAIRED,
        compact_record_offset=(
            offsets[sid]
        ),
    )

    adapted = (
        adapt_causal_womd_scenario(
            scenario
        )
    )

    supplementary_truth = (
        build_womd_truth_sidecar(
            scenario,
            T_H0_from_W=(
                adapted.frames
                .T_H0_from_W
            ),
            eligible_track_indices=(
                current_valid_indices(
                    scenario
                )
            ),
            horizons_s=HORIZONS,
        )
    )

    supplementary_meta = (
        truth_meta(
            supplementary_truth
        )
    )

    for method in methods:
        ingest_report(
            primary[method],
            record[
                "primary_evaluation"
            ][method],
            primary_meta,
        )

        ingest_report(
            supplementary[
                method
            ],
            record[
                "supplementary_evaluation"
            ][method],
            supplementary_meta,
        )

    scenario_hashes.append(
        record[
            "deterministic_sha256"
        ]
    )

    if (
        number % 20
        ==
        0
    ):
        print(
            f"recovered "
            f"{number}/120",
            flush=True,
        )


primary_final = {
    method:
        finalize(aggregate)
    for method, aggregate
    in primary.items()
}

supplementary_final = {
    method:
        finalize(aggregate)
    for method, aggregate
    in supplementary.items()
}


deterministic_run_sha = (
    canonical_sha({
        "formal_manifest_sha256":
            EXPECTED_FORMAL_SHA,

        "scenario_hashes":
            scenario_hashes,

        "primary_metrics":
            strip_runtime(
                primary_final
            ),

        "supplementary_metrics":
            strip_runtime(
                supplementary_final
            ),
    })
)


scenario_runtime_ms = [
    float(
        record[
            "scenario_runtime_ms"
        ]
    )
    for record in records
]


report = {
    "stage": 3,
    "block": "3.8F",
    "status": "PASS",

    "formal": True,
    "scenario_count": 120,

    "formal_manifest_sha256":
        EXPECTED_FORMAL_SHA,

    "observation_mode":
        "full_frozen_Stage2_degraded",

    "measured_fmcw": False,

    "prediction_horizons_s":
        list(HORIZONS),

    "checkpoint_recovery": {
        "used": True,

        "reason": (
            "all 120 scenarios completed; "
            "original aggregate-only step "
            "encountered optional FDE=None"
        ),

        "predictors_rerun":
            False,

        "checkpoint_rows_verified":
            120,

        "scenario_deterministic_hashes_verified":
            True,
    },

    "primary_evaluation": {
        "target_policy": (
            "WOMD tracks_to_predict; "
            "evaluator metadata only; "
            "accessed after prediction"
        ),

        "metrics":
            primary_final,
    },

    "supplementary_evaluation": {
        "target_policy":
            "all current-valid non-SDC actors",

        "metrics":
            supplementary_final,
    },

    "aggregation_semantics": {
        "ADE": (
            "micro mean over every available "
            "matched actor-horizon error"
        ),

        "FDE": (
            "micro mean over matched tracks "
            "with evaluator-defined final "
            "endpoint; unavailable FDE count "
            "reported explicitly"
        ),

        "reconstruction": (
            "micro TP/FP/FN pooled over "
            "the 120 frozen scenarios"
        ),

        "horizon_endpoint_miss": (
            "sum endpoint misses / "
            "sum eligible truth tracks"
        ),
    },

    "runtime": {
        "scenario_runtime_sum_s":
            sum(
                scenario_runtime_ms
            ) / 1000.0,

        "scenario_median_ms":
            statistics.median(
                scenario_runtime_ms
            ),

        "scenario_p95_ms":
            percentile95(
                scenario_runtime_ms
            ),

        "scenario_max_ms":
            max(
                scenario_runtime_ms
            ),

        "wall_clock_note": (
            "original formal execution "
            "completed all 120 scenarios "
            "before aggregate-only failure"
        ),
    },

    "future_truth": {
        "algorithm_input":
            False,

        "assignment":
            False,

        "evaluation_only":
            True,

        "tracks_to_predict_accessed_after_prediction":
            True,
    },

    "deterministic": {
        "scenario_hashes":
            scenario_hashes,

        "run_sha256":
            deterministic_run_sha,
    },

    "per_scenario_artifact":
        str(CHECKPOINT),

    "per_scenario_artifact_sha256":
        sha256_file(
            CHECKPOINT
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
print(
    "===== Stage3 Block 3.8F "
    "FORMAL EVALUATION RECOVERED ====="
)

print(
    "checkpoint rows             = 120 / 120"
)

print(
    "predictors rerun            = NO"
)

print(
    "formal manifest SHA256      =",
    EXPECTED_FORMAL_SHA,
)

for method in sorted(
    primary_final
):
    m = primary_final[
        method
    ]

    print(
        f"{method:28s}"
        f" ADE={m['ade_m']:.6f}"
        f" FDE={m['fde_m']:.6f}"
        f" FDE-n={m['fde_track_count']}"
        f" FDE-unavailable="
        f"{m['fde_unavailable_matched_track_count']}"
        f" recall="
        f"{m['reconstruction_recall']:.6f}"
        f" F1="
        f"{m['reconstruction_f1']:.6f}"
    )

print(
    "deterministic run SHA256    =",
    deterministic_run_sha,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)
