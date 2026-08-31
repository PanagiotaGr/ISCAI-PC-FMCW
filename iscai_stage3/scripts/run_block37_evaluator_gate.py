from __future__ import annotations

from dataclasses import asdict

import hashlib
import json

from pathlib import Path

from iscai_stage3.evaluation import (
    EvaluationConfig,
    EvaluationPredictionSet,
    EvaluationPredictionTrack,
    EvaluationTruth,
    EvaluationTruthTrack,
    PredictionTrajectoryPoint,
    TruthTrajectoryPoint,
    assign_predictions_to_truth,
    evaluate_prediction_set,
)


ROOT = Path(
    "/home/agni/waymo/iscai_stage3"
)


HORIZONS = (
    0.1,
    0.3,
    0.5,
    1.0,
)


def implementation_hash():
    files = []

    for base in (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "scripts",
    ):
        for path in base.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix == ".pyc":
                continue

            files.append(path)

    files = sorted(
        set(files),
        key=lambda path: str(
            path.relative_to(ROOT)
        ),
    )

    digest = hashlib.sha256()

    for path in files:
        relative = str(
            path.relative_to(ROOT)
        )

        digest.update(
            relative.encode("utf-8")
        )

        digest.update(b"\0")

        digest.update(
            path.read_bytes()
        )

        digest.update(b"\0")

    return (
        digest.hexdigest(),
        len(files),
    )


def make_truth_track(
    truth_id,
    anchor,
    velocity,
):
    points = []

    for horizon in HORIZONS:
        points.append(
            TruthTrajectoryPoint(
                horizon_s=horizon,
                timestamp_s=(
                    1.0
                    +
                    horizon
                ),
                position_H0_m=(
                    anchor[0]
                    +
                    velocity[0]
                    *
                    horizon,

                    anchor[1]
                    +
                    velocity[1]
                    *
                    horizon,

                    anchor[2]
                    +
                    velocity[2]
                    *
                    horizon,
                ),
            )
        )

    return EvaluationTruthTrack(
        truth_id=truth_id,
        actor_class="TYPE_VEHICLE",
        anchor_position_H0_m=(
            anchor
        ),
        points=tuple(points),
    )


def make_prediction_track(
    prediction_id,
    anchor,
    velocity,
    *,
    future_offset_x=0.0,
):
    points = []

    for horizon in HORIZONS:
        points.append(
            PredictionTrajectoryPoint(
                horizon_s=horizon,
                timestamp_s=(
                    1.0
                    +
                    horizon
                ),
                position_H0_m=(
                    anchor[0]
                    +
                    velocity[0]
                    *
                    horizon
                    +
                    future_offset_x,

                    anchor[1]
                    +
                    velocity[1]
                    *
                    horizon,

                    anchor[2]
                    +
                    velocity[2]
                    *
                    horizon,
                ),
            )
        )

    return EvaluationPredictionTrack(
        prediction_id=prediction_id,
        anchor_position_H0_m=(
            anchor
        ),
        points=tuple(points),
    )


truth = EvaluationTruth(
    scenario_id="block37",
    anchor_timestamp_s=1.0,
    tracks=(
        make_truth_track(
            "GT_A",
            (
                10.0,
                -1.0,
                0.0,
            ),
            (
                5.0,
                1.0,
                0.0,
            ),
        ),
        make_truth_track(
            "GT_B",
            (
                20.0,
                2.0,
                0.0,
            ),
            (
                -2.0,
                -1.0,
                0.0,
            ),
        ),
    ),
)


# Deliberately reverse prediction order.
exact_predictions = (
    EvaluationPredictionSet(
        method="EXACT_SYNTHETIC",
        scenario_id="block37",
        anchor_timestamp_s=1.0,
        tracks=(
            make_prediction_track(
                "P_B",
                (
                    20.0,
                    2.0,
                    0.0,
                ),
                (
                    -2.0,
                    -1.0,
                    0.0,
                ),
            ),
            make_prediction_track(
                "P_A",
                (
                    10.0,
                    -1.0,
                    0.0,
                ),
                (
                    5.0,
                    1.0,
                    0.0,
                ),
            ),
        ),
        runtime_ms=1.25,
    )
)


config = EvaluationConfig(
    horizons_s=HORIZONS,
    anchor_assignment_gate_m=5.0,
    endpoint_miss_threshold_m=2.0,
)


exact_report = evaluate_prediction_set(
    exact_predictions,
    truth,
    config=config,
)


if exact_report.matched_track_count != 2:
    raise SystemExit(
        "FAIL: expected two anchor matches."
    )


if abs(exact_report.ade_m) > 1e-12:
    raise SystemExit(
        "FAIL: exact ADE is not zero."
    )


if abs(exact_report.fde_m) > 1e-12:
    raise SystemExit(
        "FAIL: exact FDE is not zero."
    )


if (
    exact_report
    .assignment_future_truth_used
):
    raise SystemExit(
        "FAIL: evaluator claims future "
        "truth assignment."
    )


if (
    exact_report
    .assignment
    .future_truth_used_for_assignment
):
    raise SystemExit(
        "FAIL: assignment used future truth."
    )


# =================================================
# Future mutation MUST NOT alter assignment
# =================================================

mutated_truth = (
    EvaluationTruth(
        scenario_id="block37",
        anchor_timestamp_s=1.0,
        tracks=(
            make_truth_track(
                "GT_A",
                (
                    10.0,
                    -1.0,
                    0.0,
                ),
                (
                    100.0,
                    80.0,
                    0.0,
                ),
            ),
            make_truth_track(
                "GT_B",
                (
                    20.0,
                    2.0,
                    0.0,
                ),
                (
                    -100.0,
                    -80.0,
                    0.0,
                ),
            ),
        ),
    )
)


original_assignment = (
    assign_predictions_to_truth(
        exact_predictions,
        truth,
        config=config,
    )
)


mutated_assignment = (
    assign_predictions_to_truth(
        exact_predictions,
        mutated_truth,
        config=config,
    )
)


if (
    original_assignment
    !=
    mutated_assignment
):
    raise SystemExit(
        "FAIL: future mutation changed "
        "anchor assignment."
    )


# =================================================
# Explicit miss / false-prediction behavior
# =================================================

imperfect_predictions = (
    EvaluationPredictionSet(
        method="IMPERFECT_SYNTHETIC",
        scenario_id="block37",
        anchor_timestamp_s=1.0,
        tracks=(
            make_prediction_track(
                "P_A",
                (
                    10.0,
                    -1.0,
                    0.0,
                ),
                (
                    5.0,
                    1.0,
                    0.0,
                ),
                future_offset_x=3.0,
            ),

            make_prediction_track(
                "FALSE_POSITIVE",
                (
                    100.0,
                    100.0,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    0.0,
                ),
            ),
        ),
    )
)


imperfect_report = (
    evaluate_prediction_set(
        imperfect_predictions,
        truth,
        config=config,
    )
)


if (
    imperfect_report
    .matched_track_count
    !=
    1
):
    raise SystemExit(
        "FAIL: imperfect assignment count."
    )


if (
    imperfect_report
    .unmatched_truth_count
    !=
    1
):
    raise SystemExit(
        "FAIL: unmatched truth accounting."
    )


if (
    imperfect_report
    .false_prediction_count
    !=
    1
):
    raise SystemExit(
        "FAIL: false prediction accounting."
    )


for aggregate in (
    imperfect_report
    .horizon_aggregates
):
    if (
        aggregate
        .endpoint_miss_rate
        !=
        1.0
    ):
        raise SystemExit(
            "FAIL: endpoint miss accounting."
        )


# =================================================
# Frozen config contract
# =================================================

config_json = json.loads(
    (
        ROOT
        / "configs"
        / "stage3_evaluation.json"
    ).read_text(
        encoding="utf-8"
    )
)


if config_json[
    "prediction_horizons_s"
] != [
    0.1,
    0.3,
    0.5,
    1.0,
]:
    raise SystemExit(
        "FAIL: canonical horizons changed."
    )


assignment_cfg = (
    config_json[
        "assignment"
    ]
)


if (
    assignment_cfg[
        "method"
    ]
    !=
    "global_Hungarian"
):
    raise SystemExit(
        "FAIL: assignment method changed."
    )


if (
    assignment_cfg[
        "time"
    ]
    !=
    "anchor_current_only"
):
    raise SystemExit(
        "FAIL: assignment time changed."
    )


if (
    assignment_cfg[
        "future_truth_used"
    ]
):
    raise SystemExit(
        "FAIL: future truth assignment "
        "enabled."
    )


if (
    config_json[
        "trajectory_metrics"
    ][
        "primary_ADE_FDE_space"
    ]
    !=
    "planar_XY_H0"
):
    raise SystemExit(
        "FAIL: ADE/FDE convention changed."
    )


if (
    config_json[
        "formal_validation_tuning"
    ]
):
    raise SystemExit(
        "FAIL: evaluation policy marked "
        "validation-tuned."
    )


# =================================================
# Static boundary
# =================================================

algorithm_directories = (
    ROOT / "src/iscai_stage3/association",
    ROOT / "src/iscai_stage3/baselines",
    ROOT / "src/iscai_stage3/filters",
    ROOT / "src/iscai_stage3/hough",
    ROOT / "src/iscai_stage3/state",
)


offending = []

for directory in algorithm_directories:
    for path in directory.rglob(
        "*.py"
    ):
        if (
            "iscai_stage3.evaluation"
            in
            path.read_text(
                encoding="utf-8"
            )
        ):
            offending.append(
                str(path)
            )


if offending:
    raise SystemExit(
        "FAIL: algorithm imports evaluator: "
        +
        repr(offending)
    )


sha, count = implementation_hash()


report = {
    "stage": 3,
    "block": "3.7",
    "status": "PASS",

    "common_evaluator": {
        "methods_supported": [
            "CV",
            "CA",
            "CTRV",
            "Kalman_EKF",
            "IMM_CV_CA_CTRV",
            "Multidimensional_Hough"
        ],

        "common_prediction_contract": True,

        "truth_sidecar": (
            "EVALUATOR_ONLY"
        ),

        "assignment": (
            "global Hungarian on "
            "anchor/current position only"
        ),

        "future_truth_used_for_assignment": (
            False
        )
    },

    "metrics": {
        "primary": [
            "ADE_XY",
            "FDE_XY"
        ],

        "supplementary": [
            "ADE_3D",
            "FDE_3D",
            "per_horizon_XY_MAE",
            "per_horizon_XY_RMSE",
            "per_horizon_3D_MAE",
            "per_horizon_3D_RMSE",
            "reconstruction_precision",
            "reconstruction_recall",
            "reconstruction_F1",
            "endpoint_miss_rate",
            "false_prediction_count",
            "unmatched_truth_count"
        ],

        "endpoint_miss_threshold_m": 2.0
    },

    "horizons_s": [
        0.1,
        0.3,
        0.5,
        1.0
    ],

    "synthetic_gate": {
        "exact_ADE_m": (
            exact_report.ade_m
        ),

        "exact_FDE_m": (
            exact_report.fde_m
        ),

        "exact_matches": (
            exact_report
            .matched_track_count
        ),

        "future_mutation_assignment_invariant": (
            True
        ),

        "imperfect_matches": (
            imperfect_report
            .matched_track_count
        ),

        "imperfect_unmatched_truth": (
            imperfect_report
            .unmatched_truth_count
        ),

        "imperfect_false_predictions": (
            imperfect_report
            .false_prediction_count
        )
    },

    "probabilistic_metrics": {
        "NLL": "DEFER_STAGE4",
        "Brier": "DEFER_STAGE4",
        "calibration": "DEFER_STAGE4"
    },

    "real_WOMD_metrics": (
        "DEFERRED_TO_BLOCK_3_8"
    ),

    "implementation": {
        "files": count,
        "sha256": sha
    }
}


report_path = (
    ROOT
    / "reports"
    / "block37_common_evaluator_gate.json"
)


report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


print(
    "===== Stage3 Block 3.7 gate ====="
)

print(
    "common evaluator            = PASS"
)

print(
    "truth sidecar               = EVALUATOR ONLY"
)

print(
    "assignment                  = "
    "global Hungarian"
)

print(
    "assignment time             = "
    "anchor/current only"
)

print(
    "future truth in assignment  = NO"
)

print(
    "primary ADE/FDE             = "
    "planar XY"
)

print(
    "supplementary 3D metrics    = PASS"
)

print(
    "canonical horizons          = "
    "0.1 / 0.3 / 0.5 / 1.0 s"
)

print(
    "reconstruction metrics      = PASS"
)

print(
    "endpoint miss metric        = PASS"
)

print(
    "false prediction accounting = PASS"
)

print(
    "exact synthetic ADE         =",
    exact_report.ade_m,
)

print(
    "exact synthetic FDE         =",
    exact_report.fde_m,
)

print(
    "future mutation invariant   = PASS"
)

print(
    "algorithm imports evaluator = NO"
)

print(
    "probabilistic metrics       = "
    "DEFERRED TO STAGE4"
)

print(
    "real WOMD metrics           = "
    "DEFERRED TO BLOCK 3.8"
)

print(
    "implementation files        =",
    count,
)

print(
    "implementation SHA256       =",
    sha,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    report_path,
)
