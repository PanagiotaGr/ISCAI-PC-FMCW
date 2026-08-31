from __future__ import annotations

import ast
from collections import Counter, defaultdict
import dataclasses
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
import statistics
import time

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
)
from iscai_stage2.observations.clean_scene import (
    build_clean_observation_scene,
)
from iscai_stage2.observations.degraded_scene import (
    DegradedObservationConfig,
    build_degraded_observation_scene,
    degraded_algorithm_sha256,
)
from iscai_stage2.observations.detection import (
    DetectionProbabilityConfig,
)
from iscai_stage2.observations.detection_set import (
    FalseAlarmConfig,
)
from iscai_stage2.observations.womd_ideal_adapter import (
    build_causal_dynamic_headlamp_frames,
    build_real_ideal_observation_scene,
)

from iscai_stage3.association import (
    associate_estimated_gnn,
)
from iscai_stage3.baselines import (
    predict_ca,
    predict_ctrv,
    predict_cv,
    predict_imm,
    predict_kalman,
)
from iscai_stage3.evaluation import (
    build_womd_truth_sidecar,
)
from iscai_stage3.evaluation.anchor_alignment import (
    align_imm_states_to_global_anchor,
    align_kalman_state_to_global_anchor,
    align_state_prediction_to_global_anchor,
)
from iscai_stage3.filters import (
    IMMConfig,
    KalmanConfig,
    KalmanFrameContext,
    filter_associated_track,
    filter_associated_track_imm,
)
from iscai_stage3.geometry import (
    FrameTransformContext,
    cartesianize_associated_track,
)
from iscai_stage3.observations import (
    algorithm_sequence_from_degraded_scene,
)
from iscai_stage3.state import (
    estimate_causal_ca_state,
    estimate_causal_ctrv_state,
    estimate_causal_cv_state,
)
from iscai_stage3.validation import (
    build_compact_motion_offset_index,
    read_motion_scenario,
    read_validation_manifest,
    sha256_file,
)


ROOT = Path("/home/agni/waymo")
STAGE3 = ROOT / "iscai_stage3"
SRC = STAGE3 / "src/iscai_stage3"

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

CONFIG = (
    STAGE3
    / "configs/stage3_real_validation.json"
)

STAGE2_PROFILE_SOURCE = (
    ROOT
    / "iscai_stage2/scripts/"
      "run_stage2_degraded_scene_smoke.py"
)

PER_SCENARIO = (
    STAGE3
    / "artifacts/block38f/"
      "formal_per_scenario.jsonl"
)

REPORT = (
    STAGE3
    / "reports/block38f_formal_evaluation.json"
)

EXPECTED_BASE_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

EXPECTED_FORMAL_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

FORMAL_N = 120

HORIZONS = (
    0.1,
    0.3,
    0.5,
    1.0,
)

MISS_THRESHOLD_M = 2.0


def locate_symbol(name):
    matches = []

    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue

        tree = ast.parse(
            path.read_text(
                encoding="utf-8"
            )
        )

        for node in tree.body:
            if (
                isinstance(
                    node,
                    (
                        ast.ClassDef,
                        ast.FunctionDef,
                    ),
                )
                and node.name == name
            ):
                matches.append(path)

    if len(matches) != 1:
        raise RuntimeError(
            f"{name}: expected one "
            f"definition, got {matches}"
        )

    relative = (
        matches[0]
        .relative_to(
            STAGE3 / "src"
        )
        .with_suffix("")
    )

    module = importlib.import_module(
        ".".join(relative.parts)
    )

    return getattr(
        module,
        name,
    )


HoughConfig = locate_symbol(
    "HoughConfig"
)
HoughFrameContext = locate_symbol(
    "HoughFrameContext"
)
run_multidimensional_hough = (
    locate_symbol(
        "run_multidimensional_hough"
    )
)
predict_hough = locate_symbol(
    "predict_hough"
)

EvaluationConfig = locate_symbol(
    "EvaluationConfig"
)
EvaluationPredictionSet = (
    locate_symbol(
        "EvaluationPredictionSet"
    )
)
adapt_prediction_object = (
    locate_symbol(
        "adapt_prediction_object"
    )
)
evaluate_prediction_set = (
    locate_symbol(
        "evaluate_prediction_set"
    )
)


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return jsonable(
            dataclasses.asdict(value)
        )

    if isinstance(value, dict):
        return {
            str(k): jsonable(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (tuple, list),
    ):
        return [
            jsonable(v)
            for v in value
        ]

    return value


def percentile95(values):
    if not values:
        return None

    values = sorted(
        float(v)
        for v in values
    )

    return values[
        max(
            0,
            math.ceil(
                0.95
                *
                len(values)
            )
            -
            1,
        )
    ]


def load_frozen_stage2_configs():
    source = (
        STAGE2_PROFILE_SOURCE
        .read_text(
            encoding="utf-8"
        )
    )

    tree = ast.parse(source)

    namespace = {
        "math": math,
        "Path": Path,
        "CleanObservationConfig":
            CleanObservationConfig,
        "DegradedObservationConfig":
            DegradedObservationConfig,
        "DetectionProbabilityConfig":
            DetectionProbabilityConfig,
        "FalseAlarmConfig":
            FalseAlarmConfig,
    }

    for node in tree.body:
        name = None
        value = None

        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(
                node.targets[0],
                ast.Name,
            )
        ):
            name = node.targets[0].id
            value = node.value

        elif (
            isinstance(
                node,
                ast.AnnAssign,
            )
            and isinstance(
                node.target,
                ast.Name,
            )
        ):
            name = node.target.id
            value = node.value

        if name is None or value is None:
            continue

        try:
            namespace[name] = eval(
                compile(
                    ast.Expression(value),
                    str(
                        STAGE2_PROFILE_SOURCE
                    ),
                    "eval",
                ),
                {
                    "__builtins__": {},
                },
                namespace,
            )
        except (
            NameError,
            TypeError,
            AttributeError,
        ):
            pass

    return (
        namespace["CLEAN_CONFIG"],
        namespace["DEGRADED_CONFIG"],
    )


def strip_runtime(value):
    if isinstance(value, dict):
        return {
            key: strip_runtime(item)
            for key, item in value.items()
            if "runtime" not in key.lower()
        }

    if isinstance(value, list):
        return [
            strip_runtime(item)
            for item in value
        ]

    return value


def canonical_sha(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return sha256(
        payload
    ).hexdigest()


def build_contexts(
    scenario,
    adapted,
):
    dynamic_frames, _ = (
        build_causal_dynamic_headlamp_frames(
            adapted
        )
    )

    expected = (
        int(
            scenario.current_time_index
        )
        +
        1
    )

    if (
        len(dynamic_frames)
        != expected
        or any(
            f is None
            for f in dynamic_frames
        )
    ):
        raise RuntimeError(
            "Invalid causal dynamic "
            "headlamp frames."
        )

    timestamps = tuple(
        float(x)
        for x
        in scenario.timestamps_seconds[
            :expected
        ]
    )

    transforms = (
        FrameTransformContext(
            T_H0_from_W=(
                adapted.frames
                .T_H0_from_W
            ),
            T_Ht_from_W_by_frame=tuple(
                f.T_Ht_from_W
                for f in dynamic_frames
            ),
        )
    )

    return (
        timestamps,
        transforms,
        KalmanFrameContext(
            transforms=transforms,
            frame_timestamps_s=(
                timestamps
            ),
        ),
        HoughFrameContext(
            transforms=transforms,
            frame_timestamps_s=(
                timestamps
            ),
        ),
    )


def adapted_prediction(
    prediction,
    anchor_position,
):
    return adapt_prediction_object(
        prediction,
        anchor_position_H0_m=(
            anchor_position
        ),
    )


def make_set(
    method,
    scenario_id,
    anchor_timestamp,
    tracks,
    runtime_ms,
):
    return EvaluationPredictionSet(
        method=method,
        scenario_id=scenario_id,
        anchor_timestamp_s=(
            anchor_timestamp
        ),
        tracks=tuple(tracks),
        runtime_ms=float(runtime_ms),
    )


def run_state_method(
    *,
    name,
    tracks,
    transforms,
    estimator,
    predictor,
    association_ms,
    scenario_id,
    anchor_timestamp,
):
    start = time.perf_counter()

    predictions = []
    rejected = Counter()

    for track in tracks:
        try:
            observations = (
                cartesianize_associated_track(
                    track,
                    context=transforms,
                )
            )

            if not observations:
                raise ValueError(
                    "no Cartesian observations"
                )

            state = estimator(
                observations
            )

            predictions.append(
                align_state_prediction_to_global_anchor(
                    state,
                    predictor,
                    horizons_s=HORIZONS,
                    anchor_timestamp_s=anchor_timestamp,
                )
            )

        except ValueError as exc:
            rejected[
                str(exc)
            ] += 1

    model_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    return (
        make_set(
            name,
            scenario_id,
            anchor_timestamp,
            predictions,
            association_ms
            +
            model_ms,
        ),
        {
            "prediction_count":
                len(predictions),
            "rejected_count":
                sum(
                    rejected.values()
                ),
            "rejections":
                dict(rejected),
            "association_runtime_ms":
                association_ms,
            "model_runtime_ms":
                model_ms,
        },
    )


def run_kalman_method(
    *,
    tracks,
    transforms,
    context,
    association_ms,
    scenario_id,
    anchor_timestamp,
):
    start = time.perf_counter()

    config = KalmanConfig()
    predictions = []
    rejected = Counter()

    for track in tracks:
        try:
            observations = (
                cartesianize_associated_track(
                    track,
                    context=transforms,
                )
            )

            result = (
                filter_associated_track(
                    track,
                    context=context,
                    config=config,
                )
            )

            aligned_state = (
                align_kalman_state_to_global_anchor(
                    result.current_state,
                    anchor_timestamp_s=anchor_timestamp,
                    config=config,
                )
            )

            prediction = (
                predict_kalman(
                    track_id=(
                        result.track_id
                    ),
                    state=aligned_state,
                    horizons_s=HORIZONS,
                    config=config,
                )
            )

            predictions.append(
                adapted_prediction(
                    prediction,
                    tuple(
                        float(aligned_state.mean_6[i])
                        for i in range(3)
                    ),
                )
            )

        except ValueError as exc:
            rejected[
                str(exc)
            ] += 1

    model_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    return (
        make_set(
            "Kalman_EKF",
            scenario_id,
            anchor_timestamp,
            predictions,
            association_ms
            +
            model_ms,
        ),
        {
            "prediction_count":
                len(predictions),
            "rejected_count":
                sum(
                    rejected.values()
                ),
            "rejections":
                dict(rejected),
            "association_runtime_ms":
                association_ms,
            "model_runtime_ms":
                model_ms,
        },
    )


def run_imm_method(
    *,
    tracks,
    transforms,
    context,
    association_ms,
    scenario_id,
    anchor_timestamp,
):
    start = time.perf_counter()

    config = IMMConfig()
    predictions = []
    rejected = Counter()

    for track in tracks:
        try:
            observations = (
                cartesianize_associated_track(
                    track,
                    context=transforms,
                )
            )

            result = (
                filter_associated_track_imm(
                    track,
                    context=context,
                    config=config,
                )
            )

            (
                aligned_states,
                aligned_probabilities,
                aligned_anchor_position,
            ) = align_imm_states_to_global_anchor(
                result.current_mode_states,
                result.current_mode_probabilities,
                frame_timestamps_s=(
                    context.frame_timestamps_s
                ),
                anchor_timestamp_s=anchor_timestamp,
                config=config,
            )

            prediction = predict_imm(
                track_id=(
                    result.track_id
                ),
                states=aligned_states,
                mode_probabilities=(
                    aligned_probabilities
                ),
                horizons_s=HORIZONS,
                config=config,
            )

            predictions.append(
                adapted_prediction(
                    prediction,
                    aligned_anchor_position,
                )
            )

        except ValueError as exc:
            rejected[
                str(exc)
            ] += 1

    model_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    return (
        make_set(
            "IMM_CV_CA_CTRV",
            scenario_id,
            anchor_timestamp,
            predictions,
            association_ms
            +
            model_ms,
        ),
        {
            "prediction_count":
                len(predictions),
            "rejected_count":
                sum(
                    rejected.values()
                ),
            "rejections":
                dict(rejected),
            "association_runtime_ms":
                association_ms,
            "model_runtime_ms":
                model_ms,
        },
    )


def run_hough_method(
    *,
    sequence,
    context,
    scenario_id,
    anchor_timestamp,
):
    start = time.perf_counter()

    result = (
        run_multidimensional_hough(
            sequence,
            context=context,
            config=HoughConfig(),
        )
    )

    if any((
        bool(
            getattr(
                result,
                "truth_used",
                False,
            )
        ),
        bool(
            getattr(
                result,
                "estimated_association_used",
                False,
            )
        ),
        bool(
            getattr(
                result,
                "future_information_used",
                False,
            )
        ),
    )):
        raise RuntimeError(
            "Hough boundary violation."
        )

    predictions = []

    for track in result.tracks:
        prediction = predict_hough(
            track,
            horizons_s=HORIZONS,
        )

        predictions.append(
            adapted_prediction(
                prediction,
                track
                .anchor_position_H0_m,
            )
        )

    model_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    return (
        make_set(
            "Multidimensional_Hough",
            scenario_id,
            anchor_timestamp,
            predictions,
            model_ms,
        ),
        {
            "prediction_count":
                len(predictions),
            "hough_track_count":
                len(result.tracks),
            "model_runtime_ms":
                model_ms,
        },
    )


EVAL_CONFIG = EvaluationConfig(
    horizons_s=HORIZONS,
    anchor_assignment_gate_m=5.0,
    endpoint_miss_threshold_m=(
        MISS_THRESHOLD_M
    ),
    horizon_tolerance_s=1e-6,
    anchor_timestamp_tolerance_s=2e-3,
)


def current_valid_indices(
    scenario,
):
    anchor = int(
        scenario.current_time_index
    )

    return tuple(
        index
        for index, track
        in enumerate(
            scenario.tracks
        )
        if (
            index
            != int(
                scenario.sdc_track_index
            )
            and
            track.states[
                anchor
            ].valid
        )
    )


def benchmark_indices(
    scenario,
):
    """
    Evaluator metadata only.

    Called strictly after prediction.
    """

    anchor = int(
        scenario.current_time_index
    )

    result = []

    for required in (
        scenario.tracks_to_predict
    ):
        index = int(
            required.track_index
        )

        if index == int(
            scenario.sdc_track_index
        ):
            continue

        if not (
            0 <= index
            <
            len(scenario.tracks)
        ):
            raise RuntimeError(
                "Invalid tracks_to_predict "
                "track_index."
            )

        if (
            scenario.tracks[
                index
            ].states[
                anchor
            ].valid
        ):
            result.append(index)

    return tuple(
        sorted(
            set(result)
        )
    )


def truth_metadata(truth):
    return [
        {
            "truth_id":
                str(track.truth_id),
            "actor_class":
                track.actor_class,
            "available_horizons_s": [
                float(point.horizon_s)
                for point
                in track.points
            ],
        }
        for track in truth.tracks
    ]


def evaluate_view(
    prediction_sets,
    truth,
    *,
    ignored_truth=None,
):
    result = {}

    for method, predictions in (
        prediction_sets.items()
    ):
        report = (
            evaluate_prediction_set(
                predictions,
                truth=truth,
                config=EVAL_CONFIG,
                ignored_truth=ignored_truth,
            )
        )

        report = jsonable(report)

        if (
            report[
                "assignment_future_truth_used"
            ]
            or
            report[
                "assignment"
            ][
                "future_truth_used_for_assignment"
            ]
        ):
            raise RuntimeError(
                "Future truth entered "
                "assignment."
            )

        result[method] = report

    return result


def empty_aggregate():
    return {
        "prediction_tracks": 0,
        "truth_tracks": 0,
        "matched_tracks": 0,
        "false_predictions": 0,
        "ignored_predictions": 0,
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
                lambda: {
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
            ),
    }


def ingest_report(
    agg,
    report,
    truth_meta,
):
    agg["prediction_tracks"] += (
        report[
            "prediction_track_count"
        ]
    )
    agg["truth_tracks"] += (
        report[
            "truth_track_count"
        ]
    )
    agg["matched_tracks"] += (
        report[
            "matched_track_count"
        ]
    )
    agg["false_predictions"] += (
        report[
            "false_prediction_count"
        ]
    )
    agg["ignored_predictions"] += (
        report.get(
            "ignored_prediction_count",
            0,
        )
    )
    agg["unmatched_truth"] += (
        report[
            "unmatched_truth_count"
        ]
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

        agg[
            "horizon_eligible"
        ][h] += (
            horizon[
                "eligible_truth_tracks"
            ]
        )

        agg[
            "horizon_misses"
        ][h] += (
            horizon[
                "endpoint_miss_count"
            ]
        )

    track_by_truth = {
        str(track["truth_id"]):
            track
        for track
        in report[
            "track_evaluations"
        ]
    }

    matched_truth_ids = {
        str(match["truth_id"])
        for match
        in report[
            "assignment"
        ]["matches"]
    }

    for track in (
        report["track_evaluations"]
    ):
        class_agg = agg[
            "classes"
        ][
            track["actor_class"]
        ]

        fde_m = track["fde_m"]
        fde_3d_m = track["fde_3d_m"]

        if (
            (fde_m is None)
            !=
            (fde_3d_m is None)
        ):
            raise RuntimeError(
                "Planar/3D FDE availability "
                "disagrees."
            )

        if fde_m is None:
            agg[
                "fde_unavailable_matched_tracks"
            ] += 1

            class_agg[
                "fde_unavailable_matched_tracks"
            ] += 1

        else:
            agg[
                "fde_planar_errors"
            ].append(
                float(fde_m)
            )

            agg[
                "fde_3d_errors"
            ].append(
                float(fde_3d_m)
            )

            class_agg[
                "fde_planar_errors"
            ].append(
                float(fde_m)
            )

            class_agg[
                "fde_3d_errors"
            ].append(
                float(fde_3d_m)
            )

        agg[
            "anchor_errors"
        ].append(
            track[
                "anchor_assignment_error_m"
            ]
        )

        for point in (
            track["horizon_errors"]
        ):
            h = float(
                point["horizon_s"]
            )

            planar = float(
                point[
                    "planar_error_m"
                ]
            )
            error3 = float(
                point[
                    "error_3d_m"
                ]
            )

            agg[
                "ade_planar_errors"
            ].append(planar)

            agg[
                "ade_3d_errors"
            ].append(error3)

            agg[
                "horizon_errors"
            ][h].append(planar)

            agg[
                "horizon_errors_3d"
            ][h].append(error3)

            class_agg[
                "ade_planar_errors"
            ].append(planar)

            class_agg[
                "ade_3d_errors"
            ].append(error3)

            class_agg[
                "horizon_errors"
            ][h].append(planar)

            class_agg[
                "horizon_errors_3d"
            ][h].append(error3)

    # Class-specific truth support + miss semantics.
    for truth in truth_meta:
        truth_id = str(
            truth["truth_id"]
        )

        actor_class = (
            truth["actor_class"]
        )

        c = agg[
            "classes"
        ][actor_class]

        c["truth_tracks"] += 1

        if truth_id in matched_truth_ids:
            c["matched_tracks"] += 1
        else:
            c[
                "unmatched_truth"
            ] += 1

        evaluation = (
            track_by_truth.get(
                truth_id
            )
        )

        error_by_horizon = {}

        if evaluation is not None:
            error_by_horizon = {
                float(point["horizon_s"]):
                    float(
                        point[
                            "planar_error_m"
                        ]
                    )
                for point
                in evaluation[
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

            error = (
                error_by_horizon.get(h)
            )

            if (
                error is None
                or error
                >
                MISS_THRESHOLD_M
            ):
                c[
                    "horizon_misses"
                ][h] += 1


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
            value * value
            for value in values
        )
        /
        len(values)
    )


def finalize_aggregate(agg):
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
        / (precision + recall)
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

    class_reports = {}

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

            class_horizons[
                str(h)
            ] = {
                "eligible_truth_tracks":
                    eligible,
                "available_predictions":
                    len(
                        c[
                            "horizon_errors"
                        ][h]
                    ),
                "endpoint_miss_count":
                    misses,
                "endpoint_miss_rate":
                    (
                        misses / eligible
                        if eligible
                        else None
                    ),
                "mean_planar_error_m":
                    mean(
                        c[
                            "horizon_errors"
                        ][h]
                    ),
                "rmse_planar_error_m":
                    rmse(
                        c[
                            "horizon_errors"
                        ][h]
                    ),
            }

        class_reports[
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
                    c[
                        "matched_tracks"
                    ]
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
        "ignored_prediction_count":
            agg["ignored_predictions"],
        "unmatched_truth_count":
            fn,

        "reconstruction_precision":
            precision,
        "reconstruction_recall":
            recall,
        "reconstruction_f1":
            f1,

        # Micro ADE:
        # every available matched
        # actor-horizon point has equal weight.
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

        # FDE:
        # one evaluator-defined final error
        # per matched track.
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
                agg[
                    "anchor_errors"
                ]
            ),

        "horizons":
            horizons,

        "classes":
            class_reports,

        "runtime_ms": {
            "total":
                sum(
                    agg[
                        "runtime_ms"
                    ]
                ),
            "median":
                (
                    statistics.median(
                        agg[
                            "runtime_ms"
                        ]
                    )
                    if agg[
                        "runtime_ms"
                    ]
                    else None
                ),
            "p95":
                percentile95(
                    agg[
                        "runtime_ms"
                    ]
                ),
        },
    }


# ============================================================
# Frozen-manifest integrity
# ============================================================

if (
    sha256_file(BASE_MANIFEST)
    !=
    EXPECTED_BASE_SHA
):
    raise RuntimeError(
        "Base validation manifest "
        "SHA changed."
    )

if (
    sha256_file(FORMAL_MANIFEST)
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

if len(formal_rows) != FORMAL_N:
    raise RuntimeError(
        "Formal N changed."
    )

config = json.loads(
    CONFIG.read_text(
        encoding="utf-8"
    )
)

formal_config = config[
    "formal_validation"
]

if (
    formal_config["N"] != 120
    or formal_config[
        "manifest_status"
    ] != "FROZEN"
    or formal_config[
        "manifest_sha256"
    ] != EXPECTED_FORMAL_SHA
):
    raise RuntimeError(
        "Frozen formal config changed."
    )


base_rows = (
    read_validation_manifest(
        BASE_MANIFEST
    )
)

row_by_id = {
    row.scenario_id: row
    for row in base_rows
}

compact_offsets = (
    build_compact_motion_offset_index(
        base_rows
    )
)

clean_config, degraded_config = (
    load_frozen_stage2_configs()
)


# ============================================================
# Run the exact frozen formal set.
# ============================================================

PER_SCENARIO.parent.mkdir(
    parents=True,
    exist_ok=True,
)

primary_aggregates = {
    method: empty_aggregate()
    for method in (
        "CV",
        "CA",
        "CTRV",
        "Kalman_EKF",
        "IMM_CV_CA_CTRV",
        "Multidimensional_Hough",
    )
}

supplementary_aggregates = {
    method: empty_aggregate()
    for method in (
        primary_aggregates
    )
}

scenario_hashes = []
scenario_total_runtimes = []

formal_start = time.perf_counter()

with PER_SCENARIO.open(
    "w",
    encoding="utf-8",
) as stream:

    for number, formal_row in enumerate(
        formal_rows,
        start=1,
    ):
        sid = formal_row[
            "scenario_id"
        ]

        print(
            f"[{number:03d}/120] "
            f"{sid}",
            flush=True,
        )

        row = row_by_id[sid]

        scenario_start = (
            time.perf_counter()
        )

        scenario = (
            read_motion_scenario(
                row,
                paired_root=PAIRED,
                compact_record_offset=(
                    compact_offsets[sid]
                ),
            )
        )

        adapted = (
            adapt_causal_womd_scenario(
                scenario
            )
        )

        ideal = (
            build_real_ideal_observation_scene(
                raw_scenario=scenario,
                adapted=adapted,
                include_sdc=False,
            )
        )

        clean = (
            build_clean_observation_scene(
                ideal_scene=ideal,
                config=clean_config,
            )
        )

        degraded = (
            build_degraded_observation_scene(
                clean_scene=clean,
                config=degraded_config,
            )
        )

        sequence = (
            algorithm_sequence_from_degraded_scene(
                scenario_id=sid,
                scene=degraded,
            )
        )

        input_before = (
            degraded_algorithm_sha256(
                degraded
            )
        )

        (
            timestamps,
            transforms,
            kalman_context,
            hough_context,
        ) = build_contexts(
            scenario,
            adapted,
        )

        association_start = (
            time.perf_counter()
        )

        associated = (
            associate_estimated_gnn(
                sequence
            )
        )

        association_ms = (
            time.perf_counter()
            -
            association_start
        ) * 1000.0

        anchor_timestamp = (
            timestamps[-1]
        )

        all_associated_tracks = tuple(
            associated.tracks
        )

        # A track may exist historically but already be terminated
        # before the global anchor.  Such a track must not emit an
        # anchor forecast.  Still-active tracks with a recent miss
        # are retained and causally propagated to the anchor.
        tracks = tuple(
            track
            for track in all_associated_tracks
            if not track.terminated
        )

        stale_active_tracks = tuple(
            track
            for track in tracks
            if track.detections
            and track.detections[-1].timestamp_s
                < anchor_timestamp - 2e-3
        )

        prediction_sets = {}
        method_info = {}

        for (
            method,
            estimator,
            predictor,
        ) in (
            (
                "CV",
                estimate_causal_cv_state,
                predict_cv,
            ),
            (
                "CA",
                estimate_causal_ca_state,
                predict_ca,
            ),
            (
                "CTRV",
                estimate_causal_ctrv_state,
                predict_ctrv,
            ),
        ):
            (
                prediction_sets[method],
                method_info[method],
            ) = run_state_method(
                name=method,
                tracks=tracks,
                transforms=transforms,
                estimator=estimator,
                predictor=predictor,
                association_ms=(
                    association_ms
                ),
                scenario_id=sid,
                anchor_timestamp=(
                    anchor_timestamp
                ),
            )

        (
            prediction_sets[
                "Kalman_EKF"
            ],
            method_info[
                "Kalman_EKF"
            ],
        ) = run_kalman_method(
            tracks=tracks,
            transforms=transforms,
            context=kalman_context,
            association_ms=(
                association_ms
            ),
            scenario_id=sid,
            anchor_timestamp=(
                anchor_timestamp
            ),
        )

        (
            prediction_sets[
                "IMM_CV_CA_CTRV"
            ],
            method_info[
                "IMM_CV_CA_CTRV"
            ],
        ) = run_imm_method(
            tracks=tracks,
            transforms=transforms,
            context=kalman_context,
            association_ms=(
                association_ms
            ),
            scenario_id=sid,
            anchor_timestamp=(
                anchor_timestamp
            ),
        )

        (
            prediction_sets[
                "Multidimensional_Hough"
            ],
            method_info[
                "Multidimensional_Hough"
            ],
        ) = run_hough_method(
            sequence=sequence,
            context=hough_context,
            scenario_id=sid,
            anchor_timestamp=(
                anchor_timestamp
            ),
        )

        # ====================================================
        # Critical boundary:
        # all algorithms have finished BEFORE any access to
        # tracks_to_predict or WOMD future truth.
        # ====================================================

        input_after_prediction = (
            degraded_algorithm_sha256(
                degraded
            )
        )

        if (
            input_before
            !=
            input_after_prediction
        ):
            raise RuntimeError(
                "Algorithm input mutation."
            )

        primary_indices = (
            benchmark_indices(
                scenario
            )
        )

        all_current_indices = (
            current_valid_indices(
                scenario
            )
        )

        if not primary_indices:
            raise RuntimeError(
                "Formal scenario has zero "
                "current-valid "
                "tracks_to_predict."
            )

        primary_truth = (
            build_womd_truth_sidecar(
                scenario,
                T_H0_from_W=(
                    adapted.frames
                    .T_H0_from_W
                ),
                eligible_track_indices=(
                    primary_indices
                ),
                horizons_s=HORIZONS,
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
                    all_current_indices
                ),
                horizons_s=HORIZONS,
            )
        )

        ignored_indices = tuple(
            sorted(
                set(all_current_indices)
                - set(primary_indices)
            )
        )

        ignored_truth = (
            build_womd_truth_sidecar(
                scenario,
                T_H0_from_W=(
                    adapted.frames
                    .T_H0_from_W
                ),
                eligible_track_indices=(
                    ignored_indices
                ),
                horizons_s=HORIZONS,
            )
        )

        primary_reports = (
            evaluate_view(
                prediction_sets,
                primary_truth,
                ignored_truth=ignored_truth,
            )
        )

        supplementary_reports = (
            evaluate_view(
                prediction_sets,
                supplementary_truth,
            )
        )

        primary_meta = (
            truth_metadata(
                primary_truth
            )
        )

        supplementary_meta = (
            truth_metadata(
                supplementary_truth
            )
        )

        for method in prediction_sets:
            ingest_report(
                primary_aggregates[
                    method
                ],
                primary_reports[
                    method
                ],
                primary_meta,
            )

            ingest_report(
                supplementary_aggregates[
                    method
                ],
                supplementary_reports[
                    method
                ],
                supplementary_meta,
            )

        input_final = (
            degraded_algorithm_sha256(
                degraded
            )
        )

        if input_final != input_before:
            raise RuntimeError(
                "Evaluator mutated "
                "algorithm input."
            )

        scenario_runtime_ms = (
            time.perf_counter()
            -
            scenario_start
        ) * 1000.0

        scenario_total_runtimes.append(
            scenario_runtime_ms
        )

        record = {
            "formal_rank":
                formal_row[
                    "formal_rank"
                ],

            "scenario_id":
                sid,

            "stratum":
                formal_row["stratum"],

            "stage2_algorithm_sha256":
                input_before,

            "algorithm_input_immutable":
                True,

            "association_runtime_ms":
                association_ms,

            "association_track_count_all":
                len(all_associated_tracks),

            "anchor_active_track_count":
                len(tracks),

            "terminated_track_count_excluded":
                (
                    len(all_associated_tracks)
                    - len(tracks)
                ),

            "stale_active_track_count_propagated":
                len(stale_active_tracks),

            "scenario_runtime_ms":
                scenario_runtime_ms,

            "prediction_info":
                method_info,

            "primary_target_policy":
                "WOMD_tracks_to_predict_evaluator_only_"
                "with_current_real_nontarget_ignore",

            "primary_target_count":
                len(primary_indices),

            "primary_ignored_real_nontarget_count":
                len(ignored_indices),

            "supplementary_target_policy":
                "all_current_valid_non_SDC",

            "supplementary_target_count":
                len(
                    all_current_indices
                ),

            "primary_truth":
                primary_meta,

            "primary_evaluation":
                primary_reports,

            "supplementary_evaluation":
                supplementary_reports,
        }

        deterministic = (
            strip_runtime(record)
        )

        scenario_hash = (
            canonical_sha(
                deterministic
            )
        )

        record[
            "deterministic_sha256"
        ] = scenario_hash

        scenario_hashes.append(
            scenario_hash
        )

        stream.write(
            json.dumps(
                record,
                sort_keys=True,
                allow_nan=False,
            )
            +
            "\n"
        )

        stream.flush()

        print(
            "   targets(primary/all)=",
            len(primary_indices),
            "/",
            len(all_current_indices),
            "| associated=",
            len(tracks),
            "| Hough=",
            method_info[
                "Multidimensional_Hough"
            ][
                "prediction_count"
            ],
            "| ms=",
            round(
                scenario_runtime_ms,
                1,
            ),
            flush=True,
        )


formal_runtime_s = (
    time.perf_counter()
    -
    formal_start
)


# ============================================================
# Aggregate final formal metrics.
# ============================================================

primary_final = {
    method:
        finalize_aggregate(agg)
    for method, agg
    in primary_aggregates.items()
}

supplementary_final = {
    method:
        finalize_aggregate(agg)
    for method, agg
    in supplementary_aggregates.items()
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

    "measured_fmcw":
        False,

    "prediction_horizons_s":
        list(HORIZONS),

    "primary_evaluation": {
        "target_policy":
            "WOMD tracks_to_predict; evaluator metadata only; "
            "accessed after prediction; current-valid real "
            "non-target actors are evaluator-ignored rather "
            "than counted as false predictions",

        "metrics":
            primary_final,
    },

    "supplementary_evaluation": {
        "target_policy":
            "all current-valid non-SDC "
            "actors",

        "metrics":
            supplementary_final,
    },

    "aggregation_semantics": {
        "ADE": (
            "micro mean over every "
            "available matched "
            "actor-horizon error"
        ),

        "FDE": (
            "micro mean of evaluator-defined "
            "final error, one per matched track"
        ),

        "reconstruction": (
            "micro TP/FP/FN pooled over the 120 frozen scenarios; "
            "for the primary tracks_to_predict view, predictions "
            "matched to current-valid real non-target actors are "
            "ignored rather than counted as false positives"
        ),

        "horizon_endpoint_miss": (
            "sum endpoint misses / "
            "sum eligible truth tracks"
        ),

        "class_slices": (
            "truth class; matched forecast "
            "errors plus class-specific "
            "recall and endpoint miss"
        ),
    },

    "runtime": {
        "formal_total_s":
            formal_runtime_s,

        "scenario_median_ms":
            statistics.median(
                scenario_total_runtimes
            ),

        "scenario_p95_ms":
            percentile95(
                scenario_total_runtimes
            ),

        "scenario_max_ms":
            max(
                scenario_total_runtimes
            ),
    },

    "temporal_alignment": {
        "prediction_anchor":
            "global_WOMD_current_time_anchor",
        "terminated_tracks_forecast":
            False,
        "active_stale_tracks":
            "causally_open_loop_propagated_to_global_anchor",
        "prediction_timestamp_contract":
            "timestamp_s ~= anchor_timestamp_s + horizon_s",
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
        str(PER_SCENARIO),

    "per_scenario_artifact_sha256":
        sha256_file(
            PER_SCENARIO
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
    "FORMAL EVALUATION ====="
)

print(
    "formal scenarios             = 120"
)

print(
    "formal manifest SHA256       =",
    EXPECTED_FORMAL_SHA,
)

print(
    "primary targets              = "
    "WOMD tracks_to_predict"
)

print(
    "tracks_to_predict algorithm input = NO"
)

print(
    "future truth algorithm input = NO"
)

print(
    "future truth assignment      = NO"
)

print(
    "full degraded Stage2         = YES"
)

print()

for method in sorted(
    primary_final
):
    m = primary_final[method]

    print(
        f"{method:28s}"
        f" ADE={m['ade_m']:.6f}"
        f" FDE={m['fde_m']:.6f}"
        f" recall={m['reconstruction_recall']:.6f}"
        f" F1={m['reconstruction_f1']:.6f}"
    )

print()
print(
    "formal runtime s             =",
    round(
        formal_runtime_s,
        3,
    ),
)

print(
    "deterministic run SHA256     =",
    deterministic_run_sha,
)

print(
    "report                       =",
    REPORT,
)

print(
    "STATUS = PASS"
)
