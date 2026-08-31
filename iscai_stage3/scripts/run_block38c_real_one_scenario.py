from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import get_type_hints

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.clean_scene import (
    CleanObservationConfig,
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

from iscai_stage2.observations.womd_ideal_adapter import (
    build_causal_dynamic_headlamp_frames,
    build_real_ideal_observation_scene,
)

from iscai_stage3.observations import (
    algorithm_sequence_from_degraded_scene,
)


ROOT = Path("/home/agni/waymo")
STAGE3 = ROOT / "iscai_stage3"
SRC = STAGE3 / "src/iscai_stage3"

SID = "b85e1bd6cc8e74c0"

MOTION = (
    ROOT
    / "data/paired_womd_lidar_v1_3_0/"
      "validation/motion/"
      "paired-from-validation.tfrecord-00000-of-00150"
)

REPORT = (
    STAGE3
    / "reports/block38c_real_one_scenario_integration.json"
)

HORIZONS = (
    0.1,
    0.3,
    0.5,
    1.0,
)


# ============================================================
# Stage-2 integration-only profile
# ============================================================

CLEAN_CONFIG = CleanObservationConfig(
    sensing_snr_db=20.0,
    azimuth_std_rad=math.radians(1.0),
    elevation_std_rad=math.radians(1.0),
)

INTEGRATION_DEGRADED_CONFIG = (
    DegradedObservationConfig(
        gaussian_seed=20260830,
        detection_seed=20260831,
        detection_probability=(
            DetectionProbabilityConfig(
                snr_midpoint_db=0.0,
                transition_width_db=1.0,
                pd_floor=1.0,
                pd_ceiling=1.0,
            )
        ),
        false_alarms=None,
    )
)


# ============================================================
# Deterministic clean-Stage3 symbol resolver
#
# This avoids guessing package re-export locations while
# resolving only definitions physically present under the
# current clean iscai_stage3/src tree.
# ============================================================

_SYMBOL_CACHE = {}


def locate_symbol(name: str):
    if name in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[name]

    matches = []

    for path in sorted(
        SRC.rglob("*.py")
    ):
        if "__pycache__" in path.parts:
            continue

        source = path.read_text(
            encoding="utf-8"
        )

        try:
            tree = ast.parse(
                source,
                filename=str(path),
            )
        except SyntaxError as exc:
            raise RuntimeError(
                f"Syntax error in active Stage3 source: "
                f"{path}: {exc}"
            ) from exc

        for node in tree.body:
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ) and node.name == name:
                matches.append(path)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one active Stage3 "
            f"definition of {name!r}; "
            f"found {len(matches)}: "
            f"{[str(x) for x in matches]}"
        )

    path = matches[0]

    relative = path.relative_to(
        STAGE3 / "src"
    ).with_suffix("")

    module_name = ".".join(
        relative.parts
    )

    module = importlib.import_module(
        module_name
    )

    value = getattr(
        module,
        name,
    )

    _SYMBOL_CACHE[name] = value

    return value


def call_supported(
    function,
    *args,
    **kwargs,
):
    signature = inspect.signature(
        function
    )

    accepts_var_kwargs = any(
        parameter.kind
        is inspect.Parameter.VAR_KEYWORD
        for parameter
        in signature.parameters.values()
    )

    if accepts_var_kwargs:
        accepted = kwargs
    else:
        accepted = {
            key: value
            for key, value
            in kwargs.items()
            if key
            in signature.parameters
        }

    return function(
        *args,
        **accepted,
    )


def default_config_for(
    function,
):
    signature = inspect.signature(
        function
    )

    parameter = (
        signature.parameters.get(
            "config"
        )
    )

    if parameter is None:
        return None

    if (
        parameter.default
        is not inspect.Parameter.empty
    ):
        return parameter.default

    try:
        hints = get_type_hints(
            function
        )
    except Exception:
        hints = {}

    config_type = hints.get(
        "config"
    )

    if isinstance(
        config_type,
        type,
    ):
        return config_type()

    annotation = parameter.annotation

    if isinstance(
        annotation,
        str,
    ):
        simple_name = (
            annotation
            .split("[")[0]
            .split(".")[-1]
            .strip()
        )

        config_type = locate_symbol(
            simple_name
        )

        return config_type()

    raise RuntimeError(
        f"Cannot construct config for "
        f"{function.__name__}"
    )


# ============================================================
# JSON helpers
# ============================================================

def jsonable(value):
    if dataclasses.is_dataclass(
        value
    ):
        return jsonable(
            dataclasses.asdict(value)
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): jsonable(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (tuple, list),
    ):
        return [
            jsonable(item)
            for item in value
        ]

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(value):
            return None

    return value


# ============================================================
# Common Stage3 geometry
# ============================================================

FrameTransformContext = locate_symbol(
    "FrameTransformContext"
)

KalmanFrameContext = locate_symbol(
    "KalmanFrameContext"
)

HoughFrameContext = locate_symbol(
    "HoughFrameContext"
)


def build_contexts(
    *,
    raw,
    adapted,
):
    dynamic_frames, _ = (
        build_causal_dynamic_headlamp_frames(
            adapted
        )
    )

    if len(dynamic_frames) != (
        raw.current_time_index + 1
    ):
        raise RuntimeError(
            "Dynamic headlamp frame count "
            "does not match causal WOMD length."
        )

    if any(
        frame is None
        for frame in dynamic_frames
    ):
        raise RuntimeError(
            "Integration scenario contains "
            "an unavailable causal headlamp frame."
        )

    timestamps = tuple(
        float(value)
        for value in
        raw.timestamps_seconds[
            : raw.current_time_index + 1
        ]
    )

    transforms = (
        FrameTransformContext(
            T_H0_from_W=(
                adapted.frames
                .T_H0_from_W
            ),
            T_Ht_from_W_by_frame=tuple(
                frame.T_Ht_from_W
                for frame
                in dynamic_frames
            ),
        )
    )

    kalman = KalmanFrameContext(
        transforms=transforms,
        frame_timestamps_s=(
            timestamps
        ),
    )

    hough = HoughFrameContext(
        transforms=transforms,
        frame_timestamps_s=(
            timestamps
        ),
    )

    return (
        timestamps,
        transforms,
        kalman,
        hough,
    )


# ============================================================
# Associated-track helpers
# ============================================================

def associated_tracks_from(
    result,
):
    if hasattr(
        result,
        "tracks",
    ):
        return tuple(
            result.tracks
        )

    if isinstance(
        result,
        (tuple, list),
    ):
        return tuple(result)

    raise RuntimeError(
        "Could not resolve associated tracks "
        "from Stage3 association result."
    )


def observation_sequence_from_cartesian(
    value,
):
    for name in (
        "observations",
        "records",
        "points",
    ):
        if hasattr(
            value,
            name,
        ):
            return tuple(
                getattr(value, name)
            )

    if isinstance(
        value,
        (tuple, list),
    ):
        return tuple(value)

    raise RuntimeError(
        "Could not resolve Cartesian "
        "observation sequence."
    )


def anchor_position_from_cartesian(
    value,
):
    observations = (
        observation_sequence_from_cartesian(
            value
        )
    )

    if not observations:
        raise ValueError(
            "Track has no Cartesian "
            "observations."
        )

    last = observations[-1]

    if not hasattr(
        last,
        "position_H0_m",
    ):
        raise RuntimeError(
            "Cartesian observation lacks "
            "position_H0_m."
        )

    return tuple(
        float(x)
        for x in last.position_H0_m
    )


# ============================================================
# Prediction adaptation
# ============================================================

adapt_prediction_object = locate_symbol(
    "adapt_prediction_object"
)

EvaluationPredictionSet = locate_symbol(
    "EvaluationPredictionSet"
)


def adapt_one_prediction(
    prediction,
    *,
    anchor_position_H0_m,
):
    return call_supported(
        adapt_prediction_object,
        prediction,
        anchor_position_H0_m=(
            anchor_position_H0_m
        ),
    )


def make_prediction_set(
    *,
    method,
    anchor_timestamp_s,
    tracks,
    runtime_ms,
):
    return EvaluationPredictionSet(
        method=method,
        scenario_id=SID,
        anchor_timestamp_s=(
            anchor_timestamp_s
        ),
        tracks=tuple(tracks),
        runtime_ms=float(
            runtime_ms
        ),
    )


# ============================================================
# Classical track-based methods
# ============================================================

cartesianize_associated_track = (
    locate_symbol(
        "cartesianize_associated_track"
    )
)

estimate_causal_cv_state = (
    locate_symbol(
        "estimate_causal_cv_state"
    )
)

estimate_causal_ca_state = (
    locate_symbol(
        "estimate_causal_ca_state"
    )
)

estimate_causal_ctrv_state = (
    locate_symbol(
        "estimate_causal_ctrv_state"
    )
)

predict_cv = locate_symbol(
    "predict_cv"
)

predict_ca = locate_symbol(
    "predict_ca"
)

predict_ctrv = locate_symbol(
    "predict_ctrv"
)


def run_state_baseline(
    *,
    method,
    associated_tracks,
    transforms,
    estimator,
    predictor,
    association_runtime_ms,
    anchor_timestamp_s,
):
    start = time.perf_counter()

    predictions = []
    rejected = []

    for track in associated_tracks:
        try:
            cartesian = (
                call_supported(
                    cartesianize_associated_track,
                    track,
                    context=transforms,
                )
            )

            observations = (
                observation_sequence_from_cartesian(
                    cartesian
                )
            )

            state = estimator(
                observations
            )

            prediction = (
                call_supported(
                    predictor,
                    state,
                    horizons_s=HORIZONS,
                )
            )

            predictions.append(
                adapt_one_prediction(
                    prediction,
                    anchor_position_H0_m=(
                        anchor_position_from_cartesian(
                            cartesian
                        )
                    ),
                )
            )

        except ValueError as exc:
            rejected.append(
                str(exc)
            )

    predictor_runtime_ms = (
        (
            time.perf_counter()
            -
            start
        )
        *
        1000.0
    )

    total_runtime_ms = (
        association_runtime_ms
        +
        predictor_runtime_ms
    )

    result = make_prediction_set(
        method=method,
        anchor_timestamp_s=(
            anchor_timestamp_s
        ),
        tracks=predictions,
        runtime_ms=(
            total_runtime_ms
        ),
    )

    return (
        result,
        {
            "prediction_count":
                len(predictions),
            "rejected_track_count":
                len(rejected),
            "rejection_reasons":
                dict(Counter(rejected)),
            "association_runtime_ms":
                association_runtime_ms,
            "predictor_runtime_ms":
                predictor_runtime_ms,
            "reported_total_runtime_ms":
                total_runtime_ms,
        },
    )


# ============================================================
# EKF
# ============================================================

filter_associated_track = (
    locate_symbol(
        "filter_associated_track"
    )
)

predict_kalman = locate_symbol(
    "predict_kalman"
)

KalmanConfig = locate_symbol(
    "KalmanConfig"
)


def run_kalman(
    *,
    associated_tracks,
    kalman_context,
    transforms,
    association_runtime_ms,
    anchor_timestamp_s,
):
    config = KalmanConfig()

    start = time.perf_counter()

    predictions = []
    rejected = []

    for track in associated_tracks:
        try:
            cartesian = (
                call_supported(
                    cartesianize_associated_track,
                    track,
                    context=transforms,
                )
            )

            filtered = call_supported(
                filter_associated_track,
                track,
                context=kalman_context,
                config=config,
            )

            prediction = call_supported(
                predict_kalman,
                track_id=(
                    filtered.track_id
                ),
                state=(
                    filtered.current_state
                ),
                horizons_s=HORIZONS,
                config=config,
            )

            predictions.append(
                adapt_one_prediction(
                    prediction,
                    anchor_position_H0_m=(
                        anchor_position_from_cartesian(
                            cartesian
                        )
                    ),
                )
            )

        except ValueError as exc:
            rejected.append(
                str(exc)
            )

    predictor_runtime_ms = (
        (
            time.perf_counter()
            -
            start
        )
        *
        1000.0
    )

    total_runtime_ms = (
        association_runtime_ms
        +
        predictor_runtime_ms
    )

    result = make_prediction_set(
        method="Kalman_EKF",
        anchor_timestamp_s=(
            anchor_timestamp_s
        ),
        tracks=predictions,
        runtime_ms=(
            total_runtime_ms
        ),
    )

    return (
        result,
        {
            "prediction_count":
                len(predictions),
            "rejected_track_count":
                len(rejected),
            "rejection_reasons":
                dict(Counter(rejected)),
            "association_runtime_ms":
                association_runtime_ms,
            "predictor_runtime_ms":
                predictor_runtime_ms,
            "reported_total_runtime_ms":
                total_runtime_ms,
        },
    )


# ============================================================
# IMM
# ============================================================

filter_associated_track_imm = (
    locate_symbol(
        "filter_associated_track_imm"
    )
)

predict_imm = locate_symbol(
    "predict_imm"
)

IMMConfig = locate_symbol(
    "IMMConfig"
)


def run_imm(
    *,
    associated_tracks,
    kalman_context,
    transforms,
    association_runtime_ms,
    anchor_timestamp_s,
):
    config = IMMConfig()

    start = time.perf_counter()

    predictions = []
    rejected = []

    for track in associated_tracks:
        try:
            cartesian = (
                call_supported(
                    cartesianize_associated_track,
                    track,
                    context=transforms,
                )
            )

            filtered = call_supported(
                filter_associated_track_imm,
                track,
                context=kalman_context,
                config=config,
            )

            prediction = call_supported(
                predict_imm,
                track_id=(
                    filtered.track_id
                ),
                states=(
                    filtered
                    .current_mode_states
                ),
                mode_probabilities=(
                    filtered
                    .current_mode_probabilities
                ),
                horizons_s=HORIZONS,
                config=config,
            )

            predictions.append(
                adapt_one_prediction(
                    prediction,
                    anchor_position_H0_m=(
                        anchor_position_from_cartesian(
                            cartesian
                        )
                    ),
                )
            )

        except ValueError as exc:
            rejected.append(
                str(exc)
            )

    predictor_runtime_ms = (
        (
            time.perf_counter()
            -
            start
        )
        *
        1000.0
    )

    total_runtime_ms = (
        association_runtime_ms
        +
        predictor_runtime_ms
    )

    result = make_prediction_set(
        method="IMM_CV_CA_CTRV",
        anchor_timestamp_s=(
            anchor_timestamp_s
        ),
        tracks=predictions,
        runtime_ms=(
            total_runtime_ms
        ),
    )

    return (
        result,
        {
            "prediction_count":
                len(predictions),
            "rejected_track_count":
                len(rejected),
            "rejection_reasons":
                dict(Counter(rejected)),
            "association_runtime_ms":
                association_runtime_ms,
            "predictor_runtime_ms":
                predictor_runtime_ms,
            "reported_total_runtime_ms":
                total_runtime_ms,
        },
    )


# ============================================================
# Raw unlabeled Multidimensional Hough
# ============================================================

run_multidimensional_hough = (
    locate_symbol(
        "run_multidimensional_hough"
    )
)

predict_hough = locate_symbol(
    "predict_hough"
)

HoughConfig = locate_symbol(
    "HoughConfig"
)


def run_hough(
    *,
    sequence,
    hough_context,
    anchor_timestamp_s,
):
    config = HoughConfig()

    start = time.perf_counter()

    result = call_supported(
        run_multidimensional_hough,
        sequence,
        context=hough_context,
        config=config,
    )

    predictions = []

    for track in result.tracks:
        prediction = call_supported(
            predict_hough,
            track,
            horizons_s=HORIZONS,
        )

        predictions.append(
            adapt_one_prediction(
                prediction,
                anchor_position_H0_m=(
                    track
                    .anchor_position_H0_m
                ),
            )
        )

    runtime_ms = (
        (
            time.perf_counter()
            -
            start
        )
        *
        1000.0
    )

    prediction_set = (
        make_prediction_set(
            method=(
                "Multidimensional_Hough"
            ),
            anchor_timestamp_s=(
                anchor_timestamp_s
            ),
            tracks=predictions,
            runtime_ms=runtime_ms,
        )
    )

    info = {
        "prediction_count":
            len(predictions),
        "hough_track_count":
            len(result.tracks),
        "accumulator_cells":
            result.accumulator_cells,
        "peak_count":
            len(result.peaks),
        "runtime_ms":
            runtime_ms,
        "truth_used":
            bool(result.truth_used),
        "estimated_association_used":
            bool(
                result
                .estimated_association_used
            ),
        "future_information_used":
            bool(
                result
                .future_information_used
            ),
    }

    return (
        prediction_set,
        info,
    )


# ============================================================
# Evaluator-only WOMD truth
#
# This is intentionally called AFTER every algorithm has
# finished and after the algorithm-input hash check.
# ============================================================

EvaluationConfig = locate_symbol(
    "EvaluationConfig"
)

evaluate_prediction_set = (
    locate_symbol(
        "evaluate_prediction_set"
    )
)


def resolve_womd_truth_builder():
    module = importlib.import_module(
        "iscai_stage3.evaluation.womd_truth"
    )

    candidates = []

    for name, value in inspect.getmembers(
        module,
        inspect.isfunction,
    ):
        if value.__module__ != (
            module.__name__
        ):
            continue

        lowered = name.lower()

        if (
            "build" not in lowered
            or
            "truth" not in lowered
        ):
            continue

        params = set(
            inspect.signature(
                value
            ).parameters
        )

        score = 0

        if "evaluation" in lowered:
            score += 4

        if "womd" in lowered:
            score += 4

        if any(
            "eligible" in p.lower()
            and
            "track" in p.lower()
            for p in params
        ):
            score += 4

        if any(
            "scenario" in p.lower()
            for p in params
        ):
            score += 2

        candidates.append(
            (
                score,
                name,
                value,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No WOMD evaluator truth builder found."
        )

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    best_score = candidates[0][0]

    best = [
        item
        for item in candidates
        if item[0] == best_score
    ]

    if len(best) != 1:
        raise RuntimeError(
            "Ambiguous WOMD truth builders: "
            f"{[(x[1], x[0]) for x in best]}"
        )

    return (
        best[0][1],
        best[0][2],
    )


def build_truth(
    *,
    raw,
    adapted,
    evaluation_config,
):
    name, function = (
        resolve_womd_truth_builder()
    )

    anchor = int(
        raw.current_time_index
    )

    eligible_indices = tuple(
        index
        for index, track
        in enumerate(raw.tracks)
        if (
            index
            !=
            raw.sdc_track_index
            and
            track.states[anchor].valid
        )
    )

    eligible_ids = tuple(
        str(
            raw.tracks[index].id
        )
        for index
        in eligible_indices
    )

    signature = inspect.signature(
        function
    )

    kwargs = {}

    for parameter_name, parameter in (
        signature.parameters.items()
    ):
        lowered = (
            parameter_name.lower()
        )

        if lowered in {
            "scenario",
            "raw_scenario",
            "womd_scenario",
        }:
            kwargs[
                parameter_name
            ] = raw

        elif (
            "eligible"
            in lowered
            and
            "track"
            in lowered
            and
            "indice"
            in lowered
        ):
            kwargs[
                parameter_name
            ] = eligible_indices

        elif (
            "track"
            in lowered
            and
            "indice"
            in lowered
        ):
            kwargs[
                parameter_name
            ] = eligible_indices

        elif (
            "eligible"
            in lowered
            and
            "track"
            in lowered
            and
            "id"
            in lowered
        ):
            kwargs[
                parameter_name
            ] = eligible_ids

        elif lowered in {
            "track_ids",
            "truth_track_ids",
        }:
            kwargs[
                parameter_name
            ] = eligible_ids

        elif lowered in {
            "t_h0_from_w",
        }:
            kwargs[
                parameter_name
            ] = (
                adapted.frames
                .T_H0_from_W
            )

        elif lowered in {
            "anchor_frames",
            "frames",
        }:
            kwargs[
                parameter_name
            ] = adapted.frames

        elif (
            "horizon"
            in lowered
        ):
            kwargs[
                parameter_name
            ] = HORIZONS

        elif (
            "config"
            in lowered
        ):
            kwargs[
                parameter_name
            ] = evaluation_config

        elif lowered == (
            "scenario_id"
        ):
            kwargs[
                parameter_name
            ] = SID

        elif lowered == (
            "anchor_timestamp_s"
        ):
            kwargs[
                parameter_name
            ] = float(
                raw.timestamps_seconds[
                    anchor
                ]
            )

        elif (
            parameter.default
            is
            inspect.Parameter.empty
        ):
            raise RuntimeError(
                "Unknown required WOMD truth "
                f"builder parameter: "
                f"{parameter_name!r} "
                f"in {name}"
            )

    truth = function(
        **kwargs
    )

    return (
        truth,
        name,
        eligible_indices,
    )


# ============================================================
# Main
# ============================================================

raw = read_first_scenario(
    MOTION
)

if raw.scenario_id != SID:
    raise RuntimeError(
        "Canonical first validation scenario changed: "
        f"{raw.scenario_id}"
    )

adapted = (
    adapt_causal_womd_scenario(
        raw
    )
)

ideal = (
    build_real_ideal_observation_scene(
        raw_scenario=raw,
        adapted=adapted,
        include_sdc=False,
    )
)

clean = (
    build_clean_observation_scene(
        ideal_scene=ideal,
        config=CLEAN_CONFIG,
    )
)

degraded = (
    build_degraded_observation_scene(
        clean_scene=clean,
        config=(
            INTEGRATION_DEGRADED_CONFIG
        ),
    )
)

sequence = (
    algorithm_sequence_from_degraded_scene(
        scenario_id=SID,
        scene=degraded,
    )
)

if sequence.measured_fmcw:
    raise RuntimeError(
        "Stage3 integration falsely claims "
        "measured FMCW."
    )

if len(sequence.frames) != 11:
    raise RuntimeError(
        "Expected exactly 11 causal "
        "algorithm-facing frames."
    )

input_hash_before = (
    degraded_algorithm_sha256(
        degraded
    )
)

(
    timestamps,
    transform_context,
    kalman_context,
    hough_context,
) = build_contexts(
    raw=raw,
    adapted=adapted,
)

anchor_timestamp_s = (
    timestamps[-1]
)


# ------------------------------------------------------------
# Shared estimated association
# ------------------------------------------------------------

associate_estimated_gnn = (
    locate_symbol(
        "associate_estimated_gnn"
    )
)

association_config = (
    default_config_for(
        associate_estimated_gnn
    )
)

association_start = (
    time.perf_counter()
)

if association_config is None:
    association_result = (
        associate_estimated_gnn(
            sequence
        )
    )
else:
    association_result = (
        call_supported(
            associate_estimated_gnn,
            sequence,
            config=association_config,
        )
    )

association_runtime_ms = (
    (
        time.perf_counter()
        -
        association_start
    )
    *
    1000.0
)

associated_tracks = (
    associated_tracks_from(
        association_result
    )
)

if not associated_tracks:
    raise RuntimeError(
        "Estimated association produced "
        "zero tracks."
    )


# ------------------------------------------------------------
# Execute every Stage3 classical predictor
# ------------------------------------------------------------

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
    result, info = (
        run_state_baseline(
            method=method,
            associated_tracks=(
                associated_tracks
            ),
            transforms=(
                transform_context
            ),
            estimator=estimator,
            predictor=predictor,
            association_runtime_ms=(
                association_runtime_ms
            ),
            anchor_timestamp_s=(
                anchor_timestamp_s
            ),
        )
    )

    prediction_sets[
        method
    ] = result

    method_info[
        method
    ] = info


kalman_set, kalman_info = (
    run_kalman(
        associated_tracks=(
            associated_tracks
        ),
        kalman_context=(
            kalman_context
        ),
        transforms=(
            transform_context
        ),
        association_runtime_ms=(
            association_runtime_ms
        ),
        anchor_timestamp_s=(
            anchor_timestamp_s
        ),
    )
)

prediction_sets[
    "Kalman_EKF"
] = kalman_set

method_info[
    "Kalman_EKF"
] = kalman_info


imm_set, imm_info = (
    run_imm(
        associated_tracks=(
            associated_tracks
        ),
        kalman_context=(
            kalman_context
        ),
        transforms=(
            transform_context
        ),
        association_runtime_ms=(
            association_runtime_ms
        ),
        anchor_timestamp_s=(
            anchor_timestamp_s
        ),
    )
)

prediction_sets[
    "IMM_CV_CA_CTRV"
] = imm_set

method_info[
    "IMM_CV_CA_CTRV"
] = imm_info


hough_set, hough_info = (
    run_hough(
        sequence=sequence,
        hough_context=(
            hough_context
        ),
        anchor_timestamp_s=(
            anchor_timestamp_s
        ),
    )
)

prediction_sets[
    "Multidimensional_Hough"
] = hough_set

method_info[
    "Multidimensional_Hough"
] = hough_info


# ------------------------------------------------------------
# Gate algorithm-input immutability BEFORE touching GT future
# ------------------------------------------------------------

input_hash_after_prediction = (
    degraded_algorithm_sha256(
        degraded
    )
)

if (
    input_hash_before
    !=
    input_hash_after_prediction
):
    raise RuntimeError(
        "Algorithm-facing Stage2 input changed "
        "during prediction."
    )


required_methods = {
    "CV",
    "CA",
    "CTRV",
    "Kalman_EKF",
    "IMM_CV_CA_CTRV",
    "Multidimensional_Hough",
}

if set(
    prediction_sets
) != required_methods:
    raise RuntimeError(
        "Classical method set mismatch."
    )

for method in sorted(
    required_methods
):
    if (
        method_info[
            method
        ]["prediction_count"]
        <= 0
    ):
        raise RuntimeError(
            f"{method} produced zero "
            "real predictions."
        )


if (
    hough_info[
        "truth_used"
    ]
    or
    hough_info[
        "estimated_association_used"
    ]
    or
    hough_info[
        "future_information_used"
    ]
):
    raise RuntimeError(
        "Hough violated raw-unlabeled "
        "algorithm boundary."
    )


# ------------------------------------------------------------
# Only now construct evaluator-only future truth
# ------------------------------------------------------------

evaluation_config = (
    EvaluationConfig(
        horizons_s=HORIZONS,
        anchor_assignment_gate_m=5.0,
        endpoint_miss_threshold_m=2.0,
        horizon_tolerance_s=1e-6,
        anchor_timestamp_tolerance_s=2e-3,
    )
)

(
    truth,
    truth_builder_name,
    eligible_truth_indices,
) = build_truth(
    raw=raw,
    adapted=adapted,
    evaluation_config=(
        evaluation_config
    ),
)


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

evaluation_reports = {}

for method in sorted(
    prediction_sets
):
    evaluation_reports[
        method
    ] = call_supported(
        evaluate_prediction_set,
        prediction_sets[
            method
        ],
        truth=truth,
        config=evaluation_config,
    )


# ------------------------------------------------------------
# Final immutable-input check
# ------------------------------------------------------------

input_hash_final = (
    degraded_algorithm_sha256(
        degraded
    )
)

if input_hash_final != (
    input_hash_before
):
    raise RuntimeError(
        "Evaluator mutated algorithm-visible "
        "Stage2 input."
    )


report = {
    "status": "PASS",
    "block": (
        "stage3_block38c_real_one_scenario_integration"
    ),

    "scenario_id": SID,

    "purpose": (
        "real pipeline integration gate; "
        "NOT formal performance evaluation"
    ),

    "formal_validation": False,
    "formal_N_frozen": False,

    "source": {
        "dataset":
            "WOMD/WOMD-LiDAR v1.3.0",
        "split":
            "validation",
        "canonical_motion_file":
            str(MOTION),
        "raw_waymo_path_used":
            False,
    },

    "observation_pipeline": {
        "path": (
            "real WOMD -> Stage1 causal adapter -> "
            "Stage2 ideal -> clean -> degraded -> "
            "Stage3 AlgorithmObservationSequence"
        ),
        "measured_fmcw":
            False,
        "algorithm_input_truth_free":
            True,
        "causal_frames":
            len(sequence.frames),
        "stage2_algorithm_sha256_before":
            input_hash_before,
        "stage2_algorithm_sha256_after_prediction":
            input_hash_after_prediction,
        "stage2_algorithm_sha256_final":
            input_hash_final,
        "algorithm_input_immutable":
            (
                input_hash_before
                ==
                input_hash_after_prediction
                ==
                input_hash_final
            ),
    },

    "integration_profile": {
        "formal_degraded_profile":
            False,
        "gaussian_measurement_noise":
            True,
        "measurement_covariance_R_t":
            True,
        "sensing_snr_db":
            20.0,
        "azimuth_std_deg":
            1.0,
        "elevation_std_deg":
            1.0,
        "detection_probability":
            1.0,
        "false_alarms":
            0,
        "reason": (
            "isolate real end-to-end wiring before "
            "formal misses/false-alarm robustness pilot"
        ),
    },

    "common_tracking_frontend": {
        "track_based_methods": [
            "CV",
            "CA",
            "CTRV",
            "Kalman_EKF",
            "IMM_CV_CA_CTRV",
        ],
        "association":
            "estimated_association",
        "shared_associated_track_count":
            len(associated_tracks),
        "association_runtime_ms":
            association_runtime_ms,
    },

    "hough_frontend": {
        "mode":
            "raw_unlabeled_track_before_detect",
        "estimated_association_used":
            False,
    },

    "horizons_s":
        list(HORIZONS),

    "methods":
        method_info,

    "evaluator": {
        "truth_constructed_only_after_all_predictions":
            True,
        "truth_builder":
            truth_builder_name,
        "eligible_truth_track_count":
            len(
                eligible_truth_indices
            ),
        "eligibility_rule": (
            "current-time valid non-SDC; "
            "future validity not used for eligibility"
        ),
        "future_truth_used_for_algorithm":
            False,
    },

    "evaluation_reports": {
        method:
            jsonable(value)
        for method, value
        in evaluation_reports.items()
    },
}

REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.write_text(
    json.dumps(
        jsonable(report),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    +
    "\n",
    encoding="utf-8",
)


print(
    "===== Stage3 Block 3.8C "
    "real integration gate ====="
)

print(
    "scenario                      =",
    SID,
)

print(
    "causal Stage2 frames          =",
    len(sequence.frames),
)

print(
    "algorithm input truth-free    = YES"
)

print(
    "measured FMCW                 = NO"
)

print(
    "integration profile           = "
    "Gaussian noise + R_t, P_D=1, FA=0"
)

print(
    "formal performance run        = NO"
)

print(
    "associated tracks             =",
    len(associated_tracks),
)

for method in sorted(
    required_methods
):
    print(
        f"{method:28s} = "
        f"{method_info[method]['prediction_count']} "
        f"predictions"
    )

print(
    "Hough estimated association   = NO"
)

print(
    "truth built after prediction  = YES"
)

print(
    "input hash before             =",
    input_hash_before,
)

print(
    "input hash after prediction   =",
    input_hash_after_prediction,
)

print(
    "input hash final              =",
    input_hash_final,
)

print(
    "input immutable               =",
    (
        "PASS"
        if (
            input_hash_before
            ==
            input_hash_after_prediction
            ==
            input_hash_final
        )
        else
        "FAIL"
    ),
)

print(
    "formal N frozen               = NO"
)

print(
    "report                        =",
    REPORT,
)

print(
    "STATUS = PASS"
)
