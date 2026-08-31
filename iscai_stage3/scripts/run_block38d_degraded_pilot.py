from __future__ import annotations

import ast
from collections import Counter
import dataclasses
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
import statistics
import time

from waymo_open_dataset.protos import scenario_pb2

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
    FALSE_ALARM,
    TRUE_DETECTION,
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
    deterministic_manifest_order,
    read_motion_scenario,
    read_validation_manifest,
    sha256_file,
)


ROOT = Path("/home/agni/waymo")
STAGE3 = ROOT / "iscai_stage3"

MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
      "selected_validation.jsonl"
)

PAIRED = (
    ROOT
    / "data/paired_womd_lidar_v1_3_0"
)

STAGE2_PROFILE_SOURCE = (
    ROOT
    / "iscai_stage2/scripts/"
      "run_stage2_degraded_scene_smoke.py"
)

EXPECTED_MANIFEST_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

PILOT_COUNT = 12

HORIZONS = (
    0.1,
    0.3,
    0.5,
    1.0,
)

REPORT = (
    STAGE3
    / "reports/block38d_degraded_pilot.json"
)

PILOT_MANIFEST = (
    STAGE3
    / "artifacts/block38d/"
      "pilot_candidates.jsonl"
)


def load_frozen_stage2_configs():
    """
    Evaluate only top-level assignment expressions
    from the already-frozen Stage2 Block-3D script.

    The Stage2 script itself is NOT imported/executed,
    so its smoke experiment and reports are not rerun.

    This prevents accidental manual reconstruction of
    FalseAlarmConfig measurement-volume parameters.
    """

    source = STAGE2_PROFILE_SOURCE.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(STAGE2_PROFILE_SOURCE),
    )

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
        value_node = None

        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(
                    node.targets[0],
                    ast.Name,
                )
            ):
                name = node.targets[0].id
                value_node = node.value

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            if isinstance(
                node.target,
                ast.Name,
            ):
                name = node.target.id
                value_node = node.value

        if (
            name is None
            or value_node is None
        ):
            continue

        try:
            value = eval(
                compile(
                    ast.Expression(
                        value_node
                    ),
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
            continue

        namespace[name] = value

    if "CLEAN_CONFIG" not in namespace:
        raise RuntimeError(
            "Frozen Stage2 CLEAN_CONFIG "
            "could not be extracted."
        )

    if "DEGRADED_CONFIG" not in namespace:
        raise RuntimeError(
            "Frozen Stage2 DEGRADED_CONFIG "
            "could not be extracted."
        )

    return (
        namespace["CLEAN_CONFIG"],
        namespace["DEGRADED_CONFIG"],
    )


def locate_stage3_symbol(
    name: str,
):
    src = (
        STAGE3
        / "src/iscai_stage3"
    )

    matches = []

    for path in src.rglob("*.py"):
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
            f"Expected one Stage3 "
            f"definition of {name}; "
            f"found {matches}"
        )

    path = matches[0]

    relative = (
        path
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


HoughConfig = locate_stage3_symbol(
    "HoughConfig"
)
HoughFrameContext = (
    locate_stage3_symbol(
        "HoughFrameContext"
    )
)
run_multidimensional_hough = (
    locate_stage3_symbol(
        "run_multidimensional_hough"
    )
)
predict_hough = locate_stage3_symbol(
    "predict_hough"
)


def percentile95(values):
    if not values:
        return None

    ordered = sorted(
        float(x)
        for x in values
    )

    index = max(
        0,
        math.ceil(
            0.95 * len(ordered)
        ) - 1,
    )

    return ordered[index]


def actor_class_name(
    object_type: int,
) -> str:
    try:
        return (
            scenario_pb2.Track
            .ObjectType
            .Name(int(object_type))
        )
    except Exception:
        return (
            f"TYPE_UNKNOWN_{object_type}"
        )


def anchor_class_inventory(
    scenario,
):
    """
    Current-time only. No future state accessed.
    """

    anchor = int(
        scenario.current_time_index
    )

    result = Counter()

    for index, track in enumerate(
        scenario.tracks
    ):
        if index == (
            scenario.sdc_track_index
        ):
            continue

        if anchor >= len(track.states):
            continue

        if not track.states[
            anchor
        ].valid:
            continue

        result[
            actor_class_name(
                track.object_type
            )
        ] += 1

    return dict(
        sorted(result.items())
    )


def truth_status_counts(
    degraded,
):
    """
    Detection sidecars are inspected only AFTER
    all algorithms finish.

    This is Stage2 measurement-generation truth,
    not WOMD future trajectory truth.
    """

    strings = Counter()

    def walk(value):
        if dataclasses.is_dataclass(
            value
        ):
            walk(
                dataclasses.asdict(
                    value
                )
            )

        elif isinstance(
            value,
            dict,
        ):
            for item in value.values():
                walk(item)

        elif isinstance(
            value,
            (tuple, list),
        ):
            for item in value:
                walk(item)

        elif isinstance(
            value,
            str,
        ):
            strings[value] += 1

    for sidecar in (
        degraded.truth_sidecars
    ):
        walk(sidecar)

    missed = sum(
        count
        for token, count
        in strings.items()
        if "miss" in token.lower()
    )

    return {
        "true_detections":
            strings[TRUE_DETECTION],
        "false_alarms":
            strings[FALSE_ALARM],
        "missed_detection_statuses":
            missed,
        "status_tokens": {
            key: value
            for key, value
            in sorted(strings.items())
            if (
                "detect" in key.lower()
                or
                "false" in key.lower()
                or
                "miss" in key.lower()
            )
        },
    }


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
        scenario.current_time_index
        +
        1
    )

    if len(dynamic_frames) != expected:
        raise RuntimeError(
            "Unexpected causal headlamp "
            "frame count."
        )

    if any(
        frame is None
        for frame in dynamic_frames
    ):
        raise ValueError(
            "Unavailable causal "
            "headlamp frame."
        )

    timestamps = tuple(
        float(value)
        for value
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
            T_Ht_from_W_by_frame=(
                tuple(
                    frame.T_Ht_from_W
                    for frame
                    in dynamic_frames
                )
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
        transforms,
        kalman,
        hough,
    )


def run_state_method(
    *,
    tracks,
    transforms,
    estimator,
    predictor,
):
    start = time.perf_counter()

    predictions = 0
    rejected = Counter()

    for track in tracks:
        try:
            observations = (
                cartesianize_associated_track(
                    track,
                    context=transforms,
                )
            )

            state = estimator(
                observations
            )

            predictor(
                state,
                horizons_s=HORIZONS,
            )

            predictions += 1

        except ValueError as exc:
            rejected[
                str(exc)
            ] += 1

    runtime_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    return {
        "prediction_count":
            predictions,
        "rejected_track_count":
            sum(rejected.values()),
        "rejection_reasons":
            dict(rejected),
        "model_runtime_ms":
            runtime_ms,
    }


def run_kalman_method(
    *,
    tracks,
    context,
):
    config = KalmanConfig()

    start = time.perf_counter()

    predictions = 0
    rejected = Counter()

    for track in tracks:
        try:
            result = (
                filter_associated_track(
                    track,
                    context=context,
                    config=config,
                )
            )

            predict_kalman(
                track_id=result.track_id,
                state=result.current_state,
                horizons_s=HORIZONS,
                config=config,
            )

            predictions += 1

        except ValueError as exc:
            rejected[
                str(exc)
            ] += 1

    runtime_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    return {
        "prediction_count":
            predictions,
        "rejected_track_count":
            sum(rejected.values()),
        "rejection_reasons":
            dict(rejected),
        "model_runtime_ms":
            runtime_ms,
    }


def run_imm_method(
    *,
    tracks,
    context,
):
    config = IMMConfig()

    start = time.perf_counter()

    predictions = 0
    rejected = Counter()

    for track in tracks:
        try:
            result = (
                filter_associated_track_imm(
                    track,
                    context=context,
                    config=config,
                )
            )

            predict_imm(
                track_id=result.track_id,
                states=(
                    result
                    .current_mode_states
                ),
                mode_probabilities=(
                    result
                    .current_mode_probabilities
                ),
                horizons_s=HORIZONS,
                config=config,
            )

            predictions += 1

        except ValueError as exc:
            rejected[
                str(exc)
            ] += 1

    runtime_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    return {
        "prediction_count":
            predictions,
        "rejected_track_count":
            sum(rejected.values()),
        "rejection_reasons":
            dict(rejected),
        "model_runtime_ms":
            runtime_ms,
    }


def run_hough_method(
    *,
    sequence,
    context,
):
    config = HoughConfig()

    start = time.perf_counter()

    result = (
        run_multidimensional_hough(
            sequence,
            context=context,
            config=config,
        )
    )

    predictions = 0
    rejected = Counter()

    for track in result.tracks:
        try:
            predict_hough(
                track,
                horizons_s=HORIZONS,
            )

            predictions += 1

        except ValueError as exc:
            rejected[
                str(exc)
            ] += 1

    runtime_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    if bool(
        getattr(
            result,
            "truth_used",
            False,
        )
    ):
        raise RuntimeError(
            "Hough used truth."
        )

    if bool(
        getattr(
            result,
            "estimated_association_used",
            False,
        )
    ):
        raise RuntimeError(
            "Hough used estimated "
            "association."
        )

    if bool(
        getattr(
            result,
            "future_information_used",
            False,
        )
    ):
        raise RuntimeError(
            "Hough used future "
            "information."
        )

    return {
        "prediction_count":
            predictions,
        "hough_track_count":
            len(result.tracks),
        "rejected_track_count":
            sum(rejected.values()),
        "rejection_reasons":
            dict(rejected),
        "model_runtime_ms":
            runtime_ms,
    }


def run_one_scenario(
    *,
    row,
    compact_offset,
    clean_config,
    degraded_config,
):
    scenario_start = (
        time.perf_counter()
    )

    load_start = time.perf_counter()

    scenario = read_motion_scenario(
        row,
        paired_root=PAIRED,
        compact_record_offset=(
            compact_offset
        ),
    )

    load_ms = (
        time.perf_counter()
        -
        load_start
    ) * 1000.0

    if (
        scenario.scenario_id
        !=
        row.scenario_id
    ):
        raise RuntimeError(
            "Scenario identity mismatch."
        )

    if (
        scenario.current_time_index
        !=
        10
    ):
        raise RuntimeError(
            "Unexpected WOMD anchor."
        )

    classes = (
        anchor_class_inventory(
            scenario
        )
    )

    sensor_start = (
        time.perf_counter()
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
            scenario_id=(
                scenario.scenario_id
            ),
            scene=degraded,
        )
    )

    sensor_ms = (
        time.perf_counter()
        -
        sensor_start
    ) * 1000.0

    if len(sequence.frames) != 11:
        raise RuntimeError(
            "Expected 11 causal "
            "algorithm frames."
        )

    if sequence.measured_fmcw:
        raise RuntimeError(
            "Measured FMCW incorrectly "
            "claimed."
        )

    input_before = (
        degraded_algorithm_sha256(
            degraded
        )
    )

    detection_count = sum(
        len(frame.detections)
        for frame in sequence.frames
    )

    (
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

    tracks = tuple(
        associated.tracks
    )

    methods = {}

    methods["CV"] = (
        run_state_method(
            tracks=tracks,
            transforms=transforms,
            estimator=(
                estimate_causal_cv_state
            ),
            predictor=predict_cv,
        )
    )

    methods["CA"] = (
        run_state_method(
            tracks=tracks,
            transforms=transforms,
            estimator=(
                estimate_causal_ca_state
            ),
            predictor=predict_ca,
        )
    )

    methods["CTRV"] = (
        run_state_method(
            tracks=tracks,
            transforms=transforms,
            estimator=(
                estimate_causal_ctrv_state
            ),
            predictor=predict_ctrv,
        )
    )

    methods["Kalman_EKF"] = (
        run_kalman_method(
            tracks=tracks,
            context=kalman_context,
        )
    )

    methods["IMM_CV_CA_CTRV"] = (
        run_imm_method(
            tracks=tracks,
            context=kalman_context,
        )
    )

    methods[
        "Multidimensional_Hough"
    ] = run_hough_method(
        sequence=sequence,
        context=hough_context,
    )

    input_after = (
        degraded_algorithm_sha256(
            degraded
        )
    )

    if input_after != input_before:
        raise RuntimeError(
            "Algorithm-facing Stage2 "
            "input mutated."
        )

    # Only AFTER every method finished.
    detection_truth = (
        truth_status_counts(
            degraded
        )
    )

    scenario_ms = (
        time.perf_counter()
        -
        scenario_start
    ) * 1000.0

    for name, info in methods.items():
        if name != (
            "Multidimensional_Hough"
        ):
            info[
                "association_runtime_ms"
            ] = association_ms

            info[
                "pipeline_runtime_ms"
            ] = (
                association_ms
                +
                info[
                    "model_runtime_ms"
                ]
            )

    return {
        "scenario_id":
            scenario.scenario_id,

        "pipeline_success":
            True,

        "anchor_class_inventory":
            classes,

        "algorithm_frames":
            len(sequence.frames),

        "algorithm_detection_count":
            detection_count,

        "stage2_detection_truth":
            detection_truth,

        "shared_associated_tracks":
            len(tracks),

        "input_sha256_before":
            input_before,

        "input_sha256_after":
            input_after,

        "input_immutable":
            input_before
            ==
            input_after,

        "methods":
            methods,

        "runtime_ms": {
            "motion_load":
                load_ms,
            "stage1_stage2_bridge":
                sensor_ms,
            "estimated_association":
                association_ms,
            "scenario_total":
                scenario_ms,
        },
    }


clean_config, degraded_config = (
    load_frozen_stage2_configs()
)

if (
    degraded_config.gaussian_seed
    !=
    20260810
):
    raise RuntimeError(
        "Frozen Gaussian seed changed."
    )

if (
    degraded_config.detection_seed
    !=
    20260811
):
    raise RuntimeError(
        "Frozen detection seed changed."
    )

if (
    degraded_config.false_alarms
    is None
):
    raise RuntimeError(
        "Full degraded pilot must "
        "enable false alarms."
    )


manifest_sha = sha256_file(
    MANIFEST
)

if (
    manifest_sha
    !=
    EXPECTED_MANIFEST_SHA
):
    raise RuntimeError(
        "Frozen validation manifest "
        "SHA changed."
    )

rows = read_validation_manifest(
    MANIFEST
)

if len(rows) != 44097:
    raise RuntimeError(
        "Frozen validation scenario "
        "count changed."
    )

offsets = (
    build_compact_motion_offset_index(
        rows
    )
)

ordered = (
    deterministic_manifest_order(
        rows
    )
)

candidates = tuple(
    ordered[:PILOT_COUNT]
)

if len(candidates) != 12:
    raise RuntimeError(
        "Pilot candidate count changed."
    )


# ============================================================
# Freeze candidate list BEFORE running any model.
# ============================================================

PILOT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with PILOT_MANIFEST.open(
    "w",
    encoding="utf-8",
) as stream:
    for rank, row in enumerate(
        candidates,
        start=1,
    ):
        record = {
            "rank": rank,
            "scenario_id":
                row.scenario_id,
            "selection_hash":
                row.selection_hash,
            "source_shard":
                row.source_shard,
            "compact_record_offset":
                offsets[
                    row.scenario_id
                ],
            "selection_basis":
                "ascending_sha256_scenario_id",
            "performance_used":
                False,
            "future_used":
                False,
        }

        stream.write(
            json.dumps(
                record,
                sort_keys=True,
            )
            +
            "\n"
        )


pilot_manifest_sha = (
    sha256_file(
        PILOT_MANIFEST
    )
)


# ============================================================
# Run pilot.
# ============================================================

scenario_reports = []

for index, row in enumerate(
    candidates,
    start=1,
):
    print(
        f"[{index:02d}/{PILOT_COUNT}] "
        f"{row.scenario_id}",
        flush=True,
    )

    try:
        item = run_one_scenario(
            row=row,
            compact_offset=(
                offsets[
                    row.scenario_id
                ]
            ),
            clean_config=clean_config,
            degraded_config=(
                degraded_config
            ),
        )

        print(
            "  detections =",
            item[
                "algorithm_detection_count"
            ],
            "| associated =",
            item[
                "shared_associated_tracks"
            ],
            "| runtime_ms =",
            round(
                item["runtime_ms"][
                    "scenario_total"
                ],
                3,
            ),
            flush=True,
        )

        print(
            "  predictions =",
            {
                method:
                    info[
                        "prediction_count"
                    ]
                for method, info
                in item[
                    "methods"
                ].items()
            },
            flush=True,
        )

    except Exception as exc:
        item = {
            "scenario_id":
                row.scenario_id,
            "pipeline_success":
                False,
            "error_type":
                type(exc).__name__,
            "error":
                str(exc),
        }

        print(
            "  PIPELINE FAILURE:",
            type(exc).__name__,
            str(exc),
            flush=True,
        )

    scenario_reports.append(
        item
    )


successful = [
    item
    for item in scenario_reports
    if item[
        "pipeline_success"
    ]
]

failures = [
    item
    for item in scenario_reports
    if not item[
        "pipeline_success"
    ]
]


# ============================================================
# Pilot-only aggregates.
# NO trajectory-performance metrics.
# ============================================================

class_totals = Counter()

for item in successful:
    class_totals.update(
        item[
            "anchor_class_inventory"
        ]
    )


method_names = (
    "CV",
    "CA",
    "CTRV",
    "Kalman_EKF",
    "IMM_CV_CA_CTRV",
    "Multidimensional_Hough",
)

method_summary = {}

for method in method_names:
    counts = [
        item["methods"][method][
            "prediction_count"
        ]
        for item in successful
    ]

    runtimes = [
        item["methods"][method][
            "model_runtime_ms"
        ]
        for item in successful
    ]

    method_summary[
        method
    ] = {
        "prediction_count_total":
            sum(counts),

        "scenarios_with_predictions":
            sum(
                count > 0
                for count in counts
            ),

        "scenarios_with_zero_predictions":
            sum(
                count == 0
                for count in counts
            ),

        "prediction_count_median":
            (
                statistics.median(
                    counts
                )
                if counts
                else None
            ),

        "model_runtime_ms_median":
            (
                statistics.median(
                    runtimes
                )
                if runtimes
                else None
            ),

        "model_runtime_ms_p95":
            percentile95(
                runtimes
            ),
    }


scenario_runtime = [
    item["runtime_ms"][
        "scenario_total"
    ]
    for item in successful
]

detection_counts = [
    item[
        "algorithm_detection_count"
    ]
    for item in successful
]

association_counts = [
    item[
        "shared_associated_tracks"
    ]
    for item in successful
]


status = (
    "PASS"
    if (
        len(successful)
        ==
        PILOT_COUNT
        and all(
            item["input_immutable"]
            for item in successful
        )
    )
    else
    "PILOT_COMPLETED_WITH_FAILURES"
)


formal_config = json.loads(
    (
        STAGE3
        / "configs/"
          "stage3_real_validation.json"
    ).read_text(
        encoding="utf-8"
    )
)

if (
    formal_config[
        "formal_validation"
    ]["N"]
    is not None
):
    raise RuntimeError(
        "Formal N was frozen before "
        "the 3.8D pilot."
    )


report = {
    "stage": 3,
    "block": "3.8D",
    "status": status,

    "purpose": (
        "deterministic runtime and "
        "eligibility pilot only"
    ),

    "formal_performance_metrics":
        False,

    "future_womd_truth_accessed":
        False,

    "formal_N":
        None,

    "selection": {
        "candidate_count":
            PILOT_COUNT,
        "policy":
            "ascending_sha256_scenario_id",
        "performance_based":
            False,
        "future_based":
            False,
        "pilot_manifest":
            str(PILOT_MANIFEST),
        "pilot_manifest_sha256":
            pilot_manifest_sha,
    },

    "frozen_stage2_profile": {
        "source":
            str(
                STAGE2_PROFILE_SOURCE
            ),

        "source_sha256":
            sha256_file(
                STAGE2_PROFILE_SOURCE
            ),

        "clean_config":
            dataclasses.asdict(
                clean_config
            ),

        "degraded_config":
            dataclasses.asdict(
                degraded_config
            ),

        "profile_semantics": (
            "Gaussian measurement corruption "
            "+ SNR-dependent missed detections "
            "+ Poisson false alarms + R_t"
        ),
    },

    "pilot_result": {
        "successful_scenarios":
            len(successful),
        "failed_scenarios":
            len(failures),

        "anchor_class_totals":
            dict(
                sorted(
                    class_totals.items()
                )
            ),

        "scenario_runtime_ms": {
            "median":
                (
                    statistics.median(
                        scenario_runtime
                    )
                    if scenario_runtime
                    else None
                ),
            "p95":
                percentile95(
                    scenario_runtime
                ),
            "max":
                (
                    max(
                        scenario_runtime
                    )
                    if scenario_runtime
                    else None
                ),
        },

        "algorithm_detection_count": {
            "median":
                (
                    statistics.median(
                        detection_counts
                    )
                    if detection_counts
                    else None
                ),
            "min":
                (
                    min(
                        detection_counts
                    )
                    if detection_counts
                    else None
                ),
            "max":
                (
                    max(
                        detection_counts
                    )
                    if detection_counts
                    else None
                ),
        },

        "associated_track_count": {
            "median":
                (
                    statistics.median(
                        association_counts
                    )
                    if association_counts
                    else None
                ),
            "min":
                (
                    min(
                        association_counts
                    )
                    if association_counts
                    else None
                ),
            "max":
                (
                    max(
                        association_counts
                    )
                    if association_counts
                    else None
                ),
        },

        "methods":
            method_summary,
    },

    "scenarios":
        scenario_reports,
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
    "===== Stage3 Block 3.8D "
    "degraded pilot ====="
)

print(
    "pilot scenarios              =",
    PILOT_COUNT,
)

print(
    "successful scenarios         =",
    len(successful),
)

print(
    "failed scenarios             =",
    len(failures),
)

print(
    "pilot manifest SHA256        =",
    pilot_manifest_sha,
)

print(
    "selection performance-based  = NO"
)

print(
    "selection future-based       = NO"
)

print(
    "WOMD future truth accessed   = NO"
)

print(
    "full degraded profile        = YES"
)

print(
    "anchor class totals          =",
    dict(
        sorted(
            class_totals.items()
        )
    ),
)

if scenario_runtime:
    print(
        "scenario runtime median ms  =",
        round(
            statistics.median(
                scenario_runtime
            ),
            3,
        ),
    )

    print(
        "scenario runtime p95 ms     =",
        round(
            percentile95(
                scenario_runtime
            ),
            3,
        ),
    )

for method in method_names:
    info = method_summary[
        method
    ]

    print(
        f"{method:28s} "
        f"scenes>0 = "
        f"{info['scenarios_with_predictions']}"
        f"/{len(successful)}"
        f" | total predictions = "
        f"{info['prediction_count_total']}"
    )

print(
    "formal performance metrics   = NO"
)

print(
    "formal N                     = NOT FROZEN"
)

print(
    "report                       =",
    REPORT,
)

print(
    "STATUS =",
    status,
)

if status != "PASS":
    raise SystemExit(2)
