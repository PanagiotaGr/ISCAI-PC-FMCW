from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

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
)
from iscai_stage2.observations.detection import (
    DetectionProbabilityConfig,
)
from iscai_stage2.observations.detection_set import (
    TRUE_DETECTION,
    FalseAlarmConfig,
)
from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)


ROOT = Path(
    "/home/agni/waymo"
)

MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
    "selected_validation.jsonl"
)

TINY_PILOT = (
    ROOT
    / "iscai_stage1/artifacts/stage1a/"
    "manifests/tiny_pilot_validation.jsonl"
)

DISCOVERY_REPORT = (
    ROOT
    / "iscai_stage2/reports/"
    "block3f1_multiclass_extension_discovery.json"
)

REPORT = (
    ROOT
    / "iscai_stage2/reports/"
    "block3f_final_multiclass_coverage_gate.json"
)

MOTION_ROOT = (
    ROOT
    / "data/paired_womd_lidar_v1_3_0/"
    "validation/motion"
)


EXPECTED_MANIFEST_SHA256 = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

REQUIRED_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)

OBJECT_TYPE_TO_CLASS = {
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
}


CLEAN_CONFIG = CleanObservationConfig(
    sensing_snr_db=20.0,
    azimuth_std_rad=math.radians(1.0),
    elevation_std_rad=math.radians(1.0),
)


# Coverage configuration only:
#
# Gaussian corruption ON
# P_D = 1
# false alarms = 0
#
# This ensures class coverage cannot disappear because
# of a stochastic miss. Block 3D separately validates
# the actual degraded mode with misses + false alarms.
COVERAGE_CONFIG = DegradedObservationConfig(
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

    false_alarms=(
        FalseAlarmConfig(
            seed=20260832,

            mean_false_alarms_per_frame=0.0,

            range_min_m=1.0,
            range_max_m=100.0,

            radial_velocity_min_mps=-40.0,
            radial_velocity_max_mps=40.0,

            azimuth_min_rad=-math.pi,
            azimuth_max_rad=math.pi,

            elevation_min_rad=-0.3,
            elevation_max_rad=0.3,

            range_std_m=1.0,
            radial_velocity_std_mps=2.0,

            azimuth_std_rad=math.radians(5.0),
            elevation_std_rad=math.radians(5.0),
        )
    ),
)


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_jsonl(
    path: Path,
) -> list[dict]:
    items = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                items.append(
                    json.loads(line)
                )

    return items


def load_manifest_records_by_index(
    indices: set[int],
) -> dict[int, dict]:
    found = {}

    maximum = max(indices)

    with MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        for index, line in enumerate(
            f,
            start=1,
        ):
            if index in indices:
                found[index] = json.loads(
                    line
                )

            if index >= maximum:
                break

    if set(found) != indices:
        raise RuntimeError(
            "Could not resolve all requested "
            "manifest indices."
        )

    return found


def motion_shard_for(
    manifest_record: dict,
) -> Path:
    source_name = Path(
        manifest_record[
            "source_shard"
        ]
    ).name

    path = (
        MOTION_ROOT
        / f"paired-from-{source_name}"
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing motion shard: {path}"
        )

    return path


def read_scenario(
    record: dict,
):
    path = motion_shard_for(
        record
    )

    offset = int(
        record["record_offset"]
    )

    expected_length = int(
        record["payload_length"]
    )

    with path.open("rb") as f:
        f.seek(offset)

        length_bytes = f.read(8)

        if len(length_bytes) != 8:
            raise RuntimeError(
                "Cannot read TFRecord length."
            )

        length = struct.unpack(
            "<Q",
            length_bytes,
        )[0]

        if length != expected_length:
            raise RuntimeError(
                "Payload-length mismatch."
            )

        if len(f.read(4)) != 4:
            raise RuntimeError(
                "Truncated length CRC."
            )

        payload = f.read(length)

        if len(payload) != length:
            raise RuntimeError(
                "Truncated payload."
            )

        if len(f.read(4)) != 4:
            raise RuntimeError(
                "Truncated payload CRC."
            )

    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(
        payload
    )

    if (
        scenario.scenario_id
        != record["scenario_id"]
    ):
        raise RuntimeError(
            "Scenario-ID mismatch."
        )

    return scenario


def causal_actor_counts(
    scenario,
) -> Counter:
    result = Counter()

    anchor = int(
        scenario.current_time_index
    )

    for track in scenario.tracks:
        class_name = (
            OBJECT_TYPE_TO_CLASS.get(
                int(track.object_type)
            )
        )

        if class_name is None:
            continue

        causal_valid = any(
            state.valid
            for state
            in track.states[
                : anchor + 1
            ]
        )

        if causal_valid:
            result[class_name] += 1

    return result


def build_pipeline(
    scenario,
):
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
            config=CLEAN_CONFIG,
        )
    )

    degraded = (
        build_degraded_observation_scene(
            clean_scene=clean,
            config=COVERAGE_CONFIG,
        )
    )

    return clean, degraded


# ------------------------------------------------------------
# Frozen-manifest integrity
# ------------------------------------------------------------

manifest_sha = sha256_file(
    MANIFEST
)

if manifest_sha != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Frozen validation manifest SHA mismatch."
    )


# ------------------------------------------------------------
# Read and verify deterministic discovery result
# ------------------------------------------------------------

discovery = json.loads(
    DISCOVERY_REPORT.read_text(
        encoding="utf-8"
    )
)

if discovery["status"] != "PASS":
    raise RuntimeError(
        "Block 3F.1 discovery did not PASS."
    )

if (
    discovery["manifest_sha256"]
    != EXPECTED_MANIFEST_SHA256
):
    raise RuntimeError(
        "Block 3F.1 used a different manifest."
    )

semantics = discovery[
    "selection_semantics"
]

if (
    semantics[
        "future_actor_states_used_for_selection"
    ]
    is not False
):
    raise RuntimeError(
        "Future-state selection detected."
    )

if (
    semantics[
        "future_labels_used_for_selection"
    ]
    is not False
):
    raise RuntimeError(
        "Future-label selection detected."
    )

if (
    semantics["cherry_picking"]
    is not False
):
    raise RuntimeError(
        "Discovery was not deterministic."
    )

if (
    semantics[
        "first_qualifying_scenario_per_class"
    ]
    is not True
):
    raise RuntimeError(
        "First-qualifying semantics missing."
    )


# ------------------------------------------------------------
# Original frozen first-3 pilots
# ------------------------------------------------------------

tiny_pilot_records = (
    read_jsonl(
        TINY_PILOT
    )
)

if len(tiny_pilot_records) != 3:
    raise RuntimeError(
        "Expected exactly three frozen "
        "Stage-1 tiny pilots."
    )

original_ids = [
    record["scenario_id"]
    for record in tiny_pilot_records
]

if original_ids != [
    "b85e1bd6cc8e74c0",
    "4d82fec943ddaa44",
    "bbc29ed5e271f29b",
]:
    raise RuntimeError(
        "Frozen first-three pilot ordering changed."
    )


# ------------------------------------------------------------
# Extension indices come ONLY from frozen discovery report
# ------------------------------------------------------------

extension_indices = {}

for class_name in (
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
):
    if class_name not in discovery["found"]:
        raise RuntimeError(
            f"Missing discovered class: {class_name}"
        )

    item = discovery[
        "found"
    ][class_name]

    extension_indices[
        class_name
    ] = int(
        item[
            "manifest_record_index_1_based"
        ]
    )


manifest_extensions = (
    load_manifest_records_by_index(
        set(
            extension_indices.values()
        )
    )
)


extension_records = []

for class_name in (
    "TYPE_CYCLIST",
    "TYPE_PEDESTRIAN",
):
    index = extension_indices[
        class_name
    ]

    record = manifest_extensions[
        index
    ]

    expected_sid = (
        discovery["found"][
            class_name
        ]["scenario_id"]
    )

    if (
        record["scenario_id"]
        != expected_sid
    ):
        raise RuntimeError(
            "Discovery/manifest scenario mismatch."
        )

    extension_records.append(
        {
            **record,
            "_coverage_reason":
                class_name,
            "_manifest_index":
                index,
        }
    )


# ------------------------------------------------------------
# Combined frozen coverage set
#
# First 3 frozen pilots
# +
# first qualifying cyclist
# +
# first qualifying pedestrian
# ------------------------------------------------------------

coverage_records = []

for index, record in enumerate(
    tiny_pilot_records,
    start=1,
):
    coverage_records.append(
        {
            **record,
            "_coverage_reason":
                "ORIGINAL_STAGE1_TINY_PILOT",
            "_manifest_index":
                index,
        }
    )

coverage_records.extend(
    extension_records
)


ids = [
    item["scenario_id"]
    for item in coverage_records
]

if len(ids) != len(set(ids)):
    raise RuntimeError(
        "Duplicate scenario in coverage set."
    )


# ------------------------------------------------------------
# Run actual Stage-2 clean + degraded pipeline
# ------------------------------------------------------------

aggregate_causal_actors = Counter()
aggregate_clean_valid = Counter()
aggregate_degraded_true = Counter()

scenario_reports = []


for record in coverage_records:
    scenario = read_scenario(
        record
    )

    causal_counts = (
        causal_actor_counts(
            scenario
        )
    )

    clean, degraded = (
        build_pipeline(
            scenario
        )
    )

    clean_valid = Counter()

    for actor in clean.actors:
        count = sum(
            record.measurement_valid
            for record in actor.records
        )

        if count > 0:
            clean_valid[
                actor.object_class
            ] += count

    degraded_true = Counter()

    for frame in degraded.frames:
        if (
            frame.association_information_present
            is not False
        ):
            raise RuntimeError(
                "Association information leaked."
            )

        if (
            frame.truth_labels_present
            is not False
        ):
            raise RuntimeError(
                "Truth labels leaked."
            )

        for detection in frame.detections:
            for forbidden in (
                "track_id",
                "object_class",
                "source_type",
                "source_track_id",
            ):
                if hasattr(
                    detection,
                    forbidden,
                ):
                    raise RuntimeError(
                        "Oracle field leaked into "
                        "algorithm detection: "
                        f"{forbidden}"
                    )

    for sidecar in (
        degraded.truth_sidecars
    ):
        for entry in sidecar.entries:
            if (
                entry.source_type
                != TRUE_DETECTION
            ):
                raise RuntimeError(
                    "False alarm found in "
                    "coverage-only mode."
                )

            if (
                entry.source_object_class
                is None
            ):
                raise RuntimeError(
                    "Evaluator true detection "
                    "missing object class."
                )

            degraded_true[
                entry.source_object_class
            ] += 1

    compared_classes = (
        set(clean_valid)
        | set(degraded_true)
    )

    mismatch = {
        class_name: {
            "clean":
                clean_valid[class_name],
            "degraded":
                degraded_true[class_name],
        }
        for class_name in compared_classes
        if (
            clean_valid[class_name]
            != degraded_true[class_name]
        )
    }

    if mismatch:
        raise RuntimeError(
            "P_D=1 coverage-mode mismatch: "
            f"{mismatch}"
        )

    aggregate_causal_actors.update(
        causal_counts
    )

    aggregate_clean_valid.update(
        clean_valid
    )

    aggregate_degraded_true.update(
        degraded_true
    )

    scenario_reports.append(
        {
            "manifest_index_1_based":
                record["_manifest_index"],

            "scenario_id":
                scenario.scenario_id,

            "coverage_reason":
                record["_coverage_reason"],

            "causal_actor_counts":
                dict(
                    sorted(
                        causal_counts.items()
                    )
                ),

            "clean_valid_measurements":
                dict(
                    sorted(
                        clean_valid.items()
                    )
                ),

            "degraded_true_measurements":
                dict(
                    sorted(
                        degraded_true.items()
                    )
                ),
        }
    )


# ------------------------------------------------------------
# Required multiclass gate
# ------------------------------------------------------------

coverage = {}

for class_name in REQUIRED_CLASSES:
    causal_count = (
        aggregate_causal_actors[
            class_name
        ]
    )

    clean_count = (
        aggregate_clean_valid[
            class_name
        ]
    )

    degraded_count = (
        aggregate_degraded_true[
            class_name
        ]
    )

    passed = (
        causal_count > 0
        and clean_count > 0
        and degraded_count > 0
    )

    coverage[class_name] = {
        "causal_actors":
            causal_count,

        "clean_valid_measurements":
            clean_count,

        "degraded_true_measurements":
            degraded_count,

        "pass":
            passed,
    }


missing = [
    class_name
    for class_name in REQUIRED_CLASSES
    if not coverage[
        class_name
    ]["pass"]
]

if missing:
    raise RuntimeError(
        "Final multiclass coverage failed for: "
        + ", ".join(missing)
    )


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

payload = {
    "status": "PASS",

    "block": (
        "stage2_block3f_final_"
        "multiclass_coverage_gate"
    ),

    "frozen_validation_manifest": str(
        MANIFEST
    ),

    "frozen_validation_manifest_sha256":
        manifest_sha,

    "selection_policy": (
        "first_3_frozen_pilots_plus_"
        "first_deterministically_qualifying_"
        "scenario_per_missing_class"
    ),

    "original_pilot_count": 3,
    "extension_count": 2,
    "coverage_scenario_count":
        len(coverage_records),

    "coverage_scenario_ids": ids,

    "extension": {
        "TYPE_CYCLIST": {
            "manifest_index_1_based":
                extension_indices[
                    "TYPE_CYCLIST"
                ],

            "scenario_id":
                discovery["found"][
                    "TYPE_CYCLIST"
                ]["scenario_id"],
        },

        "TYPE_PEDESTRIAN": {
            "manifest_index_1_based":
                extension_indices[
                    "TYPE_PEDESTRIAN"
                ],

            "scenario_id":
                discovery["found"][
                    "TYPE_PEDESTRIAN"
                ]["scenario_id"],
        },
    },

    "coverage_semantics": {
        "full_scenario_actor_presence":
            "not_sufficient",

        "causal_actor_requirement":
            "at_least_one_valid_state_at_or_before_anchor",

        "stage2_qualification":
            "at_least_one_clean_measurement_valid",

        "future_actor_states_used_for_selection":
            False,

        "future_labels_used_for_selection":
            False,

        "cherry_picking":
            False,
    },

    "coverage_mode": {
        "gaussian_noise": True,
        "pd": 1.0,
        "false_alarm_mean_per_frame":
            0.0,

        "purpose":
            "pipeline_class_coverage_only",
    },

    "required_classes": list(
        REQUIRED_CLASSES
    ),

    "coverage": coverage,

    "aggregate_causal_actor_counts":
        dict(
            sorted(
                aggregate_causal_actors.items()
            )
        ),

    "aggregate_clean_valid_measurements":
        dict(
            sorted(
                aggregate_clean_valid.items()
            )
        ),

    "aggregate_degraded_true_measurements":
        dict(
            sorted(
                aggregate_degraded_true.items()
            )
        ),

    "scenarios":
        scenario_reports,

    "algorithm_input_truth_free":
        True,

    "measured_fmcw":
        False,
}


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== Stage2 final multiclass coverage gate ====="
)

print(
    "coverage scenarios =",
    len(coverage_records),
)

print()

for item in scenario_reports:
    print(
        f"record={item['manifest_index_1_based']:2d} "
        f"scenario={item['scenario_id']} "
        f"reason={item['coverage_reason']}"
    )

print()

for class_name in REQUIRED_CLASSES:
    item = coverage[
        class_name
    ]

    print(
        f"{class_name:16s} "
        f"causal_actors="
        f"{item['causal_actors']:4d} "
        f"clean_valid="
        f"{item['clean_valid_measurements']:5d} "
        f"degraded_true="
        f"{item['degraded_true_measurements']:5d} "
        f"PASS={item['pass']}"
    )

print()

print(
    "algorithm truth leakage = NONE"
)

print(
    "future-state selection  = NO"
)

print(
    "cherry-picking          = NO"
)

print(
    "measured FMCW           = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)
