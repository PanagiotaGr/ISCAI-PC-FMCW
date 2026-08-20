from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import struct

from waymo_open_dataset.protos import (
    scenario_pb2,
)

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


MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage2/"
    "reports/block3f_multiclass_pilot_gate.json"
)


# Frozen directly from Stage-1A closure manifest.
PILOTS = (
    {
        "scenario_id": "b85e1bd6cc8e74c0",
        "record_offset": 0,
        "payload_length": 647785,
    },
    {
        "scenario_id": "4d82fec943ddaa44",
        "record_offset": 647801,
        "payload_length": 690167,
    },
    {
        "scenario_id": "bbc29ed5e271f29b",
        "record_offset": 1337984,
        "payload_length": 1115755,
    },
)


REQUIRED_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)


CLEAN_CONFIG = CleanObservationConfig(
    sensing_snr_db=20.0,
    azimuth_std_rad=math.radians(1.0),
    elevation_std_rad=math.radians(1.0),
)


# Coverage mode:
#
# Gaussian corruption remains enabled, but P_D=1 and
# false-alarm rate=0 so a class cannot disappear merely
# because of a random missed-detection draw.
#
# This is a coverage gate, NOT the degraded-performance
# experiment from Block 3D.
COVERAGE_DEGRADED_CONFIG = (
    DegradedObservationConfig(
        gaussian_seed=20260820,
        detection_seed=20260821,

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
                seed=20260822,

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
)


def read_scenario_at_offset(
    *,
    path: Path,
    offset: int,
    expected_payload_length: int,
    expected_scenario_id: str,
):
    """
    Read one frozen TFRecord record by exact Stage-1A offset.

    TFRecord layout:
        uint64 length
        uint32 masked_crc(length)
        payload
        uint32 masked_crc(payload)

    CRC values are skipped here because the exact records were
    already frozen/audited upstream; scenario ID and payload
    length are re-validated.
    """

    with path.open("rb") as f:
        f.seek(offset)

        length_bytes = f.read(8)

        if len(length_bytes) != 8:
            raise RuntimeError(
                f"Cannot read TFRecord length at offset {offset}."
            )

        payload_length = struct.unpack(
            "<Q",
            length_bytes,
        )[0]

        if payload_length != expected_payload_length:
            raise RuntimeError(
                "Frozen payload length mismatch: "
                f"expected={expected_payload_length}, "
                f"actual={payload_length}"
            )

        length_crc = f.read(4)

        if len(length_crc) != 4:
            raise RuntimeError(
                "Truncated TFRecord length CRC."
            )

        payload = f.read(
            payload_length
        )

        if len(payload) != payload_length:
            raise RuntimeError(
                "Truncated TFRecord payload."
            )

        data_crc = f.read(4)

        if len(data_crc) != 4:
            raise RuntimeError(
                "Truncated TFRecord data CRC."
            )

    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(
        payload
    )

    if (
        scenario.scenario_id
        != expected_scenario_id
    ):
        raise RuntimeError(
            "Frozen scenario ID mismatch: "
            f"expected={expected_scenario_id}, "
            f"actual={scenario.scenario_id}"
        )

    return scenario


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
            config=(
                COVERAGE_DEGRADED_CONFIG
            ),
        )
    )

    return ideal, clean, degraded


aggregate_actor_counts = Counter()
aggregate_clean_valid = Counter()
aggregate_degraded_true = Counter()

scenario_reports = []


for pilot in PILOTS:
    scenario = read_scenario_at_offset(
        path=MOTION,

        offset=pilot[
            "record_offset"
        ],

        expected_payload_length=pilot[
            "payload_length"
        ],

        expected_scenario_id=pilot[
            "scenario_id"
        ],
    )

    ideal, clean, degraded = (
        build_pipeline(
            scenario
        )
    )

    actor_counts = Counter(
        actor.object_class
        for actor in clean.actors
    )

    clean_valid = Counter()

    for actor in clean.actors:
        valid_count = sum(
            record.measurement_valid
            for record in actor.records
        )

        clean_valid[
            actor.object_class
        ] += valid_count

    degraded_true = Counter()

    for sidecar in (
        degraded.truth_sidecars
    ):
        for entry in sidecar.entries:
            if (
                entry.source_type
                != TRUE_DETECTION
            ):
                raise RuntimeError(
                    "Coverage mode unexpectedly "
                    "contains a false alarm."
                )

            if (
                entry.source_object_class
                is None
            ):
                raise RuntimeError(
                    "True detection missing "
                    "evaluator class."
                )

            degraded_true[
                entry.source_object_class
            ] += 1

    # With P_D=1 and no false alarms, every clean-valid
    # measurement must survive into degraded truth.
    #
    # Counter may contain explicit zero-count class keys
    # while degraded_true omits those keys entirely.
    # Compare values over the union of classes instead of
    # comparing raw dictionaries.
    compared_classes = (
        set(clean_valid)
        | set(degraded_true)
    )

    mismatched_classes = {
        object_class: {
            "clean_valid":
                clean_valid[object_class],
            "degraded_true":
                degraded_true[object_class],
        }
        for object_class
        in compared_classes
        if (
            clean_valid[object_class]
            != degraded_true[object_class]
        )
    }

    if mismatched_classes:
        raise RuntimeError(
            "Coverage-mode degraded true counts "
            "do not match clean-valid counts: "
            f"{mismatched_classes}"
        )

    aggregate_actor_counts.update(
        actor_counts
    )

    aggregate_clean_valid.update(
        clean_valid
    )

    aggregate_degraded_true.update(
        degraded_true
    )

    scenario_reports.append(
        {
            "scenario_id":
                scenario.scenario_id,

            "anchor_index":
                ideal.anchor_index,

            "actors_by_class":
                dict(
                    sorted(
                        actor_counts.items()
                    )
                ),

            "clean_valid_measurements_by_class":
                dict(
                    sorted(
                        clean_valid.items()
                    )
                ),

            "degraded_true_measurements_by_class":
                dict(
                    sorted(
                        degraded_true.items()
                    )
                ),
        }
    )


coverage = {}

for object_class in REQUIRED_CLASSES:
    actor_count = (
        aggregate_actor_counts[
            object_class
        ]
    )

    clean_count = (
        aggregate_clean_valid[
            object_class
        ]
    )

    degraded_count = (
        aggregate_degraded_true[
            object_class
        ]
    )

    passed = (
        actor_count > 0
        and clean_count > 0
        and degraded_count > 0
    )

    coverage[
        object_class
    ] = {
        "actors": actor_count,
        "clean_valid_measurements":
            clean_count,
        "degraded_true_measurements":
            degraded_count,
        "pass": passed,
    }


missing = [
    object_class
    for object_class in REQUIRED_CLASSES
    if not coverage[
        object_class
    ]["pass"]
]


if missing:
    print()
    print(
        "===== MULTICLASS COVERAGE DIAGNOSTIC ====="
    )

    print(
        "aggregate actor counts =",
        dict(
            sorted(
                aggregate_actor_counts.items()
            )
        ),
    )

    print(
        "aggregate clean-valid =",
        dict(
            sorted(
                aggregate_clean_valid.items()
            )
        ),
    )

    print(
        "aggregate degraded-true =",
        dict(
            sorted(
                aggregate_degraded_true.items()
            )
        ),
    )

    print()
    print("per-scenario:")

    for item in scenario_reports:
        print()
        print(
            "scenario =",
            item["scenario_id"],
        )
        print(
            " actors =",
            item["actors_by_class"],
        )
        print(
            " clean  =",
            item[
                "clean_valid_measurements_by_class"
            ],
        )
        print(
            " degraded =",
            item[
                "degraded_true_measurements_by_class"
            ],
        )

    print()
    print(
        "missing required coverage =",
        missing,
    )

    raise RuntimeError(
        "Frozen 3-pilot multiclass coverage "
        "is insufficient for: "
        + ", ".join(missing)
    )


payload = {
    "status": "PASS",

    "block": (
        "stage2_block3f_frozen_"
        "three_pilot_multiclass_gate"
    ),

    "pilot_policy": (
        "first_3_records_of_"
        "frozen_validation_jsonl"
    ),

    "scenario_ids": [
        item["scenario_id"]
        for item in PILOTS
    ],

    "scenario_count": 3,

    "required_classes": list(
        REQUIRED_CLASSES
    ),

    "coverage": coverage,

    "aggregate_actor_counts":
        dict(
            sorted(
                aggregate_actor_counts.items()
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

    "scenarios": scenario_reports,

    "coverage_mode": {
        "gaussian_noise": True,
        "pd": 1.0,
        "false_alarm_mean_per_frame":
            0.0,

        "purpose": (
            "class_pipeline_coverage_only"
        ),
    },

    "algorithm_truth_separation":
        "unchanged",

    "measured_fmcw": False,
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
    "===== Stage2 frozen 3-pilot multiclass gate ====="
)

print(
    "scenario IDs ="
)

for pilot in PILOTS:
    print(
        " ",
        pilot["scenario_id"],
    )


print()

for object_class in REQUIRED_CLASSES:
    result = coverage[
        object_class
    ]

    print(
        f"{object_class:16s} "
        f"actors={result['actors']:4d} "
        f"clean_valid="
        f"{result['clean_valid_measurements']:5d} "
        f"degraded_true="
        f"{result['degraded_true_measurements']:5d} "
        f"PASS={result['pass']}"
    )


print()
print(
    "measured FMCW       = NO"
)

print(
    "new dataset scan    = NO"
)

print(
    "frozen pilot policy = YES"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)
