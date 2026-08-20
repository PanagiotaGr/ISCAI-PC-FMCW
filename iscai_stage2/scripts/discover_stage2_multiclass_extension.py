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
from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)


MANIFEST = Path(
    "/home/agni/waymo/iscai_data_prep/"
    "manifests/selected_validation.jsonl"
)

MOTION_ROOT = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage2/"
    "reports/"
    "block3f1_multiclass_extension_discovery.json"
)


EXPECTED_MANIFEST_SHA256 = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

START_RECORD_1_BASED = 4

TARGETS = {
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
}


CLEAN_CONFIG = CleanObservationConfig(
    sensing_snr_db=20.0,
    azimuth_std_rad=math.radians(1.0),
    elevation_std_rad=math.radians(1.0),
)


def file_sha256(
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


def resolve_motion_shard(
    manifest_record: dict,
) -> Path:
    source_name = Path(
        manifest_record["source_shard"]
    ).name

    candidate = (
        MOTION_ROOT
        / f"paired-from-{source_name}"
    )

    if not candidate.exists():
        raise RuntimeError(
            "Expected paired motion shard missing: "
            f"{candidate}"
        )

    return candidate


def read_scenario(
    manifest_record: dict,
):
    shard = resolve_motion_shard(
        manifest_record
    )

    offset = int(
        manifest_record["record_offset"]
    )

    expected_length = int(
        manifest_record["payload_length"]
    )

    with shard.open("rb") as f:
        f.seek(offset)

        length_bytes = f.read(8)

        if len(length_bytes) != 8:
            raise RuntimeError(
                f"Cannot read TFRecord length "
                f"from {shard} at {offset}."
            )

        actual_length = struct.unpack(
            "<Q",
            length_bytes,
        )[0]

        if actual_length != expected_length:
            raise RuntimeError(
                "Payload length mismatch for "
                f"{manifest_record['scenario_id']}: "
                f"expected={expected_length}, "
                f"actual={actual_length}"
            )

        # Skip TFRecord length CRC.
        if len(f.read(4)) != 4:
            raise RuntimeError(
                "Truncated TFRecord length CRC."
            )

        payload = f.read(
            actual_length
        )

        if len(payload) != actual_length:
            raise RuntimeError(
                "Truncated TFRecord payload."
            )

        # Skip TFRecord payload CRC.
        if len(f.read(4)) != 4:
            raise RuntimeError(
                "Truncated TFRecord payload CRC."
            )

    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(
        payload
    )

    if (
        scenario.scenario_id
        != manifest_record["scenario_id"]
    ):
        raise RuntimeError(
            "Scenario-ID mismatch between "
            "manifest and TFRecord."
        )

    return scenario, shard


def causal_raw_counts(
    scenario,
) -> Counter:
    """
    Counts actors of each target class that have at least
    one valid state at or before current_time_index.

    No future state is used for qualification.
    """
    anchor = int(
        scenario.current_time_index
    )

    counts = Counter()

    for track in scenario.tracks:
        object_type = int(
            track.object_type
        )

        if object_type not in TARGETS:
            continue

        causal_valid = any(
            state.valid
            for state in track.states[
                : anchor + 1
            ]
        )

        if causal_valid:
            counts[
                TARGETS[object_type]
            ] += 1

    return counts


def clean_qualification(
    scenario,
) -> dict[str, dict]:
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

    result = {}

    for class_name in TARGETS.values():
        actors = [
            actor
            for actor in clean.actors
            if actor.object_class == class_name
        ]

        valid_by_track = {}

        for actor in actors:
            valid_count = sum(
                record.measurement_valid
                for record in actor.records
            )

            if valid_count > 0:
                valid_by_track[
                    actor.track_id
                ] = valid_count

        result[class_name] = {
            "actor_count_in_stage2_scene":
                len(actors),

            "qualifying_actor_count":
                len(valid_by_track),

            "clean_valid_measurements":
                sum(
                    valid_by_track.values()
                ),

            "qualifying_tracks":
                dict(
                    sorted(
                        valid_by_track.items()
                    )
                ),
        }

    return result


actual_manifest_sha256 = (
    file_sha256(
        MANIFEST
    )
)

if (
    actual_manifest_sha256
    != EXPECTED_MANIFEST_SHA256
):
    raise RuntimeError(
        "Frozen validation manifest SHA mismatch."
    )


found = {}
raw_candidate_scenarios = Counter()
pipeline_candidate_scenarios = Counter()

records_examined = 0
last_record_index = None


with MANIFEST.open(
    "r",
    encoding="utf-8",
) as f:

    for record_index, line in enumerate(
        f,
        start=1,
    ):
        if record_index < START_RECORD_1_BASED:
            continue

        line = line.strip()

        if not line:
            continue

        record = json.loads(
            line
        )

        if (
            record.get("selection_policy")
            != "all_exact_matching_validation"
        ):
            raise RuntimeError(
                "Selection policy changed inside "
                "frozen manifest."
            )

        records_examined += 1
        last_record_index = record_index

        scenario, shard = (
            read_scenario(
                record
            )
        )

        raw_counts = (
            causal_raw_counts(
                scenario
            )
        )

        unresolved_with_raw_candidate = [
            class_name
            for class_name
            in TARGETS.values()
            if (
                class_name not in found
                and raw_counts[
                    class_name
                ] > 0
            )
        ]

        if not unresolved_with_raw_candidate:
            continue

        for class_name in (
            unresolved_with_raw_candidate
        ):
            raw_candidate_scenarios[
                class_name
            ] += 1

        qualification = (
            clean_qualification(
                scenario
            )
        )

        for class_name in (
            unresolved_with_raw_candidate
        ):
            pipeline_candidate_scenarios[
                class_name
            ] += 1

            q = qualification[
                class_name
            ]

            if (
                q["clean_valid_measurements"]
                <= 0
            ):
                continue

            found[class_name] = {
                "manifest_record_index_1_based":
                    record_index,

                "scenario_id":
                    scenario.scenario_id,

                "source_shard":
                    record["source_shard"],

                "resolved_motion_shard":
                    str(shard),

                "record_offset":
                    int(
                        record["record_offset"]
                    ),

                "payload_length":
                    int(
                        record["payload_length"]
                    ),

                "selection_hash":
                    record.get(
                        "selection_hash"
                    ),

                "raw_causal_actor_count":
                    int(
                        raw_counts[
                            class_name
                        ]
                    ),

                **q,
            }

            print(
                "FOUND",
                class_name,
                "at manifest record",
                record_index,
                "scenario",
                scenario.scenario_id,
                "clean_valid=",
                q[
                    "clean_valid_measurements"
                ],
            )

        if len(found) == len(TARGETS):
            break


missing = [
    class_name
    for class_name in TARGETS.values()
    if class_name not in found
]


payload = {
    "status": (
        "PASS"
        if not missing
        else "INCOMPLETE"
    ),

    "block": (
        "stage2_block3f1_deterministic_"
        "causal_multiclass_extension_discovery"
    ),

    "manifest": str(
        MANIFEST
    ),

    "manifest_sha256":
        actual_manifest_sha256,

    "manifest_selection_policy":
        "all_exact_matching_validation",

    "scan_start_record_1_based":
        START_RECORD_1_BASED,

    "records_examined":
        records_examined,

    "last_record_index_1_based":
        last_record_index,

    "selection_semantics": {
        "full_scenario_actor_presence":
            "not_sufficient",

        "raw_prefilter":
            "at_least_one_valid_state_at_or_before_anchor",

        "qualification":
            "at_least_one_stage2_clean_measurement_valid",

        "future_actor_states_used_for_selection":
            False,

        "future_labels_used_for_selection":
            False,

        "cherry_picking":
            False,

        "first_qualifying_scenario_per_class":
            True,
    },

    "targets": list(
        TARGETS.values()
    ),

    "raw_candidate_scenarios_seen":
        dict(
            raw_candidate_scenarios
        ),

    "pipeline_candidate_scenarios_evaluated":
        dict(
            pipeline_candidate_scenarios
        ),

    "found": found,

    "missing": missing,

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


print()
print(
    "===== Stage2 deterministic multiclass discovery ====="
)

print(
    "manifest SHA      =",
    actual_manifest_sha256,
)

print(
    "scan start record =",
    START_RECORD_1_BASED,
)

print(
    "records examined  =",
    records_examined,
)

print(
    "last record index =",
    last_record_index,
)

print()

for class_name in TARGETS.values():
    if class_name in found:
        item = found[
            class_name
        ]

        print(
            f"{class_name:16s} "
            f"record="
            f"{item['manifest_record_index_1_based']} "
            f"scenario="
            f"{item['scenario_id']} "
            f"causal_actors="
            f"{item['raw_causal_actor_count']} "
            f"clean_valid="
            f"{item['clean_valid_measurements']} "
            "FOUND"
        )
    else:
        print(
            f"{class_name:16s} NOT FOUND"
        )

print()
print(
    "future-state selection = NO"
)

print(
    "cherry-picking         = NO"
)

print(
    "measured FMCW          = NO"
)

print(
    "STATUS =",
    payload["status"],
)

print(
    "report =",
    REPORT,
)


if missing:
    raise RuntimeError(
        "No qualifying scenario found for: "
        + ", ".join(missing)
    )
