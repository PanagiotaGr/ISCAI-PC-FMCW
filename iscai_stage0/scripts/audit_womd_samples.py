from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from iscai_stage0.womd_proto_io import iter_scenarios


ROOT = Path("/waymo/data/v1_3_0_scenario")
OUTPUT = Path(
    "/waymo/iscai_stage0/reports/stage0/"
    "multi_scenario_audit.json"
)

FILES = {
    "training": ROOT / "training/training.tfrecord-00000-of-01000",
    "validation": ROOT / "validation/validation.tfrecord-00000-of-00150",
}

SCENARIOS_PER_SPLIT = 50


def object_type_name(track) -> str:
    """Return protobuf enum name when available."""
    field = track.DESCRIPTOR.fields_by_name["object_type"]
    value = field.enum_type.values_by_number.get(track.object_type)

    if value is None:
        return f"UNKNOWN_{track.object_type}"

    return value.name


def audit_split(path: Path) -> dict[str, Any]:
    scenarios = list(
        iter_scenarios(path, limit=SCENARIOS_PER_SPLIT)
    )

    if not scenarios:
        raise RuntimeError(f"No scenarios found in {path}")

    timestamp_counts: Counter[int] = Counter()
    current_indices: Counter[int] = Counter()
    tracks_to_predict_counts: Counter[int] = Counter()
    object_types: Counter[str] = Counter()

    track_counts: list[int] = []
    map_counts: list[int] = []
    dynamic_map_counts: list[int] = []
    timestep_deltas: list[float] = []

    total_states = 0
    valid_states = 0
    causal_states = 0
    valid_causal_states = 0
    future_states = 0
    valid_future_states = 0

    bad_sdc_indices = 0
    bad_track_lengths = 0
    non_monotonic_timestamps = 0
    scenarios_with_lidar_payload = 0
    scenarios_with_camera_tokens = 0

    for scenario in scenarios:
        timestamps = list(scenario.timestamps_seconds)
        n_timestamps = len(timestamps)
        anchor = scenario.current_time_index

        timestamp_counts[n_timestamps] += 1
        current_indices[anchor] += 1
        tracks_to_predict_counts[len(scenario.tracks_to_predict)] += 1

        track_counts.append(len(scenario.tracks))
        map_counts.append(len(scenario.map_features))
        dynamic_map_counts.append(len(scenario.dynamic_map_states))

        if len(scenario.compressed_frame_laser_data) > 0:
            scenarios_with_lidar_payload += 1

        if len(scenario.frame_camera_tokens) > 0:
            scenarios_with_camera_tokens += 1

        if not (0 <= scenario.sdc_track_index < len(scenario.tracks)):
            bad_sdc_indices += 1

        deltas = [
            b - a
            for a, b in zip(timestamps[:-1], timestamps[1:])
        ]

        timestep_deltas.extend(deltas)

        if any(dt <= 0 for dt in deltas):
            non_monotonic_timestamps += 1

        for track in scenario.tracks:
            object_types[object_type_name(track)] += 1

            if len(track.states) != n_timestamps:
                bad_track_lengths += 1

            for index, state in enumerate(track.states):
                total_states += 1
                valid_states += int(state.valid)

                if index <= anchor:
                    causal_states += 1
                    valid_causal_states += int(state.valid)
                else:
                    future_states += 1
                    valid_future_states += int(state.valid)

    return {
        "source_file": str(path),
        "scenarios_audited": len(scenarios),

        "timestamp_counts": dict(timestamp_counts),
        "current_time_indices": dict(current_indices),
        "tracks_to_predict_counts": dict(
            sorted(tracks_to_predict_counts.items())
        ),

        "object_type_track_counts": dict(object_types),

        "tracks_per_scenario": {
            "min": min(track_counts),
            "max": max(track_counts),
            "mean": mean(track_counts),
        },

        "map_features_per_scenario": {
            "min": min(map_counts),
            "max": max(map_counts),
            "mean": mean(map_counts),
        },

        "dynamic_map_states_per_scenario": {
            "min": min(dynamic_map_counts),
            "max": max(dynamic_map_counts),
            "mean": mean(dynamic_map_counts),
        },

        "timestamp_delta_seconds": {
            "min": min(timestep_deltas),
            "max": max(timestep_deltas),
            "mean": mean(timestep_deltas),
        },

        "validity": {
            "all_states": {
                "total": total_states,
                "valid": valid_states,
                "fraction": valid_states / total_states,
            },
            "causal_history_and_current": {
                "total": causal_states,
                "valid": valid_causal_states,
                "fraction": valid_causal_states / causal_states,
            },
            "future_labels": {
                "total": future_states,
                "valid": valid_future_states,
                "fraction": valid_future_states / future_states,
            },
        },

        "integrity": {
            "bad_sdc_indices": bad_sdc_indices,
            "bad_track_lengths": bad_track_lengths,
            "non_monotonic_timestamp_scenarios":
                non_monotonic_timestamps,
            "scenarios_with_lidar_payload":
                scenarios_with_lidar_payload,
            "scenarios_with_camera_tokens":
                scenarios_with_camera_tokens,
        },
    }


def main() -> None:
    result: dict[str, Any] = {
        "dataset_release": "v1.3.0",
        "scenarios_per_split_requested": SCENARIOS_PER_SPLIT,
        "splits": {},
    }

    for split, path in FILES.items():
        print(f"Auditing {split}: {path}")
        result["splits"][split] = audit_split(path)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Report written to: {OUTPUT}")

    for split, data in result["splits"].items():
        print(f"\n[{split}]")
        print("scenarios:", data["scenarios_audited"])
        print("timestamps:", data["timestamp_counts"])
        print("current indices:", data["current_time_indices"])
        print(
            "object types:",
            data["object_type_track_counts"],
        )
        print("integrity:", data["integrity"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(
            f"Multi-scenario audit failed: {exc}"
        ) from exc