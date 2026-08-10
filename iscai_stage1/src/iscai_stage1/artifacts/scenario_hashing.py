from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def manifest_record_sha256(raw_jsonl_line: bytes) -> str:
    if not raw_jsonl_line.strip():
        raise ValueError("Manifest record is empty.")

    return hashlib.sha256(raw_jsonl_line).hexdigest()


def combined_causal_artifact_sha256(
    *,
    scenario_id: str,
    anchor_index: int,
    actor_hashes: Sequence[str],
    map_hash: str,
) -> str:
    return _json_sha256(
        {
            "scenario_id": scenario_id,
            "anchor_index": anchor_index,
            "actor_hashes": list(actor_hashes),
            "map_hash": map_hash,
            "artifact_semantics":
                "causal_womd_annotation_upstream",
            "sensor_realistic": False,
        }
    )


def future_labels_payload(scenario: Any) -> dict:
    """
    Explicit non-causal/evaluation branch.

    This function is deliberately separate from realistic-input
    extraction and is allowed to read states after the anchor.
    """
    anchor = int(scenario.current_time_index)

    tracks = []

    for track_index, track in enumerate(scenario.tracks):
        future_states = []

        for time_index in range(
            anchor + 1,
            len(track.states),
        ):
            state = track.states[time_index]

            future_states.append(
                {
                    "time_index": time_index,
                    "valid": bool(state.valid),
                    "center_W_m": [
                        float(state.center_x),
                        float(state.center_y),
                        float(state.center_z),
                    ],
                    "dimensions_lwh_m": [
                        float(state.length),
                        float(state.width),
                        float(state.height),
                    ],
                    "heading_rad": float(state.heading),

                    # WOMD annotated velocity is legal here:
                    # labels/oracle branch only.
                    "annotated_velocity_W_xy_mps": [
                        float(state.velocity_x),
                        float(state.velocity_y),
                    ],
                }
            )

        tracks.append(
            {
                "track_index": track_index,
                "track_id": str(track.id),
                "object_type": int(track.object_type),
                "future_states": future_states,
            }
        )

    return {
        "scenario_id": str(scenario.scenario_id),
        "anchor_index": anchor,
        "tracks": tracks,
        "objects_of_interest": [
            int(value)
            for value in scenario.objects_of_interest
        ],
        "tracks_to_predict_serialized": [
            item.SerializeToString(
                deterministic=True
            ).hex()
            for item in scenario.tracks_to_predict
        ],
    }


def future_labels_sha256(scenario: Any) -> str:
    return _json_sha256(
        future_labels_payload(scenario)
    )


def future_dynamic_map_sha256(scenario: Any) -> str:
    anchor = int(scenario.current_time_index)

    payload = [
        state.SerializeToString(
            deterministic=True
        ).hex()
        for state in scenario.dynamic_map_states[
            anchor + 1:
        ]
    ]

    return _json_sha256(payload)
