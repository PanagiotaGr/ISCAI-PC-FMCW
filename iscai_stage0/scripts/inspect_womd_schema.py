from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message
from waymo_open_dataset.protos import scenario_pb2


TFRECORD_PATH = Path(
    "/waymo/data/v1_3_0_scenario/validation/"
    "validation.tfrecord-00000-of-00150"
)

OUTPUT_PATH = Path(
    "/waymo/iscai_stage0/reports/stage0/schema_snapshot.json"
)


def read_first_tfrecord(path: Path) -> bytes:
    """
    Read the payload of the first record from a TFRecord file.

    TFRecord layout:
        uint64 length
        uint32 masked CRC(length)
        byte[length] data
        uint32 masked CRC(data)

    CRC validation is intentionally omitted for this Stage 0 schema inspection.
    """
    if not path.exists():
        raise FileNotFoundError(f"TFRecord not found: {path}")

    with path.open("rb") as file:
        length_bytes = file.read(8)
        if len(length_bytes) != 8:
            raise RuntimeError("Could not read TFRecord length header.")

        record_length = struct.unpack("<Q", length_bytes)[0]

        length_crc = file.read(4)
        if len(length_crc) != 4:
            raise RuntimeError("Could not read TFRecord length CRC.")

        payload = file.read(record_length)
        if len(payload) != record_length:
            raise RuntimeError(
                f"Expected {record_length} payload bytes, got {len(payload)}."
            )

        data_crc = file.read(4)
        if len(data_crc) != 4:
            raise RuntimeError("Could not read TFRecord data CRC.")

    return payload


def field_type_name(field: FieldDescriptor) -> str:
    """Return a readable protobuf field type compatible with protobuf 3.20.x."""
    if field.message_type is not None:
        return field.message_type.full_name

    if field.enum_type is not None:
        return field.enum_type.full_name

    type_names = {
        FieldDescriptor.TYPE_DOUBLE: "double",
        FieldDescriptor.TYPE_FLOAT: "float",
        FieldDescriptor.TYPE_INT64: "int64",
        FieldDescriptor.TYPE_UINT64: "uint64",
        FieldDescriptor.TYPE_INT32: "int32",
        FieldDescriptor.TYPE_FIXED64: "fixed64",
        FieldDescriptor.TYPE_FIXED32: "fixed32",
        FieldDescriptor.TYPE_BOOL: "bool",
        FieldDescriptor.TYPE_STRING: "string",
        FieldDescriptor.TYPE_GROUP: "group",
        FieldDescriptor.TYPE_MESSAGE: "message",
        FieldDescriptor.TYPE_BYTES: "bytes",
        FieldDescriptor.TYPE_UINT32: "uint32",
        FieldDescriptor.TYPE_ENUM: "enum",
        FieldDescriptor.TYPE_SFIXED32: "sfixed32",
        FieldDescriptor.TYPE_SFIXED64: "sfixed64",
        FieldDescriptor.TYPE_SINT32: "sint32",
        FieldDescriptor.TYPE_SINT64: "sint64",
    }

    return type_names.get(field.type, f"unknown_type_{field.type}")


def descriptor_snapshot(message: Message) -> list[dict[str, Any]]:
    """Describe all fields known by the installed protobuf schema."""
    result: list[dict[str, Any]] = []

    for field in message.DESCRIPTOR.fields:
        result.append(
            {
                "name": field.name,
                "number": field.number,
                "type": field_type_name(field),
                "repeated": field.label == FieldDescriptor.LABEL_REPEATED,
            }
        )

    return result


def main() -> None:
    payload = read_first_tfrecord(TFRECORD_PATH)

    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(payload)

    timestamps = list(scenario.timestamps_seconds)

    snapshot: dict[str, Any] = {
        "source_file": str(TFRECORD_PATH),
        "record_payload_bytes": len(payload),
        "proto_full_name": scenario.DESCRIPTOR.full_name,

        "scenario_id": scenario.scenario_id,

        "counts": {
            "timestamps": len(scenario.timestamps_seconds),
            "tracks": len(scenario.tracks),
            "map_features": len(scenario.map_features),
            "dynamic_map_states": len(scenario.dynamic_map_states),
            "tracks_to_predict": len(scenario.tracks_to_predict),
            "objects_of_interest": len(scenario.objects_of_interest),
            "compressed_frame_laser_data": len(
                scenario.compressed_frame_laser_data
            ),
        },

        "indices": {
            "current_time_index": scenario.current_time_index,
            "sdc_track_index": scenario.sdc_track_index,
        },

        "timestamps": {
            "first": timestamps[0] if timestamps else None,
            "current": (
                timestamps[scenario.current_time_index]
                if timestamps
                and 0 <= scenario.current_time_index < len(timestamps)
                else None
            ),
            "last": timestamps[-1] if timestamps else None,
            "first_deltas": [
                timestamps[i + 1] - timestamps[i]
                for i in range(min(10, len(timestamps) - 1))
            ],
        },

        "installed_scenario_proto_fields": descriptor_snapshot(scenario),

        "present_top_level_fields": [
            field.name for field, _value in scenario.ListFields()
        ],

        "first_track": None,
    }

    if scenario.tracks:
        track = scenario.tracks[0]

        snapshot["first_track"] = {
            "id": track.id,
            "object_type": int(track.object_type),
            "num_states": len(track.states),
            "track_proto_fields": descriptor_snapshot(track),
            "state_proto_fields": (
                descriptor_snapshot(track.states[0])
                if track.states
                else []
            ),
            "valid_state_count": sum(
                int(state.valid) for state in track.states
            ),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2, ensure_ascii=False)

    print("Scenario parsed successfully.")
    print(f"scenario_id: {scenario.scenario_id}")
    print(f"timestamps: {len(scenario.timestamps_seconds)}")
    print(f"tracks: {len(scenario.tracks)}")
    print(f"map_features: {len(scenario.map_features)}")
    print(f"dynamic_map_states: {len(scenario.dynamic_map_states)}")
    print(f"current_time_index: {scenario.current_time_index}")
    print(f"sdc_track_index: {scenario.sdc_track_index}")
    print(f"tracks_to_predict: {len(scenario.tracks_to_predict)}")
    print(f"objects_of_interest: {len(scenario.objects_of_interest)}")
    print(
        "compressed_frame_laser_data:",
        len(scenario.compressed_frame_laser_data),
    )
    print(f"schema snapshot: {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Stage 0 schema inspection failed: {exc}") from exc