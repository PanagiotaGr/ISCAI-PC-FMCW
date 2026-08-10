from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

from iscai_stage0.womd_proto_io import read_first_scenario


TFRECORD = Path(
    "/waymo/data/v1_3_0_scenario/validation/"
    "validation.tfrecord-00000-of-00150"
)

OUTPUT = Path(
    "/waymo/iscai_stage0/reports/stage0/"
    "map_nested_schema.json"
)


def describe_message(message: Message) -> dict[str, Any]:
    descriptor = message.DESCRIPTOR

    return {
        "full_name": descriptor.full_name,
        "fields": [
            {
                "name": field.name,
                "number": field.number,
                "repeated": (
                    field.label == FieldDescriptor.LABEL_REPEATED
                ),
                "message_type": (
                    field.message_type.full_name
                    if field.message_type is not None
                    else None
                ),
                "enum_type": (
                    field.enum_type.full_name
                    if field.enum_type is not None
                    else None
                ),
            }
            for field in descriptor.fields
        ],
    }


def active_oneof(message: Message) -> str | None:
    for oneof in message.DESCRIPTOR.oneofs:
        value = message.WhichOneof(oneof.name)
        if value is not None:
            return value
    return None


def main() -> None:
    scenario = read_first_scenario(TFRECORD)

    map_schemas: dict[str, Any] = {}

    for feature in scenario.map_features:
        kind = active_oneof(feature)

        if kind is None or kind in map_schemas:
            continue

        nested = getattr(feature, kind)

        map_schemas[kind] = {
            "map_feature_id": feature.id,
            "schema": describe_message(nested),
            "present_fields": [
                field.name
                for field, _ in nested.ListFields()
            ],
        }

    dynamic_schema = None
    lane_state_schema = None

    if scenario.dynamic_map_states:
        dynamic = scenario.dynamic_map_states[0]

        dynamic_schema = {
            "schema": describe_message(dynamic),
            "present_fields": [
                field.name
                for field, _ in dynamic.ListFields()
            ],
        }

        if dynamic.lane_states:
            lane_state = dynamic.lane_states[0]

            lane_state_schema = {
                "schema": describe_message(lane_state),
                "present_fields": [
                    field.name
                    for field, _ in lane_state.ListFields()
                ],
                "example": {
                    field.name: (
                        int(value)
                        if field.enum_type is not None
                        else value
                    )
                    for field, value in lane_state.ListFields()
                    if field.message_type is None
                    and field.label
                    != FieldDescriptor.LABEL_REPEATED
                },
            }

    result = {
        "source_file": str(TFRECORD),
        "map_feature_schemas": map_schemas,
        "dynamic_map_state": dynamic_schema,
        "lane_state": lane_state_schema,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Map feature nested fields:")

    for kind, data in map_schemas.items():
        names = [
            x["name"]
            for x in data["schema"]["fields"]
        ]
        print(f"  {kind}: {names}")

    print("\nDynamicMapState:")
    if dynamic_schema:
        print([
            x["name"]
            for x in dynamic_schema["schema"]["fields"]
        ])

    print("\nLaneState:")
    if lane_state_schema:
        print([
            x["name"]
            for x in lane_state_schema["schema"]["fields"]
        ])
        print("example:", lane_state_schema["example"])

    print(f"\nReport: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(
            f"Nested map schema inspection failed: {exc}"
        ) from exc