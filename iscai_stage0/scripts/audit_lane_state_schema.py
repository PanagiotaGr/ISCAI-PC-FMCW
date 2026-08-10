from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from google.protobuf.message import Message

from iscai_stage0.womd_proto_io import iter_scenarios


TFRECORD = Path(
    "/waymo/data/v1_3_0_scenario/validation/"
    "validation.tfrecord-00000-of-00150"
)

OUTPUT = Path(
    "/waymo/iscai_stage0/reports/stage0/"
    "lane_state_schema_audit.json"
)

SCENARIO_LIMIT = 50


def describe_message(message: Message) -> dict[str, Any]:
    descriptor = message.DESCRIPTOR

    return {
        "full_name": descriptor.full_name,
        "fields": [
            {
                "name": field.name,
                "number": field.number,
                "repeated": field.is_repeated,
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


def enum_name(message: Message, field_name: str) -> str:
    field = message.DESCRIPTOR.fields_by_name.get(field_name)

    if field is None or field.enum_type is None:
        return "NOT_AN_ENUM"

    value = getattr(message, field_name)
    enum_value = field.enum_type.values_by_number.get(value)

    if enum_value is None:
        return f"UNKNOWN_{value}"

    return enum_value.name


def main() -> None:
    first_lane_schema = None
    first_lane_state_schema = None

    lane_type_counts: Counter[str] = Counter()
    traffic_state_counts: Counter[str] = Counter()

    total_lane_features = 0
    total_lane_states = 0

    referenced_lane_ids = set()
    map_lane_ids = set()

    for scenario in iter_scenarios(
        TFRECORD,
        limit=SCENARIO_LIMIT,
    ):
        # Static map lanes
        for feature in scenario.map_features:
            if feature.WhichOneof("feature_data") != "lane":
                continue

            total_lane_features += 1
            map_lane_ids.add(feature.id)

            lane = feature.lane

            if first_lane_schema is None:
                first_lane_schema = describe_message(lane)

            if "type" in lane.DESCRIPTOR.fields_by_name:
                lane_type_counts[enum_name(lane, "type")] += 1

        # Dynamic traffic-signal lane states
        for dynamic_state in scenario.dynamic_map_states:
            for lane_state in dynamic_state.lane_states:
                total_lane_states += 1

                if first_lane_state_schema is None:
                    first_lane_state_schema = describe_message(
                        lane_state
                    )

                if "state" in lane_state.DESCRIPTOR.fields_by_name:
                    traffic_state_counts[
                        enum_name(lane_state, "state")
                    ] += 1

                if "lane" in lane_state.DESCRIPTOR.fields_by_name:
                    referenced_lane_ids.add(lane_state.lane)

    unmatched_lane_ids = referenced_lane_ids - map_lane_ids

    result = {
        "source_file": str(TFRECORD),
        "scenarios_audited": SCENARIO_LIMIT,

        "lane": {
            "total_features": total_lane_features,
            "type_counts": dict(sorted(lane_type_counts.items())),
            "schema": first_lane_schema,
        },

        "dynamic_lane_state": {
            "total_states": total_lane_states,
            "traffic_state_counts": dict(
                sorted(traffic_state_counts.items())
            ),
            "schema": first_lane_state_schema,
        },

        "lane_reference_check": {
            "unique_map_lane_ids": len(map_lane_ids),
            "unique_referenced_lane_ids": len(
                referenced_lane_ids
            ),
            "unmatched_referenced_lane_ids": len(
                unmatched_lane_ids
            ),
            "sample_unmatched_ids": sorted(
                unmatched_lane_ids
            )[:20],
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== Lane schema ===")
    print(json.dumps(first_lane_schema, indent=2))

    print("\nLane types:")
    for name, count in sorted(lane_type_counts.items()):
        print(f"  {name}: {count}")

    print("\n=== Dynamic lane-state schema ===")
    print(json.dumps(first_lane_state_schema, indent=2))

    print("\nTraffic states:")
    for name, count in sorted(traffic_state_counts.items()):
        print(f"  {name}: {count}")

    print("\nLane reference check:")
    print(
        json.dumps(
            result["lane_reference_check"],
            indent=2,
        )
    )

    print(f"\nReport: {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(
            f"Lane-state audit failed: {exc}"
        ) from exc