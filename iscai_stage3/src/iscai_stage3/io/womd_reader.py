from __future__ import annotations

import json
from pathlib import Path

from iscai_stage0.womd_proto_io import (
    iter_scenarios,
)


MANIFEST = Path(
    "/home/agni/waymo/iscai_data_prep/"
    "manifests/selected_validation.jsonl"
)

LOCAL_MOTION_ROOT = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)


def load_manifest_record(
    scenario_id: str,
) -> dict:
    """
    Return the frozen validation-manifest record
    for an exact scenario_id.
    """

    with MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):
            record = json.loads(line)

            if record.get(
                "scenario_id"
            ) == scenario_id:

                result = dict(record)
                result["manifest_line"] = (
                    line_number
                )

                return result

    raise KeyError(
        "Scenario not found in frozen "
        f"validation manifest: {scenario_id}"
    )


def compact_motion_shard_for_record(
    record: dict,
) -> Path:
    """
    Map the immutable source_shard basename onto
    the local paired compact-motion layout.

    Example:

      source:
        validation.tfrecord-00000-of-00150

      local:
        paired-from-validation.tfrecord-00000-of-00150
    """

    source = Path(
        record["source_shard"]
    )

    basename = source.name

    local = (
        LOCAL_MOTION_ROOT
        / f"paired-from-{basename}"
    )

    if not local.is_file():
        raise FileNotFoundError(
            "Compact motion shard missing: "
            f"{local}"
        )

    return local


def read_scenario_by_id(
    scenario_id: str,
):
    """
    Read one exact frozen validation scenario
    from its local compact motion shard.

    Selection uses only:
      - frozen manifest scenario_id
      - frozen manifest source_shard

    No future-state content is inspected.
    """

    record = load_manifest_record(
        scenario_id
    )

    shard = compact_motion_shard_for_record(
        record
    )

    for local_record_index, scenario in enumerate(
        iter_scenarios(
            shard,
            limit=None,
        )
    ):
        if scenario.scenario_id == scenario_id:

            return (
                scenario,
                record,
                shard,
                local_record_index,
            )

    raise RuntimeError(
        "Scenario not found in resolved compact "
        f"motion shard: scenario_id={scenario_id}, "
        f"shard={shard}"
    )
