from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from waymo_open_dataset.protos import scenario_pb2


MOTION_ROOT = Path(
    "/waymo/data/v1_3_0_scenario"
)

GCS_ROOT = (
    "gs://waymo_open_dataset_motion_v_1_3_0/"
    "uncompressed/lidar_and_camera"
)

WORK_ROOT = Path(
    "/waymo/iscai_data_prep"
)

REPORT_ROOT = WORK_ROOT / "reports"

DB_PATH = REPORT_ROOT / "paired_inventory.sqlite"

SUMMARY_JSON = (
    REPORT_ROOT / "paired_inventory_summary.json"
)

SUMMARY_MD = (
    REPORT_ROOT / "paired_inventory_summary.md"
)

LOCAL_SPLITS = ("training", "validation")

ALL_LOCAL_BRANCHES = (
    "training",
    "validation",
    "testing",
    "training_20s",
    "validation_interactive",
    "testing_interactive",
)

SCENARIO_ID_FIELD_NUMBER = (
    scenario_pb2.Scenario.DESCRIPTOR
    .fields_by_name["scenario_id"]
    .number
)

TFRECORD_OVERHEAD_BYTES = 16

GCS_LINE_RE = re.compile(
    r"^\s*(\d+)\s+\S+\s+(gs://\S+)\s*$"
)


def human_bytes(value: int | float) -> str:
    value = float(value)

    units = (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    )

    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    raise AssertionError("Unreachable")


def read_varint(
    data: bytes,
    offset: int,
) -> tuple[int, int]:
    value = 0
    shift = 0

    while True:
        if offset >= len(data):
            raise RuntimeError(
                "Truncated protobuf varint."
            )

        byte = data[offset]
        offset += 1

        value |= (byte & 0x7F) << shift

        if not byte & 0x80:
            return value, offset

        shift += 7

        if shift >= 70:
            raise RuntimeError(
                "Invalid protobuf varint."
            )


def extract_scenario_id(
    payload: bytes,
) -> str:
    """
    Extract only Scenario.scenario_id from raw protobuf bytes.

    This avoids fully decoding tracks/maps for every local record.
    """
    offset = 0
    payload_length = len(payload)

    while offset < payload_length:
        key, offset = read_varint(
            payload,
            offset,
        )

        field_number = key >> 3
        wire_type = key & 0x07

        if wire_type == 0:
            _, offset = read_varint(
                payload,
                offset,
            )

        elif wire_type == 1:
            offset += 8

        elif wire_type == 2:
            length, value_offset = read_varint(
                payload,
                offset,
            )

            end = value_offset + length

            if end > payload_length:
                raise RuntimeError(
                    "Truncated length-delimited field."
                )

            if (
                field_number
                == SCENARIO_ID_FIELD_NUMBER
            ):
                return payload[
                    value_offset:end
                ].decode("utf-8")

            offset = end

        elif wire_type == 5:
            offset += 4

        else:
            raise RuntimeError(
                "Unsupported protobuf wire type "
                f"{wire_type}."
            )

        if offset > payload_length:
            raise RuntimeError(
                "Protobuf field exceeds payload."
            )

    raise RuntimeError(
        "Scenario record has no scenario_id."
    )


def initialize_database(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = FILE;

        CREATE TABLE IF NOT EXISTS local_shards (
            split TEXT NOT NULL,
            shard_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            record_count INTEGER,
            scan_complete INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS motion_records (
            split TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            shard_path TEXT NOT NULL,
            record_offset INTEGER NOT NULL,
            payload_length INTEGER NOT NULL,
            record_bytes INTEGER NOT NULL,
            PRIMARY KEY (split, scenario_id)
        );

        CREATE INDEX IF NOT EXISTS
        idx_motion_shard
        ON motion_records(shard_path);

        CREATE TABLE IF NOT EXISTS lidar_objects (
            split TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            uri TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            PRIMARY KEY (split, scenario_id)
        );

        CREATE INDEX IF NOT EXISTS
        idx_lidar_uri
        ON lidar_objects(uri);
        """
    )

    connection.commit()


def scan_motion_shard(
    connection: sqlite3.Connection,
    split: str,
    shard: Path,
) -> int:
    stat = shard.stat()

    previous = connection.execute(
        """
        SELECT
            size_bytes,
            mtime_ns,
            record_count,
            scan_complete
        FROM local_shards
        WHERE shard_path = ?
        """,
        (str(shard),),
    ).fetchone()

    if (
        previous is not None
        and previous[0] == stat.st_size
        and previous[1] == stat.st_mtime_ns
        and previous[3] == 1
    ):
        return int(previous[2])

    with connection:
        connection.execute(
            """
            DELETE FROM motion_records
            WHERE shard_path = ?
            """,
            (str(shard),),
        )

        connection.execute(
            """
            INSERT INTO local_shards (
                split,
                shard_path,
                size_bytes,
                mtime_ns,
                record_count,
                scan_complete
            )
            VALUES (?, ?, ?, ?, NULL, 0)
            ON CONFLICT(shard_path) DO UPDATE SET
                split = excluded.split,
                size_bytes = excluded.size_bytes,
                mtime_ns = excluded.mtime_ns,
                record_count = NULL,
                scan_complete = 0
            """,
            (
                split,
                str(shard),
                stat.st_size,
                stat.st_mtime_ns,
            ),
        )

    records = []

    count = 0

    with shard.open("rb") as stream:
        while True:
            record_offset = stream.tell()

            header = stream.read(12)

            if not header:
                break

            if len(header) != 12:
                raise RuntimeError(
                    f"Truncated TFRecord header: {shard}"
                )

            payload_length = struct.unpack(
                "<Q",
                header[:8],
            )[0]

            payload = stream.read(
                payload_length
            )

            if len(payload) != payload_length:
                raise RuntimeError(
                    f"Truncated payload: {shard}"
                )

            data_crc = stream.read(4)

            if len(data_crc) != 4:
                raise RuntimeError(
                    f"Missing data CRC: {shard}"
                )

            scenario_id = extract_scenario_id(
                payload
            )

            records.append(
                (
                    split,
                    scenario_id,
                    str(shard),
                    record_offset,
                    payload_length,
                    payload_length
                    + TFRECORD_OVERHEAD_BYTES,
                )
            )

            count += 1

    try:
        with connection:
            connection.executemany(
                """
                INSERT INTO motion_records (
                    split,
                    scenario_id,
                    shard_path,
                    record_offset,
                    payload_length,
                    record_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                records,
            )

            connection.execute(
                """
                UPDATE local_shards
                SET
                    record_count = ?,
                    scan_complete = 1
                WHERE shard_path = ?
                """,
                (
                    count,
                    str(shard),
                ),
            )

    except sqlite3.IntegrityError as exc:
        raise RuntimeError(
            "Duplicate local scenario_id detected "
            f"while indexing {shard}: {exc}"
        ) from exc

    return count


def inventory_local_motion(
    connection: sqlite3.Connection,
) -> None:
    print("\n=== Local motion inventory ===")

    for split in LOCAL_SPLITS:
        split_dir = MOTION_ROOT / split

        shards = sorted(
            split_dir.glob("*.tfrecord-*")
        )

        if not shards:
            raise RuntimeError(
                f"No canonical shards found for {split}"
            )

        split_records = 0
        started = time.time()

        for index, shard in enumerate(
            shards,
            start=1,
        ):
            records = scan_motion_shard(
                connection,
                split,
                shard,
            )

            split_records += records

            if (
                index == 1
                or index % 25 == 0
                or index == len(shards)
            ):
                elapsed = time.time() - started

                print(
                    f"{split}: "
                    f"{index}/{len(shards)} shards, "
                    f"{split_records} scenarios, "
                    f"elapsed={elapsed / 60:.1f} min"
                )


def list_remote_lidar_split(
    connection: sqlite3.Connection,
    split: str,
) -> None:
    prefix = f"{GCS_ROOT}/{split}/"

    print(
        f"\n=== Remote LiDAR inventory: {split} ==="
    )
    print("prefix:", prefix)

    with connection:
        connection.execute(
            """
            DELETE FROM lidar_objects
            WHERE split = ?
            """,
            (split,),
        )

    command = [
        "gcloud",
        "storage",
        "ls",
        "--long",
        "--recursive",
        prefix,
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError(
            "Could not capture gcloud output."
        )

    batch = []
    count = 0
    total_bytes = 0

    for line in process.stdout:
        match = GCS_LINE_RE.match(line)

        if not match:
            continue

        size_bytes = int(match.group(1))
        uri = match.group(2)

        filename = uri.rsplit("/", 1)[-1]

        if not filename.endswith(".tfrecord"):
            continue

        scenario_id = filename[
            :-len(".tfrecord")
        ]

        if not scenario_id:
            continue

        batch.append(
            (
                split,
                scenario_id,
                uri,
                size_bytes,
            )
        )

        count += 1
        total_bytes += size_bytes

        if len(batch) >= 2000:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO lidar_objects (
                        split,
                        scenario_id,
                        uri,
                        size_bytes
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    batch,
                )

            batch.clear()

        if count % 25_000 == 0:
            print(
                f"{split}: "
                f"{count} objects, "
                f"{human_bytes(total_bytes)}"
            )

    if batch:
        with connection:
            connection.executemany(
                """
                INSERT INTO lidar_objects (
                    split,
                    scenario_id,
                    uri,
                    size_bytes
                )
                VALUES (?, ?, ?, ?)
                """,
                batch,
            )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            "gcloud listing failed for "
            f"{split} with code {return_code}"
        )

    print(
        f"{split}: completed, "
        f"{count} objects, "
        f"{human_bytes(total_bytes)}"
    )


def directory_size(
    path: Path,
) -> int:
    if not path.exists():
        return 0

    total = 0

    for root, _, filenames in os.walk(path):
        root_path = Path(root)

        for filename in filenames:
            file_path = root_path / filename

            try:
                total += file_path.stat().st_size
            except FileNotFoundError:
                # Ignore a file that disappeared during inventory.
                pass

    return total


def percentile(
    values: list[int],
    fraction: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    position = (
        len(ordered) - 1
    ) * fraction

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def size_statistics(
    values: list[int],
) -> dict:
    if not values:
        return {
            "count": 0,
            "total_bytes": 0,
            "min_bytes": None,
            "median_bytes": None,
            "p95_bytes": None,
            "max_bytes": None,
        }

    return {
        "count": len(values),
        "total_bytes": int(sum(values)),
        "min_bytes": int(min(values)),
        "median_bytes": percentile(
            values,
            0.50,
        ),
        "p95_bytes": percentile(
            values,
            0.95,
        ),
        "max_bytes": int(max(values)),
    }


def fetch_pair_rows(
    connection: sqlite3.Connection,
    split: str,
) -> list[tuple[str, int, int, str]]:
    rows = connection.execute(
        """
        SELECT
            motion_records.scenario_id,
            motion_records.record_bytes,
            lidar_objects.size_bytes,
            motion_records.shard_path
        FROM motion_records
        INNER JOIN lidar_objects
        ON
            motion_records.split
                = lidar_objects.split
            AND
            motion_records.scenario_id
                = lidar_objects.scenario_id
        WHERE motion_records.split = ?
        """,
        (split,),
    ).fetchall()

    return [
        (
            str(row[0]),
            int(row[1]),
            int(row[2]),
            str(row[3]),
        )
        for row in rows
    ]


def deterministic_hash_key(
    scenario_id: str,
) -> bytes:
    return hashlib.sha256(
        scenario_id.encode("utf-8")
    ).digest()


def count_max_training_fit(
    training_pairs: list[
        tuple[str, int, int, str]
    ],
    budget_bytes: int,
) -> dict:
    ordered = sorted(
        training_pairs,
        key=lambda row: (
            row[1] + row[2],
            row[0],
        ),
    )

    count = 0
    used = 0

    for _, motion_bytes, lidar_bytes, _ in ordered:
        pair_bytes = (
            motion_bytes + lidar_bytes
        )

        if used + pair_bytes > budget_bytes:
            break

        used += pair_bytes
        count += 1

    return {
        "training_pairs": count,
        "training_bytes": used,
        "selection_semantics": (
            "count-maximizing upper bound; "
            "size-biased and not recommended "
            "as the final scientific sampler"
        ),
    }


def hash_order_training_fit(
    training_pairs: list[
        tuple[str, int, int, str]
    ],
    budget_bytes: int,
) -> dict:
    ordered = sorted(
        training_pairs,
        key=lambda row: (
            deterministic_hash_key(row[0]),
            row[0],
        ),
    )

    count = 0
    used = 0

    for _, motion_bytes, lidar_bytes, _ in ordered:
        pair_bytes = (
            motion_bytes + lidar_bytes
        )

        if used + pair_bytes <= budget_bytes:
            used += pair_bytes
            count += 1

    return {
        "training_pairs": count,
        "training_bytes": used,
        "selection_semantics": (
            "deterministic scenario-id hash order; "
            "less size-biased and recommended "
            "for the final scientific sampler"
        ),
    }


def capacity_plan(
    *,
    pair_capacity_bytes: int,
    validation_pairs: list[
        tuple[str, int, int, str]
    ],
    training_pairs: list[
        tuple[str, int, int, str]
    ],
    packaging_margin_fraction: float,
) -> dict:
    effective_budget = int(
        pair_capacity_bytes
        * (1.0 - packaging_margin_fraction)
    )

    validation_bytes = sum(
        motion_bytes + lidar_bytes
        for _, motion_bytes, lidar_bytes, _
        in validation_pairs
    )

    result = {
        "raw_pair_capacity_bytes":
            pair_capacity_bytes,
        "effective_pair_capacity_bytes":
            effective_budget,
        "packaging_margin_fraction":
            packaging_margin_fraction,
        "all_validation_pairs":
            len(validation_pairs),
        "all_validation_bytes":
            validation_bytes,
    }

    if validation_bytes > effective_budget:
        result.update(
            {
                "all_validation_fits": False,
                "remaining_training_budget_bytes": 0,
                "count_max_plan": None,
                "hash_order_plan": None,
            }
        )

        return result

    training_budget = (
        effective_budget - validation_bytes
    )

    count_max = count_max_training_fit(
        training_pairs,
        training_budget,
    )

    hash_order = hash_order_training_fit(
        training_pairs,
        training_budget,
    )

    result.update(
        {
            "all_validation_fits": True,
            "remaining_training_budget_bytes":
                training_budget,

            "count_max_plan": {
                **count_max,
                "total_pairs": (
                    len(validation_pairs)
                    + count_max["training_pairs"]
                ),
                "total_bytes": (
                    validation_bytes
                    + count_max["training_bytes"]
                ),
            },

            "hash_order_plan": {
                **hash_order,
                "total_pairs": (
                    len(validation_pairs)
                    + hash_order["training_pairs"]
                ),
                "total_bytes": (
                    validation_bytes
                    + hash_order["training_bytes"]
                ),
            },
        }
    )

    return result


def build_summary(
    connection: sqlite3.Connection,
    reserves_gib: Iterable[int],
) -> dict:
    disk = shutil.disk_usage("/waymo")

    local_branch_sizes = {
        branch: directory_size(
            MOTION_ROOT / branch
        )
        for branch in ALL_LOCAL_BRANCHES
    }

    split_summary = {}

    pair_rows = {}

    for split in LOCAL_SPLITS:
        local_count, local_record_bytes = (
            connection.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(record_bytes), 0)
                FROM motion_records
                WHERE split = ?
                """,
                (split,),
            ).fetchone()
        )

        local_shards, local_shard_bytes = (
            connection.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(size_bytes), 0)
                FROM local_shards
                WHERE
                    split = ?
                    AND scan_complete = 1
                """,
                (split,),
            ).fetchone()
        )

        remote_rows = connection.execute(
            """
            SELECT size_bytes
            FROM lidar_objects
            WHERE split = ?
            """,
            (split,),
        ).fetchall()

        remote_sizes = [
            int(row[0])
            for row in remote_rows
        ]

        pairs = fetch_pair_rows(
            connection,
            split,
        )

        pair_rows[split] = pairs

        paired_motion_sizes = [
            row[1]
            for row in pairs
        ]

        paired_lidar_sizes = [
            row[2]
            for row in pairs
        ]

        paired_combined_sizes = [
            row[1] + row[2]
            for row in pairs
        ]

        split_summary[split] = {
            "local_motion": {
                "shards": int(local_shards),
                "shard_file_bytes":
                    int(local_shard_bytes),
                "scenario_records":
                    int(local_count),
                "record_bytes":
                    int(local_record_bytes),
            },

            "remote_lidar": size_statistics(
                remote_sizes
            ),

            "exact_intersection": {
                "paired_scenarios":
                    len(pairs),

                "paired_fraction_of_local":
                    (
                        len(pairs) / local_count
                        if local_count
                        else None
                    ),

                "motion_record_sizes":
                    size_statistics(
                        paired_motion_sizes
                    ),

                "lidar_sidecar_sizes":
                    size_statistics(
                        paired_lidar_sizes
                    ),

                "combined_pair_sizes":
                    size_statistics(
                        paired_combined_sizes
                    ),
            },
        }

    duplicate_cross_split = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT scenario_id
            FROM motion_records
            WHERE split IN ('training', 'validation')
            GROUP BY scenario_id
            HAVING COUNT(DISTINCT split) > 1
        )
        """
    ).fetchone()[0]

    core_original_bytes = (
        local_branch_sizes["training"]
        + local_branch_sizes["validation"]
    )

    optional_branch_bytes = sum(
        local_branch_sizes[name]
        for name in (
            "testing",
            "training_20s",
            "validation_interactive",
            "testing_interactive",
        )
    )

    used_bytes = disk.total - disk.free

    capacity_scenarios = {}

    for reserve_gib in reserves_gib:
        reserve_bytes = (
            reserve_gib * 1024**3
        )

        no_delete_capacity = max(
            0,
            disk.free - reserve_bytes,
        )

        conservative_capacity = max(
            0,
            disk.free
            + core_original_bytes
            - reserve_bytes,
        )

        recommended_capacity = max(
            0,
            disk.free
            + core_original_bytes
            + optional_branch_bytes
            - reserve_bytes,
        )

        capacity_scenarios[
            f"reserve_{reserve_gib}_gib"
        ] = {
            "reserve_bytes":
                reserve_bytes,

            "strict_no_original_deletion": (
                capacity_plan(
                    pair_capacity_bytes=
                        no_delete_capacity,
                    validation_pairs=
                        pair_rows["validation"],
                    training_pairs=
                        pair_rows["training"],
                    packaging_margin_fraction=
                        0.02,
                )
            ),

            "transactional_replace_training_validation_only": (
                capacity_plan(
                    pair_capacity_bytes=
                        conservative_capacity,
                    validation_pairs=
                        pair_rows["validation"],
                    training_pairs=
                        pair_rows["training"],
                    packaging_margin_fraction=
                        0.02,
                )
            ),

            "transactional_replace_core_and_remove_unused_branches": (
                capacity_plan(
                    pair_capacity_bytes=
                        recommended_capacity,
                    validation_pairs=
                        pair_rows["validation"],
                    training_pairs=
                        pair_rows["training"],
                    packaging_margin_fraction=
                        0.02,
                )
            ),
        }

    per_shard_pair_bytes = connection.execute(
        """
        SELECT
            motion_records.shard_path,
            COUNT(*),
            SUM(
                motion_records.record_bytes
                + lidar_objects.size_bytes
            )
        FROM motion_records
        INNER JOIN lidar_objects
        ON
            motion_records.split
                = lidar_objects.split
            AND
            motion_records.scenario_id
                = lidar_objects.scenario_id
        WHERE motion_records.split
            IN ('training', 'validation')
        GROUP BY motion_records.shard_path
        """
    ).fetchall()

    shard_pair_totals = [
        int(row[2])
        for row in per_shard_pair_bytes
        if row[2] is not None
    ]

    summary = {
        "inventory_mode": "read_only",

        "dataset": {
            "motion_release":
                "WOMD Scenario v1.3.0",
            "lidar_release":
                "WOMD-LiDAR v1.3.0",
            "local_motion_root":
                str(MOTION_ROOT),
            "remote_lidar_root":
                GCS_ROOT,
            "included_splits": [
                "training",
                "validation",
            ],
            "excluded_from_pair_priority": [
                "testing",
                "training_20s",
                "validation_interactive",
                "testing_interactive",
            ],
        },

        "disk": {
            "mount": "/waymo",
            "total_bytes": disk.total,
            "used_bytes": used_bytes,
            "free_bytes": disk.free,
        },

        "local_branch_sizes":
            local_branch_sizes,

        "split_summary":
            split_summary,

        "cross_split_duplicate_scenario_ids":
            int(duplicate_cross_split),

        "transactional_repacking": {
            "source_shards_with_pairs":
                len(per_shard_pair_bytes),

            "max_all_pairs_from_one_source_shard_bytes":
                (
                    max(shard_pair_totals)
                    if shard_pair_totals
                    else 0
                ),

            "p95_all_pairs_from_one_source_shard_bytes":
                percentile(
                    shard_pair_totals,
                    0.95,
                ),

            "note": (
                "These values are upper bounds for "
                "per-source-shard temporary output. "
                "Smaller batches can be used if necessary."
            ),
        },

        "capacity_estimates":
            capacity_scenarios,

        "generated_files": {
            "sqlite_index":
                str(DB_PATH),
            "summary_json":
                str(SUMMARY_JSON),
            "summary_markdown":
                str(SUMMARY_MD),
        },
    }

    return summary


def render_markdown(
    summary: dict,
) -> str:
    disk = summary["disk"]

    lines = [
        "# Paired WOMD/WOMD-LiDAR inventory",
        "",
        "**Mode:** read-only",
        "",
        "## Disk",
        "",
        f"- Total: {human_bytes(disk['total_bytes'])}",
        f"- Used: {human_bytes(disk['used_bytes'])}",
        f"- Free: {human_bytes(disk['free_bytes'])}",
        "",
        "## Local branches",
        "",
    ]

    for branch, size in (
        summary["local_branch_sizes"].items()
    ):
        lines.append(
            f"- `{branch}`: {human_bytes(size)}"
        )

    lines.extend(
        [
            "",
            "## Split inventory",
            "",
        ]
    )

    for split, data in (
        summary["split_summary"].items()
    ):
        local = data["local_motion"]
        remote = data["remote_lidar"]
        paired = data["exact_intersection"]

        lines.extend(
            [
                f"### {split}",
                "",
                (
                    f"- Local motion scenarios: "
                    f"{local['scenario_records']:,}"
                ),
                (
                    f"- Local shard bytes: "
                    f"{human_bytes(local['shard_file_bytes'])}"
                ),
                (
                    f"- Remote LiDAR objects: "
                    f"{remote['count']:,}"
                ),
                (
                    f"- Remote LiDAR total: "
                    f"{human_bytes(remote['total_bytes'])}"
                ),
                (
                    f"- Exact paired scenarios: "
                    f"{paired['paired_scenarios']:,}"
                ),
                (
                    f"- Paired fraction of local: "
                    f"{paired['paired_fraction_of_local']:.4%}"
                    if paired[
                        "paired_fraction_of_local"
                    ] is not None
                    else "- Paired fraction: unavailable"
                ),
                (
                    f"- Exact paired combined bytes: "
                    f"{human_bytes(paired['combined_pair_sizes']['total_bytes'])}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Capacity estimates",
            "",
            (
                "Each plan reserves 2% of calculated "
                "pair capacity for packaging/index overhead."
            ),
            "",
        ]
    )

    for reserve, modes in (
        summary[
            "capacity_estimates"
        ].items()
    ):
        lines.append(f"### {reserve}")

        for mode, plan in modes.items():
            if mode == "reserve_bytes":
                continue

            lines.append(f"- `{mode}`:")

            if not plan["all_validation_fits"]:
                lines.append(
                    "  - all paired validation does not fit"
                )
                continue

            count_plan = plan["count_max_plan"]
            hash_plan = plan["hash_order_plan"]

            lines.append(
                "  - count-max upper bound: "
                f"{count_plan['total_pairs']:,} pairs, "
                f"{human_bytes(count_plan['total_bytes'])}"
            )

            lines.append(
                "  - deterministic hash-order plan: "
                f"{hash_plan['total_pairs']:,} pairs, "
                f"{human_bytes(hash_plan['total_bytes'])}"
            )

        lines.append("")

    transactional = summary[
        "transactional_repacking"
    ]

    lines.extend(
        [
            "## Transactional staging",
            "",
            (
                "- Source shards containing at least one pair: "
                f"{transactional['source_shards_with_pairs']:,}"
            ),
            (
                "- Maximum all-paired output from one source shard: "
                f"{human_bytes(transactional['max_all_pairs_from_one_source_shard_bytes'])}"
            ),
            (
                "- P95 all-paired output from one source shard: "
                f"{human_bytes(transactional['p95_all_pairs_from_one_source_shard_bytes'] or 0)}"
            ),
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reserves-gib",
        type=int,
        nargs="+",
        default=[200, 250, 300],
        help=(
            "Free-space reserve scenarios to evaluate."
        ),
    )

    parser.add_argument(
        "--skip-local",
        action="store_true",
        help=(
            "Reuse completed local SQLite index."
        ),
    )

    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help=(
            "Reuse existing remote LiDAR SQLite inventory."
        ),
    )

    args = parser.parse_args()

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    try:
        initialize_database(
            connection
        )

        if not args.skip_local:
            inventory_local_motion(
                connection
            )

        if not args.skip_remote:
            for split in LOCAL_SPLITS:
                list_remote_lidar_split(
                    connection,
                    split,
                )

        summary = build_summary(
            connection,
            args.reserves_gib,
        )

        SUMMARY_JSON.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        SUMMARY_MD.write_text(
            render_markdown(summary),
            encoding="utf-8",
        )

        print("\n=== Inventory complete ===")
        print("Database:", DB_PATH)
        print("JSON:", SUMMARY_JSON)
        print("Markdown:", SUMMARY_MD)

        print("\nDisk:")
        print(
            "  total:",
            human_bytes(
                summary["disk"]["total_bytes"]
            ),
        )
        print(
            "  used:",
            human_bytes(
                summary["disk"]["used_bytes"]
            ),
        )
        print(
            "  free:",
            human_bytes(
                summary["disk"]["free_bytes"]
            ),
        )

        for split in LOCAL_SPLITS:
            data = summary[
                "split_summary"
            ][split]

            print(f"\n{split}:")
            print(
                "  local scenarios:",
                data["local_motion"][
                    "scenario_records"
                ],
            )
            print(
                "  remote LiDAR:",
                data["remote_lidar"]["count"],
            )
            print(
                "  exact pairs:",
                data["exact_intersection"][
                    "paired_scenarios"
                ],
            )
            print(
                "  exact pair bytes:",
                human_bytes(
                    data[
                        "exact_intersection"
                    ][
                        "combined_pair_sizes"
                    ][
                        "total_bytes"
                    ]
                ),
            )

    finally:
        connection.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(
            "\nInventory interrupted. "
            "Completed local shards remain resumable "
            "in the SQLite database."
        )
    except Exception as exc:
        raise SystemExit(
            f"Inventory failed: {exc}"
        ) from exc