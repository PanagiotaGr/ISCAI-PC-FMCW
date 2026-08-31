from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Any

from waymo_open_dataset.protos import scenario_pb2


ROOT = Path("/waymo/iscai_data_prep")
REPORT_DIR = ROOT / "reports"
MANIFEST_DIR = ROOT / "manifests"

DB_PATH = REPORT_DIR / "paired_inventory.sqlite"
SUMMARY_PATH = REPORT_DIR / "paired_inventory_summary.json"

AUDIT_JSON = REPORT_DIR / "class_distribution_audit.json"
AUDIT_MD = REPORT_DIR / "class_distribution_audit.md"

SELECTION_CONFIG = MANIFEST_DIR / "selection_config.json"
TRAIN_MANIFEST = MANIFEST_DIR / "selected_training.jsonl"
VALIDATION_MANIFEST = MANIFEST_DIR / "selected_validation.jsonl"

LOCAL_WITHOUT_LIDAR = (
    MANIFEST_DIR / "excluded_local_without_lidar.jsonl"
)

REMOTE_WITHOUT_LOCAL = (
    MANIFEST_DIR / "remote_without_local_motion.jsonl"
)

PARTS_ROOT = REPORT_DIR / "class_audit_parts"

SPLITS = ("training", "validation")

SUPPORTED_CORE_CLASSES = {
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
}

RATIO_METRICS = (
    "context_actors",
    "anchor_valid_actors",
    "forecasting_target_candidates",
    "scenario_presence",
)

MAX_SUPPORTED_ABSOLUTE_DIFF_PP = 0.50
MAX_SUPPORTED_RELATIVE_DIFF = 0.10

MIN_FORECAST_CAUSAL_STATES = 2


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    temporary.replace(path)


def write_json(
    path: Path,
    value: Any,
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    text = "".join(
        json.dumps(
            row,
            sort_keys=True,
        )
        + "\n"
        for row in rows
    )

    atomic_write_text(path, text)


def read_payloads(path: Path):
    with path.open("rb") as stream:
        while True:
            header = stream.read(12)

            if not header:
                return

            if len(header) != 12:
                raise RuntimeError(
                    f"Truncated TFRecord header: {path}"
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
                    f"Truncated payload: {path}"
                )

            data_crc = stream.read(4)

            if len(data_crc) != 4:
                raise RuntimeError(
                    f"Missing data CRC: {path}"
                )

            yield payload


def object_type_name(track) -> str:
    field = (
        track.DESCRIPTOR
        .fields_by_name["object_type"]
    )

    enum_value = (
        field.enum_type
        .values_by_number
        .get(int(track.object_type))
    )

    if enum_value is None:
        return (
            f"UNKNOWN_OBJECT_TYPE_"
            f"{int(track.object_type)}"
        )

    return enum_value.name


def empty_statistics() -> dict[str, Any]:
    return {
        "scenarios": 0,
        "sdc_scenarios": 0,
        "all_non_sdc_tracks": Counter(),
        "context_actors": Counter(),
        "anchor_valid_actors": Counter(),
        "forecasting_target_candidates": Counter(),
        "scenario_presence": Counter(),
    }


def serialize_statistics(
    statistics: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: (
            dict(sorted(value.items()))
            if isinstance(value, Counter)
            else value
        )
        for key, value in statistics.items()
    }


def deserialize_statistics(
    value: dict[str, Any],
) -> dict[str, Any]:
    output = {}

    for key, item in value.items():
        if key in {
            "all_non_sdc_tracks",
            "context_actors",
            "anchor_valid_actors",
            "forecasting_target_candidates",
            "scenario_presence",
        }:
            output[key] = Counter(item)
        else:
            output[key] = item

    return output


def merge_statistics(
    destination: dict[str, Any],
    source: dict[str, Any],
) -> None:
    destination["scenarios"] += source["scenarios"]
    destination["sdc_scenarios"] += source["sdc_scenarios"]

    for metric in (
        "all_non_sdc_tracks",
        "context_actors",
        "anchor_valid_actors",
        "forecasting_target_candidates",
        "scenario_presence",
    ):
        destination[metric].update(
            source[metric]
        )


def update_statistics(
    statistics: dict[str, Any],
    scenario,
) -> None:
    anchor = int(
        scenario.current_time_index
    )

    statistics["scenarios"] += 1

    if (
        0 <= scenario.sdc_track_index
        < len(scenario.tracks)
    ):
        statistics["sdc_scenarios"] += 1

    context_classes_in_scenario: set[str] = set()

    for track_index, track in enumerate(
        scenario.tracks
    ):
        if (
            track_index
            == scenario.sdc_track_index
        ):
            continue

        class_name = object_type_name(
            track
        )

        statistics[
            "all_non_sdc_tracks"
        ][class_name] += 1

        causal_states = track.states[
            : anchor + 1
        ]

        valid_causal_count = sum(
            bool(state.valid)
            for state in causal_states
        )

        is_context_actor = (
            valid_causal_count >= 1
        )

        is_anchor_valid = (
            anchor < len(track.states)
            and bool(
                track.states[anchor].valid
            )
        )

        is_forecasting_candidate = (
            is_anchor_valid
            and valid_causal_count
            >= MIN_FORECAST_CAUSAL_STATES
        )

        if is_context_actor:
            statistics[
                "context_actors"
            ][class_name] += 1

            context_classes_in_scenario.add(
                class_name
            )

        if is_anchor_valid:
            statistics[
                "anchor_valid_actors"
            ][class_name] += 1

        if is_forecasting_candidate:
            statistics[
                "forecasting_target_candidates"
            ][class_name] += 1

    for class_name in context_classes_in_scenario:
        statistics[
            "scenario_presence"
        ][class_name] += 1


def query_paired_rows(
    connection: sqlite3.Connection,
    split: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            motion_records.scenario_id,
            motion_records.shard_path,
            motion_records.record_offset,
            motion_records.payload_length,
            motion_records.record_bytes,
            lidar_objects.uri,
            lidar_objects.size_bytes
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
        {
            "split": split,
            "scenario_id": str(row[0]),
            "source_shard": str(row[1]),
            "record_offset": int(row[2]),
            "payload_length": int(row[3]),
            "motion_record_bytes": int(row[4]),
            "lidar_uri": str(row[5]),
            "lidar_bytes": int(row[6]),
            "pair_bytes": (
                int(row[4]) + int(row[6])
            ),
        }
        for row in rows
    ]


def select_training(
    rows: list[dict[str, Any]],
    budget_bytes: int,
) -> tuple[list[dict[str, Any]], int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            bytes.fromhex(
                sha256_text(
                    row["scenario_id"]
                )
            ),
            row["scenario_id"],
        ),
    )

    selected = []
    used_bytes = 0

    for hash_rank, row in enumerate(
        ordered
    ):
        pair_bytes = row["pair_bytes"]

        if (
            used_bytes + pair_bytes
            > budget_bytes
        ):
            continue

        output_row = dict(row)

        output_row.update(
            {
                "selection_policy": (
                    "deterministic_sha256_scenario_id"
                ),
                "selection_hash": sha256_text(
                    row["scenario_id"]
                ),
                "selection_hash_rank":
                    hash_rank,
            }
        )

        selected.append(output_row)
        used_bytes += pair_bytes

    return selected, used_bytes


def select_validation(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    selected = []

    for row in sorted(
        rows,
        key=lambda item: (
            item["source_shard"],
            item["record_offset"],
        ),
    ):
        output_row = dict(row)

        output_row.update(
            {
                "selection_policy": (
                    "all_exact_matching_validation"
                ),
                "selection_hash": sha256_text(
                    row["scenario_id"]
                ),
                "selection_hash_rank": None,
            }
        )

        selected.append(output_row)

    return (
        selected,
        sum(
            row["pair_bytes"]
            for row in selected
        ),
    )


def id_digest(
    scenario_ids: set[str],
) -> str:
    digest = hashlib.sha256()

    for scenario_id in sorted(
        scenario_ids
    ):
        digest.update(
            scenario_id.encode("utf-8")
        )
        digest.update(b"\n")

    return digest.hexdigest()


def scan_split(
    *,
    split: str,
    paired_ids: set[str],
    selected_ids: set[str],
) -> dict[str, Any]:
    shard_rows: dict[str, list[str]] = {}

    # Obtain source shard paths from the database-derived IDs
    # through the canonical local directory.
    local_dir = Path(
        "/waymo/data/v1_3_0_scenario"
    ) / split

    shards = sorted(
        local_dir.glob("*.tfrecord-*")
    )

    if not shards:
        raise RuntimeError(
            f"No local shards found for {split}"
        )

    pair_digest = id_digest(
        paired_ids
    )

    selection_digest = id_digest(
        selected_ids
    )

    split_parts = PARTS_ROOT / split

    split_parts.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_total = empty_statistics()
    selected_total = empty_statistics()

    paired_records_seen = 0
    selected_records_seen = 0

    started = time.time()

    for shard_index, shard in enumerate(
        shards,
        start=1,
    ):
        stat = shard.stat()

        partial_path = (
            split_parts
            / f"{shard.name}.json"
        )

        partial = None

        if partial_path.is_file():
            candidate = json.loads(
                partial_path.read_text(
                    encoding="utf-8"
                )
            )

            metadata = candidate.get(
                "metadata",
                {},
            )

            if (
                metadata.get("source_size_bytes")
                == stat.st_size
                and metadata.get("source_mtime_ns")
                == stat.st_mtime_ns
                and metadata.get("pair_id_digest")
                == pair_digest
                and metadata.get(
                    "selection_id_digest"
                )
                == selection_digest
            ):
                partial = candidate

        if partial is None:
            full_statistics = (
                empty_statistics()
            )

            selected_statistics = (
                empty_statistics()
            )

            shard_paired_seen = 0
            shard_selected_seen = 0

            for payload in read_payloads(
                shard
            ):
                scenario = (
                    scenario_pb2.Scenario()
                )

                scenario.ParseFromString(
                    payload
                )

                scenario_id = (
                    scenario.scenario_id
                )

                if scenario_id not in paired_ids:
                    continue

                shard_paired_seen += 1

                update_statistics(
                    full_statistics,
                    scenario,
                )

                if scenario_id in selected_ids:
                    shard_selected_seen += 1

                    update_statistics(
                        selected_statistics,
                        scenario,
                    )

            partial = {
                "metadata": {
                    "split": split,
                    "source_shard": str(shard),
                    "source_size_bytes":
                        stat.st_size,
                    "source_mtime_ns":
                        stat.st_mtime_ns,
                    "pair_id_digest":
                        pair_digest,
                    "selection_id_digest":
                        selection_digest,
                },
                "paired_records_seen":
                    shard_paired_seen,
                "selected_records_seen":
                    shard_selected_seen,
                "full_paired_statistics":
                    serialize_statistics(
                        full_statistics
                    ),
                "selected_statistics":
                    serialize_statistics(
                        selected_statistics
                    ),
            }

            write_json(
                partial_path,
                partial,
            )

        full_statistics = deserialize_statistics(
            partial[
                "full_paired_statistics"
            ]
        )

        selected_statistics = (
            deserialize_statistics(
                partial[
                    "selected_statistics"
                ]
            )
        )

        merge_statistics(
            full_total,
            full_statistics,
        )

        merge_statistics(
            selected_total,
            selected_statistics,
        )

        paired_records_seen += int(
            partial["paired_records_seen"]
        )

        selected_records_seen += int(
            partial["selected_records_seen"]
        )

        if (
            shard_index == 1
            or shard_index % 25 == 0
            or shard_index == len(shards)
        ):
            elapsed_minutes = (
                time.time() - started
            ) / 60.0

            print(
                f"{split}: "
                f"{shard_index}/{len(shards)} shards, "
                f"paired={paired_records_seen}, "
                f"selected={selected_records_seen}, "
                f"elapsed={elapsed_minutes:.1f} min"
            )

    if paired_records_seen != len(
        paired_ids
    ):
        raise RuntimeError(
            f"{split}: paired records seen "
            f"{paired_records_seen}, expected "
            f"{len(paired_ids)}"
        )

    if selected_records_seen != len(
        selected_ids
    ):
        raise RuntimeError(
            f"{split}: selected records seen "
            f"{selected_records_seen}, expected "
            f"{len(selected_ids)}"
        )

    return {
        "full_paired": serialize_statistics(
            full_total
        ),
        "selected": serialize_statistics(
            selected_total
        ),
        "paired_records_seen":
            paired_records_seen,
        "selected_records_seen":
            selected_records_seen,
        "pair_id_digest": pair_digest,
        "selection_id_digest":
            selection_digest,
    }


def compare_distributions(
    full_statistics: dict[str, Any],
    selected_statistics: dict[str, Any],
) -> dict[str, Any]:
    metrics = {}

    missing_classes: set[str] = set()
    supported_ratio_failures = []

    for metric in (
        "all_non_sdc_tracks",
        "context_actors",
        "anchor_valid_actors",
        "forecasting_target_candidates",
        "scenario_presence",
    ):
        full_counts = {
            key: int(value)
            for key, value
            in full_statistics[metric].items()
        }

        selected_counts = {
            key: int(value)
            for key, value
            in selected_statistics[
                metric
            ].items()
        }

        all_classes = sorted(
            set(full_counts)
            | set(selected_counts)
        )

        full_total = sum(
            full_counts.values()
        )

        selected_total = sum(
            selected_counts.values()
        )

        class_rows = {}

        for class_name in all_classes:
            full_count = full_counts.get(
                class_name,
                0,
            )

            selected_count = (
                selected_counts.get(
                    class_name,
                    0,
                )
            )

            full_share = (
                full_count / full_total
                if full_total
                else 0.0
            )

            selected_share = (
                selected_count
                / selected_total
                if selected_total
                else 0.0
            )

            absolute_difference_pp = (
                100.0
                * (
                    selected_share
                    - full_share
                )
            )

            relative_difference = (
                abs(
                    selected_share
                    - full_share
                )
                / full_share
                if full_share > 0
                else None
            )

            if (
                full_count > 0
                and selected_count == 0
            ):
                missing_classes.add(
                    class_name
                )

            ratio_pass = None

            if (
                metric in RATIO_METRICS
                and class_name
                in SUPPORTED_CORE_CLASSES
                and full_count > 0
            ):
                ratio_pass = (
                    abs(
                        absolute_difference_pp
                    )
                    <= MAX_SUPPORTED_ABSOLUTE_DIFF_PP
                    and relative_difference
                    is not None
                    and relative_difference
                    <= MAX_SUPPORTED_RELATIVE_DIFF
                )

                if not ratio_pass:
                    supported_ratio_failures.append(
                        {
                            "metric": metric,
                            "class":
                                class_name,
                            "absolute_difference_pp":
                                absolute_difference_pp,
                            "relative_difference":
                                relative_difference,
                        }
                    )

            class_rows[class_name] = {
                "full_count": full_count,
                "selected_count":
                    selected_count,
                "full_share": full_share,
                "selected_share":
                    selected_share,
                "absolute_difference_pp":
                    absolute_difference_pp,
                "relative_difference":
                    relative_difference,
                "supported_ratio_gate":
                    ratio_pass,
            }

        metrics[metric] = {
            "full_total": full_total,
            "selected_total":
                selected_total,
            "classes": class_rows,
        }

    return {
        "metrics": metrics,
        "missing_classes":
            sorted(missing_classes),
        "supported_ratio_failures":
            supported_ratio_failures,
        "all_observed_classes_preserved":
            not missing_classes,
        "supported_ratio_gate_pass":
            not supported_ratio_failures,
        "class_preservation_gate_pass": (
            not missing_classes
            and not supported_ratio_failures
        ),
    }


def query_unmatched(
    connection: sqlite3.Connection,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    local_without_lidar = (
        connection.execute(
            """
            SELECT
                motion_records.split,
                motion_records.scenario_id,
                motion_records.shard_path,
                motion_records.record_offset,
                motion_records.record_bytes
            FROM motion_records
            LEFT JOIN lidar_objects
            ON
                motion_records.split
                    = lidar_objects.split
                AND
                motion_records.scenario_id
                    = lidar_objects.scenario_id
            WHERE
                motion_records.split
                    IN ('training', 'validation')
                AND lidar_objects.scenario_id
                    IS NULL
            ORDER BY
                motion_records.split,
                motion_records.scenario_id
            """
        ).fetchall()
    )

    remote_without_local = (
        connection.execute(
            """
            SELECT
                lidar_objects.split,
                lidar_objects.scenario_id,
                lidar_objects.uri,
                lidar_objects.size_bytes
            FROM lidar_objects
            LEFT JOIN motion_records
            ON
                lidar_objects.split
                    = motion_records.split
                AND
                lidar_objects.scenario_id
                    = motion_records.scenario_id
            WHERE
                lidar_objects.split
                    IN ('training', 'validation')
                AND motion_records.scenario_id
                    IS NULL
            ORDER BY
                lidar_objects.split,
                lidar_objects.scenario_id
            """
        ).fetchall()
    )

    return (
        [
            {
                "split": str(row[0]),
                "scenario_id": str(row[1]),
                "source_shard": str(row[2]),
                "record_offset": int(row[3]),
                "motion_record_bytes":
                    int(row[4]),
            }
            for row in local_without_lidar
        ],
        [
            {
                "split": str(row[0]),
                "scenario_id": str(row[1]),
                "lidar_uri": str(row[2]),
                "lidar_bytes": int(row[3]),
            }
            for row in remote_without_local
        ],
    )


def render_markdown(
    audit: dict[str, Any],
) -> str:
    lines = [
        "# Paired dataset class-distribution audit",
        "",
        f"**Status:** `{audit['status']}`",
        "",
        "## Selection",
        "",
        (
            f"- Training: "
            f"{audit['selection']['training_scenarios']:,}"
        ),
        (
            f"- Validation: "
            f"{audit['selection']['validation_scenarios']:,}"
        ),
        (
            f"- Total: "
            f"{audit['selection']['total_scenarios']:,}"
        ),
        (
            f"- Exact combined bytes: "
            f"{audit['selection']['total_pair_bytes']:,}"
        ),
        "",
        "## Class-preservation gate",
        "",
    ]

    for split in SPLITS:
        comparison = audit[
            "class_comparison"
        ][split]

        lines.extend(
            [
                f"### {split}",
                "",
                (
                    "- All observed classes preserved: "
                    f"`{comparison['all_observed_classes_preserved']}`"
                ),
                (
                    "- Supported-class ratio gate: "
                    f"`{comparison['supported_ratio_gate_pass']}`"
                ),
                (
                    "- Overall class gate: "
                    f"`{comparison['class_preservation_gate_pass']}`"
                ),
                (
                    "- Missing classes: "
                    f"`{comparison['missing_classes']}`"
                ),
                "",
            ]
        )

        for metric in RATIO_METRICS:
            lines.append(
                f"#### {metric}"
            )
            lines.append("")

            class_rows = comparison[
                "metrics"
            ][metric]["classes"]

            for class_name, row in (
                class_rows.items()
            ):
                lines.append(
                    f"- `{class_name}`: "
                    f"full={row['full_count']:,}, "
                    f"selected={row['selected_count']:,}, "
                    f"share Δ="
                    f"{row['absolute_difference_pp']:+.4f} pp"
                )

            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "- Complete Scenario records are retained; "
                "actors are not filtered from selected records."
            ),
            (
                "- Validation is the complete exact-matching "
                "validation intersection."
            ),
            (
                "- Training is deterministic SHA-256 "
                "scenario-ID selection."
            ),
            (
                "- The audit is based on non-SDC actors."
            ),
            (
                "- Forecasting target eligibility uses anchor "
                "validity and at least two valid causal states."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    approved_plan = (
        summary["capacity_estimates"]
        ["reserve_250_gib"]
        [
            "transactional_replace_core_and_remove_unused_branches"
        ]
    )

    expected_training_count = (
        approved_plan[
            "hash_order_plan"
        ]["training_pairs"]
    )

    expected_training_bytes = (
        approved_plan[
            "hash_order_plan"
        ]["training_bytes"]
    )

    training_budget = (
        approved_plan[
            "remaining_training_budget_bytes"
        ]
    )

    expected_validation_count = (
        approved_plan[
            "all_validation_pairs"
        ]
    )

    expected_validation_bytes = (
        approved_plan[
            "all_validation_bytes"
        ]
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    try:
        training_pairs = query_paired_rows(
            connection,
            "training",
        )

        validation_pairs = query_paired_rows(
            connection,
            "validation",
        )

        (
            selected_training,
            selected_training_bytes,
        ) = select_training(
            training_pairs,
            training_budget,
        )

        (
            selected_validation,
            selected_validation_bytes,
        ) = select_validation(
            validation_pairs
        )

        if (
            len(selected_training)
            != expected_training_count
        ):
            raise RuntimeError(
                "Training selection count mismatch: "
                f"{len(selected_training)} "
                f"!= {expected_training_count}"
            )

        if (
            selected_training_bytes
            != expected_training_bytes
        ):
            raise RuntimeError(
                "Training selection byte mismatch: "
                f"{selected_training_bytes} "
                f"!= {expected_training_bytes}"
            )

        if (
            len(selected_validation)
            != expected_validation_count
        ):
            raise RuntimeError(
                "Validation selection count mismatch."
            )

        if (
            selected_validation_bytes
            != expected_validation_bytes
        ):
            raise RuntimeError(
                "Validation selection byte mismatch."
            )

        local_without_lidar, remote_without_local = (
            query_unmatched(connection)
        )

    finally:
        connection.close()

    write_jsonl(
        TRAIN_MANIFEST,
        selected_training,
    )

    write_jsonl(
        VALIDATION_MANIFEST,
        selected_validation,
    )

    write_jsonl(
        LOCAL_WITHOUT_LIDAR,
        local_without_lidar,
    )

    write_jsonl(
        REMOTE_WITHOUT_LOCAL,
        remote_without_local,
    )

    training_paired_ids = {
        row["scenario_id"]
        for row in training_pairs
    }

    validation_paired_ids = {
        row["scenario_id"]
        for row in validation_pairs
    }

    selected_training_ids = {
        row["scenario_id"]
        for row in selected_training
    }

    selected_validation_ids = {
        row["scenario_id"]
        for row in selected_validation
    }

    selection_config = {
        "status":
            "candidate_pending_class_audit",
        "dataset_release":
            "WOMD/WOMD-LiDAR v1.3.0",
        "hard_free_space_reserve_gib":
            250,
        "packaging_margin_fraction":
            approved_plan[
                "packaging_margin_fraction"
            ],
        "association_key":
            "exact_scenario_id",
        "training_selection_policy":
            "deterministic_sha256_scenario_id",
        "validation_selection_policy":
            "all_exact_matching_validation",
        "training_pair_budget_bytes":
            training_budget,
        "training_scenarios":
            len(selected_training),
        "validation_scenarios":
            len(selected_validation),
        "total_scenarios": (
            len(selected_training)
            + len(selected_validation)
        ),
        "training_pair_bytes":
            selected_training_bytes,
        "validation_pair_bytes":
            selected_validation_bytes,
        "total_pair_bytes": (
            selected_training_bytes
            + selected_validation_bytes
        ),
        "class_gate": {
            "all_observed_classes_must_remain":
                True,
            "supported_core_classes":
                sorted(
                    SUPPORTED_CORE_CLASSES
                ),
            "max_absolute_share_difference_pp":
                MAX_SUPPORTED_ABSOLUTE_DIFF_PP,
            "max_relative_share_difference":
                MAX_SUPPORTED_RELATIVE_DIFF,
            "forecast_min_valid_causal_states":
                MIN_FORECAST_CAUSAL_STATES,
        },
    }

    write_json(
        SELECTION_CONFIG,
        selection_config,
    )

    print("Selection reproduced exactly.")
    print(
        "Training:",
        len(selected_training),
        "bytes:",
        selected_training_bytes,
    )
    print(
        "Validation:",
        len(selected_validation),
        "bytes:",
        selected_validation_bytes,
    )
    print(
        "Total:",
        len(selected_training)
        + len(selected_validation),
    )

    split_scans = {}

    split_scans["training"] = scan_split(
        split="training",
        paired_ids=training_paired_ids,
        selected_ids=selected_training_ids,
    )

    split_scans["validation"] = scan_split(
        split="validation",
        paired_ids=validation_paired_ids,
        selected_ids=selected_validation_ids,
    )

    comparisons = {}

    for split in SPLITS:
        comparisons[split] = (
            compare_distributions(
                split_scans[split][
                    "full_paired"
                ],
                split_scans[split][
                    "selected"
                ],
            )
        )

    overall_pass = all(
        comparisons[split][
            "class_preservation_gate_pass"
        ]
        for split in SPLITS
    )

    status = (
        "PASS_FROZEN"
        if overall_pass
        else "BLOCKED_CLASS_REVIEW"
    )

    selection_config["status"] = (
        "frozen_class_audited"
        if overall_pass
        else "blocked_class_audit_review"
    )

    selection_config[
        "class_preservation_gate_pass"
    ] = overall_pass

    write_json(
        SELECTION_CONFIG,
        selection_config,
    )

    audit = {
        "status": status,
        "class_preservation_gate_pass":
            overall_pass,
        "selection": {
            "training_scenarios":
                len(selected_training),
            "validation_scenarios":
                len(selected_validation),
            "total_scenarios": (
                len(selected_training)
                + len(selected_validation)
            ),
            "training_pair_bytes":
                selected_training_bytes,
            "validation_pair_bytes":
                selected_validation_bytes,
            "total_pair_bytes": (
                selected_training_bytes
                + selected_validation_bytes
            ),
        },
        "definitions": {
            "non_sdc_only": True,
            "context_actor": (
                "at least one valid causal state"
            ),
            "anchor_valid": (
                "valid state at current_time_index"
            ),
            "forecasting_target_candidate": (
                "anchor valid and at least two "
                "valid causal states"
            ),
            "scenario_presence": (
                "at least one causal context actor "
                "of the class in the scenario"
            ),
        },
        "thresholds": {
            "supported_classes":
                sorted(
                    SUPPORTED_CORE_CLASSES
                ),
            "max_absolute_difference_pp":
                MAX_SUPPORTED_ABSOLUTE_DIFF_PP,
            "max_relative_difference":
                MAX_SUPPORTED_RELATIVE_DIFF,
        },
        "split_statistics":
            split_scans,
        "class_comparison":
            comparisons,
        "unmatched": {
            "local_motion_without_lidar":
                len(local_without_lidar),
            "remote_lidar_without_local_motion":
                len(remote_without_local),
        },
        "manifests": {
            "training":
                str(TRAIN_MANIFEST),
            "validation":
                str(VALIDATION_MANIFEST),
            "selection_config":
                str(SELECTION_CONFIG),
            "local_without_lidar":
                str(LOCAL_WITHOUT_LIDAR),
            "remote_without_local":
                str(REMOTE_WITHOUT_LOCAL),
        },
        "binding_decision": (
            "D11: complete Scenario records are retained; "
            "all observed classes must remain represented; "
            "supported causal class ratios must pass "
            "the configured tolerance."
        ),
    }

    write_json(
        AUDIT_JSON,
        audit,
    )

    atomic_write_text(
        AUDIT_MD,
        render_markdown(audit),
    )

    print()
    print(
        "Class-preservation gate:",
        status,
    )

    for split in SPLITS:
        comparison = comparisons[split]

        print()
        print(split)
        print(
            "  missing classes:",
            comparison["missing_classes"],
        )
        print(
            "  ratio failures:",
            comparison[
                "supported_ratio_failures"
            ],
        )

    print()
    print("Audit:", AUDIT_JSON)
    print("Summary:", AUDIT_MD)
    print(
        "Selection config:",
        SELECTION_CONFIG,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(
            "\nInterrupted. Completed per-shard class "
            "audit files are resumable."
        )
    except Exception as exc:
        raise SystemExit(
            f"Selection/class audit failed: {exc}"
        ) from exc