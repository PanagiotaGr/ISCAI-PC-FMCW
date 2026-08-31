from __future__ import annotations

import argparse
import hashlib
import fcntl
import json
import os
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path
from typing import Any

from waymo_open_dataset.protos import scenario_pb2


SOURCE_ROOT = Path("/waymo/data/v1_3_0_scenario")
WORK_ROOT = Path("/waymo/iscai_data_prep")
PAIRED_ROOT = Path("/waymo/data/paired_womd_lidar_v1_3_0")

INVENTORY_DB = WORK_ROOT / "reports/paired_inventory.sqlite"
SELECTION_CONFIG = WORK_ROOT / "manifests/selection_config.json"
MANIFESTS = {
    "training": WORK_ROOT / "manifests/selected_training.jsonl",
    "validation": WORK_ROOT / "manifests/selected_validation.jsonl",
}

REPORT_ROOT = PAIRED_ROOT / "reports/shards"
TRANSACTION_ROOT = PAIRED_ROOT / ".transactions"

HARD_RESERVE_BYTES = 250 * 1024**3
TEMP_SAFETY_BYTES = 1 * 1024**3
DOWNLOAD_BATCH_SIZE = 64
DELETE_TOKEN = "DELETE_SOURCE_AFTER_VERIFIED_PAIRED_COMMIT"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Missing manifest: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_selected(split: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(MANIFESTS[split]):
        grouped.setdefault(row["source_shard"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["record_offset"]))
    return grouped


def source_shards(split: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(INVENTORY_DB)
    try:
        rows = connection.execute(
            """
            SELECT shard_path, size_bytes, record_count
            FROM local_shards
            WHERE split = ? AND scan_complete = 1
            ORDER BY shard_path
            """,
            (split,),
        ).fetchall()
    finally:
        connection.close()

    return [
        {
            "path": str(row[0]),
            "size_bytes": int(row[1]),
            "record_count": int(row[2]),
        }
        for row in rows
    ]


def shard_key(source: Path) -> str:
    return source.name.replace("/", "_")


def read_record_bytes(source: Path, row: dict[str, Any]) -> bytes:
    expected_record_bytes = int(row["motion_record_bytes"])
    expected_payload = int(row["payload_length"])

    with source.open("rb") as f:
        f.seek(int(row["record_offset"]))
        record = f.read(expected_record_bytes)

    if len(record) != expected_record_bytes:
        raise RuntimeError(
            f"Short record read for {row['scenario_id']} in {source}"
        )

    if len(record) < 16:
        raise RuntimeError("Invalid TFRecord record length.")

    payload_length = struct.unpack("<Q", record[:8])[0]
    if payload_length != expected_payload:
        raise RuntimeError(
            f"Payload length mismatch for {row['scenario_id']}"
        )

    payload = record[12 : 12 + payload_length]
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(payload)

    if scenario.scenario_id != row["scenario_id"]:
        raise RuntimeError(
            f"Motion scenario ID mismatch: "
            f"{scenario.scenario_id} != {row['scenario_id']}"
        )

    return record


def write_compact_motion(
    source: Path,
    rows: list[dict[str, Any]],
    output_partial: Path,
) -> dict[str, Any]:
    output_partial.parent.mkdir(parents=True, exist_ok=True)

    expected_ids = []
    record_hashes = []

    with output_partial.open("wb") as out:
        for row in rows:
            record = read_record_bytes(source, row)
            out.write(record)
            expected_ids.append(row["scenario_id"])
            record_hashes.append(hashlib.sha256(record).hexdigest())

        out.flush()
        os.fsync(out.fileno())

    expected_bytes = sum(int(row["motion_record_bytes"]) for row in rows)
    if output_partial.stat().st_size != expected_bytes:
        raise RuntimeError("Compact motion size mismatch.")

    verified_ids = []
    verified_hashes = []

    with output_partial.open("rb") as f:
        while True:
            header = f.read(12)
            if not header:
                break
            if len(header) != 12:
                raise RuntimeError("Truncated compact TFRecord header.")

            payload_length = struct.unpack("<Q", header[:8])[0]
            payload = f.read(payload_length)
            data_crc = f.read(4)
            if len(payload) != payload_length or len(data_crc) != 4:
                raise RuntimeError("Truncated compact TFRecord record.")

            record = header + payload + data_crc
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(payload)
            verified_ids.append(scenario.scenario_id)
            verified_hashes.append(hashlib.sha256(record).hexdigest())

    if verified_ids != expected_ids:
        raise RuntimeError("Compact motion scenario ordering/ID mismatch.")

    if verified_hashes != record_hashes:
        raise RuntimeError("Compact motion record-byte mismatch.")

    return {
        "records": len(expected_ids),
        "bytes": output_partial.stat().st_size,
        "sha256": sha256_file(output_partial),
        "scenario_ids_sha256": hashlib.sha256(
            ("\n".join(expected_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "record_hashes_sha256": hashlib.sha256(
            ("\n".join(record_hashes) + "\n").encode("utf-8")
        ).hexdigest(),
    }


def read_single_scenario_tfrecord(path: Path) -> scenario_pb2.Scenario:
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) != 12:
            raise RuntimeError(f"Invalid LiDAR TFRecord header: {path}")

        payload_length = struct.unpack("<Q", header[:8])[0]
        payload = f.read(payload_length)
        data_crc = f.read(4)
        trailing = f.read(1)

    if len(payload) != payload_length or len(data_crc) != 4:
        raise RuntimeError(f"Truncated LiDAR TFRecord: {path}")

    if trailing:
        raise RuntimeError(f"Expected one LiDAR record only: {path}")

    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(payload)
    return scenario


def verify_lidar(
    path: Path,
    row: dict[str, Any],
    motion_scenario: scenario_pb2.Scenario,
) -> dict[str, Any]:
    expected_id = str(row["scenario_id"])
    expected_name = f"{expected_id}.tfrecord"
    source_uri = str(row["lidar_uri"])

    # WOMD-LiDAR sidecars normally do not populate Scenario.scenario_id.
    # Their identity is defined by the official {scenario_id}.tfrecord
    # filename and by the exact URI stored in the frozen manifest.
    if path.name != expected_name:
        raise RuntimeError(
            f"LiDAR local filename mismatch: "
            f"{path.name} != {expected_name}"
        )

    remote_name = source_uri.rstrip("/").rsplit("/", 1)[-1]

    if remote_name != expected_name:
        raise RuntimeError(
            f"LiDAR manifest URI mismatch: "
            f"{remote_name} != {expected_name}"
        )

    if motion_scenario.scenario_id != expected_id:
        raise RuntimeError(
            f"Motion scenario ID mismatch during LiDAR verification: "
            f"{motion_scenario.scenario_id} != {expected_id}"
        )

    expected_size = int(row["lidar_bytes"])

    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"LiDAR size mismatch for {expected_id}: "
            f"{path.stat().st_size} != {expected_size}"
        )

    lidar = read_single_scenario_tfrecord(path)

    embedded_id = (
        lidar.scenario_id
        if lidar.HasField("scenario_id")
        else None
    )

    # Accept the normal sidecar case where scenario_id is absent.
    # If a future sidecar does contain it, it must match exactly.
    if embedded_id is not None and embedded_id != expected_id:
        raise RuntimeError(
            f"Embedded LiDAR scenario ID mismatch: "
            f"{embedded_id} != {expected_id}"
        )

    if not lidar.compressed_frame_laser_data:
        raise RuntimeError(
            f"LiDAR sidecar has no compressed frames: {expected_id}"
        )

    expected_frames = int(
        motion_scenario.current_time_index
    ) + 1

    actual_frames = len(
        lidar.compressed_frame_laser_data
    )

    if actual_frames != expected_frames:
        raise RuntimeError(
            f"LiDAR frame mismatch for {expected_id}: "
            f"{actual_frames} != {expected_frames}"
        )

    timestamp_mode = "index_contract"
    max_timestamp_error = None

    if lidar.timestamps_seconds:
        lidar_ts = list(
            lidar.timestamps_seconds
        )

        motion_ts = list(
            motion_scenario.timestamps_seconds[
                :expected_frames
            ]
        )

        if len(lidar_ts) != expected_frames:
            raise RuntimeError(
                f"LiDAR timestamp count mismatch for {expected_id}"
            )

        max_timestamp_error = max(
            abs(float(a) - float(b))
            for a, b in zip(
                lidar_ts,
                motion_ts,
            )
        )

        if max_timestamp_error > 1e-6:
            raise RuntimeError(
                f"Timestamp mismatch for {expected_id}: "
                f"{max_timestamp_error}"
            )

        timestamp_mode = "embedded_exact"

    identity_mode = (
        "embedded_scenario_id"
        if embedded_id is not None
        else "manifest_uri_and_filename"
    )

    return {
        "scenario_id": expected_id,
        "identity_mode": identity_mode,
        "embedded_scenario_id": embedded_id,
        "source_uri": source_uri,
        "bytes": expected_size,
        "sha256": sha256_file(path),
        "frames": actual_frames,
        "timestamp_mode": timestamp_mode,
        "max_timestamp_error_seconds":
            max_timestamp_error,
    }


def ensure_space(required_new_bytes: int) -> None:
    free = shutil.disk_usage("/waymo").free
    required = HARD_RESERVE_BYTES + TEMP_SAFETY_BYTES + required_new_bytes

    if free < required:
        raise RuntimeError(
            "Hard reserve would be violated. "
            f"free={free}, required={required}"
        )


def download_missing(
    rows: list[dict[str, Any]],
    temp_lidar_dir: Path,
    transfer_dir: Path,
) -> None:
    temp_lidar_dir.mkdir(parents=True, exist_ok=True)
    transfer_dir.mkdir(parents=True, exist_ok=True)

    missing = []

    for row in rows:
        path = temp_lidar_dir / f"{row['scenario_id']}.tfrecord"

        if path.is_file() and path.stat().st_size == int(row["lidar_bytes"]):
            continue

        if path.exists():
            path.unlink()

        missing.append(row)

    for batch_index in range(0, len(missing), DOWNLOAD_BATCH_SIZE):
        batch = missing[batch_index : batch_index + DOWNLOAD_BATCH_SIZE]
        manifest_path = transfer_dir / f"gcloud_batch_{batch_index // DOWNLOAD_BATCH_SIZE:04d}.csv"

        if manifest_path.exists():
            manifest_path.unlink()

        command = [
            "gcloud",
            "storage",
            "cp",
            "--read-paths-from-stdin",
            f"--manifest-path={manifest_path}",
            str(temp_lidar_dir) + "/",
        ]

        subprocess.run(
            command,
            input="\n".join(row["lidar_uri"] for row in batch) + "\n",
            text=True,
            check=True,
        )

        for row in batch:
            path = temp_lidar_dir / f"{row['scenario_id']}.tfrecord"
            if not path.is_file():
                raise RuntimeError(f"Missing downloaded file: {path}")
            if path.stat().st_size != int(row["lidar_bytes"]):
                raise RuntimeError(f"Downloaded size mismatch: {path}")


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_committed_outputs(report: dict[str, Any]) -> None:
    motion_output = report.get("motion_output")
    motion = report.get("motion", {})

    if motion_output:
        path = Path(motion_output)
        if not path.is_file():
            raise RuntimeError(f"Committed motion output missing: {path}")
        if path.stat().st_size != int(motion["bytes"]):
            raise RuntimeError(f"Committed motion size mismatch: {path}")
        if sha256_file(path) != motion["sha256"]:
            raise RuntimeError(f"Committed motion checksum mismatch: {path}")

    split = report["split"]
    for item in report.get("lidar", []):
        scenario_id = item["scenario_id"]
        path = (
            PAIRED_ROOT
            / split
            / "lidar"
            / scenario_id[:2]
            / f"{scenario_id}.tfrecord"
        )
        if not path.is_file():
            raise RuntimeError(f"Committed LiDAR missing: {path}")
        if path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"Committed LiDAR size mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Committed LiDAR checksum mismatch: {path}")


def process_shard(
    split: str,
    shard_info: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    *,
    delete_source: bool,
) -> dict[str, Any]:
    source = Path(shard_info["path"])
    key = shard_key(source)
    report_path = REPORT_ROOT / split / f"{key}.json"

    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = report.get("status")

        if status == "COMMITTED_SOURCE_DELETED":
            if source.exists():
                raise RuntimeError(
                    f"Committed source unexpectedly exists: {source}"
                )
            verify_committed_outputs(report)
            return report

        if status in {
            "COMMITTED_SOURCE_DELETION_AUTHORIZED",
            "COMMITTED_SOURCE_RETAINED",
        }:
            verify_committed_outputs(report)

            if delete_source:
                if source.exists():
                    if source.stat().st_size != int(report["source_size_bytes"]):
                        raise RuntimeError(f"Source size changed: {source}")
                    source.unlink()
                    fsync_directory(source.parent)

                report["status"] = "COMMITTED_SOURCE_DELETED"
                report["source_deleted"] = True
                report["free_bytes_after"] = shutil.disk_usage("/waymo").free
                atomic_json(report_path, report)
                return report

            return report

    if not source.is_file():
        raise RuntimeError(
            f"Source shard missing without completed report: {source}"
        )

    if source.stat().st_size != int(shard_info["size_bytes"]):
        raise RuntimeError(f"Source shard size changed: {source}")

    selected_rows = sorted(
        selected_rows,
        key=lambda row: int(row["record_offset"]),
    )

    output_motion_dir = PAIRED_ROOT / split / "motion"
    output_motion_dir.mkdir(parents=True, exist_ok=True)
    final_motion = output_motion_dir / f"paired-from-{source.name}"

    transaction_dir = TRANSACTION_ROOT / split / key
    transaction_dir.mkdir(parents=True, exist_ok=True)

    partial_motion = transaction_dir / (final_motion.name + ".partial")
    temp_lidar_dir = transaction_dir / "lidar"
    transfer_dir = transaction_dir / "transfers"

    estimated_new_bytes = sum(int(row["pair_bytes"]) for row in selected_rows)
    ensure_space(estimated_new_bytes)

    if selected_rows:
        motion_meta = write_compact_motion(
            source,
            selected_rows,
            partial_motion,
        )
    else:
        motion_meta = {
            "records": 0,
            "bytes": 0,
            "sha256": None,
            "scenario_ids_sha256": hashlib.sha256(b"").hexdigest(),
            "record_hashes_sha256": hashlib.sha256(b"").hexdigest(),
        }

    motion_scenarios: dict[str, scenario_pb2.Scenario] = {}
    for row in selected_rows:
        record = read_record_bytes(source, row)
        payload_length = struct.unpack("<Q", record[:8])[0]
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(record[12 : 12 + payload_length])
        motion_scenarios[row["scenario_id"]] = scenario

    download_missing(selected_rows, temp_lidar_dir, transfer_dir)

    lidar_meta = []
    for row in selected_rows:
        path = temp_lidar_dir / f"{row['scenario_id']}.tfrecord"
        lidar_meta.append(
            verify_lidar(path, row, motion_scenarios[row["scenario_id"]])
        )

    if selected_rows:
        if final_motion.exists():
            if sha256_file(final_motion) != motion_meta["sha256"]:
                raise RuntimeError(f"Existing final motion mismatch: {final_motion}")
            partial_motion.unlink(missing_ok=True)
        else:
            os.replace(partial_motion, final_motion)
            fsync_directory(final_motion.parent)

    for item in lidar_meta:
        scenario_id = item["scenario_id"]
        source_lidar = temp_lidar_dir / f"{scenario_id}.tfrecord"
        final_dir = PAIRED_ROOT / split / "lidar" / scenario_id[:2]
        final_dir.mkdir(parents=True, exist_ok=True)
        final_lidar = final_dir / f"{scenario_id}.tfrecord"

        if final_lidar.exists():
            if (
                final_lidar.stat().st_size != item["bytes"]
                or sha256_file(final_lidar) != item["sha256"]
            ):
                raise RuntimeError(f"Existing final LiDAR mismatch: {final_lidar}")
            source_lidar.unlink(missing_ok=True)
        else:
            os.replace(source_lidar, final_lidar)
            fsync_directory(final_dir)

    committed = {
        "status": "COMMITTED_SOURCE_DELETION_AUTHORIZED",
        "split": split,
        "source_shard": str(source),
        "source_size_bytes": int(shard_info["size_bytes"]),
        "selected_scenarios": len(selected_rows),
        "motion_output": str(final_motion) if selected_rows else None,
        "motion": motion_meta,
        "lidar_total_bytes": sum(item["bytes"] for item in lidar_meta),
        "lidar_embedded_timestamp_scenarios": sum(
            item["timestamp_mode"] == "embedded_exact" for item in lidar_meta
        ),
        "lidar_index_contract_scenarios": sum(
            item["timestamp_mode"] == "index_contract" for item in lidar_meta
        ),
        "lidar": lidar_meta,
        "hard_reserve_bytes": HARD_RESERVE_BYTES,
    }
    atomic_json(report_path, committed)

    if delete_source:
        source.unlink()
        fsync_directory(source.parent)

        committed["status"] = "COMMITTED_SOURCE_DELETED"
        committed["source_deleted"] = True
        committed["free_bytes_after"] = shutil.disk_usage("/waymo").free
        atomic_json(report_path, committed)
    else:
        committed["status"] = "COMMITTED_SOURCE_RETAINED"
        committed["source_deleted"] = False
        atomic_json(report_path, committed)

    shutil.rmtree(transaction_dir, ignore_errors=True)
    return committed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("training", "validation", "all"),
        default="validation",
    )
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()

    lock_path = WORK_ROOT / "reports/paired_migration.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another migration process is already running.") from exc

    config = json.loads(SELECTION_CONFIG.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_class_audited":
        raise RuntimeError("Selection is not frozen.")
    if config.get("class_preservation_gate_pass") is not True:
        raise RuntimeError("Class-preservation gate is not PASS.")

    if args.delete_source and args.confirmation != DELETE_TOKEN:
        raise RuntimeError("Invalid source-deletion confirmation token.")

    splits = ("validation", "training") if args.split == "all" else (args.split,)

    planned = []
    for split in splits:
        selected_by_shard = load_selected(split)
        for info in source_shards(split):
            source = info["path"]
            rows = selected_by_shard.get(source, [])
            report_path = REPORT_ROOT / split / f"{shard_key(Path(source))}.json"

            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("status") == "COMMITTED_SOURCE_DELETED":
                    continue

            planned.append((split, info, rows))

    if args.max_shards > 0:
        planned = planned[: args.max_shards]

    total_pair_bytes = sum(
        sum(int(row["pair_bytes"]) for row in rows)
        for _, _, rows in planned
    )

    print("Planned shards:", len(planned))
    print("Estimated new paired bytes:", total_pair_bytes)
    print("Free bytes now:", shutil.disk_usage("/waymo").free)
    print("Hard reserve bytes:", HARD_RESERVE_BYTES)

    for split, info, rows in planned:
        print(
            split,
            info["path"],
            "selected_scenarios=",
            len(rows),
            "pair_bytes=",
            sum(int(row["pair_bytes"]) for row in rows),
        )

    if args.plan_only:
        print("PLAN ONLY: no files changed.")
        return

    completed = 0
    for split, info, rows in planned:
        report = process_shard(
            split,
            info,
            rows,
            delete_source=args.delete_source,
        )
        completed += 1
        print(
            "COMPLETED",
            split,
            info["path"],
            report["status"],
            "selected=",
            report["selected_scenarios"],
            "free=",
            shutil.disk_usage("/waymo").free,
        )

    print("Completed shards:", completed)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Transactional migration failed: {exc}") from exc
