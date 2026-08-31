from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/waymo/data/v1_3_0_scenario")
WORK_ROOT = Path("/waymo/iscai_data_prep")

MANIFEST_DIR = WORK_ROOT / "manifests"
REPORT_DIR = WORK_ROOT / "reports"

PRE_DELETE_MANIFEST = (
    MANIFEST_DIR / "approved_branch_deletion_manifest.jsonl"
)

PRE_DELETE_SUMMARY = (
    REPORT_DIR / "approved_branch_deletion_preflight.json"
)

DELETION_REPORT = (
    REPORT_DIR / "approved_branch_deletion_report.json"
)

SELECTION_CONFIG = (
    MANIFEST_DIR / "selection_config.json"
)

CLASS_AUDIT = (
    REPORT_DIR / "class_distribution_audit.json"
)

TRAIN_MANIFEST = (
    MANIFEST_DIR / "selected_training.jsonl"
)

VALIDATION_MANIFEST = (
    MANIFEST_DIR / "selected_validation.jsonl"
)

CANDIDATE_BRANCHES = (
    "testing",
    "training_20s",
    "validation_interactive",
    "testing_interactive",
)

PROTECTED_BRANCHES = {
    "training": 1000,
    "validation": 150,
}

EXPECTED_SELECTED_COUNTS = {
    "training": 72085,
    "validation": 44097,
}

CONFIRMATION_TOKEN = (
    "DELETE_ONLY_APPROVED_OUT_OF_SCOPE_BRANCHES"
)

QUARANTINE_ROOT = (
    DATA_ROOT / ".approved_deletion_quarantine"
)


def human_bytes(value: int | float) -> str:
    value = float(value)

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"

        value /= 1024.0

    raise AssertionError("Unreachable")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())

    temporary.replace(path)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        raise RuntimeError(f"Required manifest missing: {path}")

    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def verify_frozen_selection() -> dict[str, Any]:
    if not SELECTION_CONFIG.is_file():
        raise RuntimeError(
            f"Selection config missing: {SELECTION_CONFIG}"
        )

    if not CLASS_AUDIT.is_file():
        raise RuntimeError(
            f"Class audit missing: {CLASS_AUDIT}"
        )

    selection = json.loads(
        SELECTION_CONFIG.read_text(encoding="utf-8")
    )

    audit = json.loads(
        CLASS_AUDIT.read_text(encoding="utf-8")
    )

    if selection.get("status") != "frozen_class_audited":
        raise RuntimeError(
            "Selection is not frozen_class_audited."
        )

    if selection.get("class_preservation_gate_pass") is not True:
        raise RuntimeError(
            "Selection class-preservation gate did not pass."
        )

    if audit.get("status") != "PASS_FROZEN":
        raise RuntimeError(
            "Class audit status is not PASS_FROZEN."
        )

    if audit.get("class_preservation_gate_pass") is not True:
        raise RuntimeError(
            "Class audit gate is false."
        )

    training_count = count_jsonl(TRAIN_MANIFEST)
    validation_count = count_jsonl(VALIDATION_MANIFEST)

    if training_count != EXPECTED_SELECTED_COUNTS["training"]:
        raise RuntimeError(
            f"Training manifest count changed: {training_count}"
        )

    if validation_count != EXPECTED_SELECTED_COUNTS["validation"]:
        raise RuntimeError(
            f"Validation manifest count changed: {validation_count}"
        )

    return {
        "selection_status": selection["status"],
        "class_audit_status": audit["status"],
        "training_selected": training_count,
        "validation_selected": validation_count,
    }


def verify_protected_branches() -> dict[str, Any]:
    result: dict[str, Any] = {}

    for branch, expected_shards in PROTECTED_BRANCHES.items():
        path = DATA_ROOT / branch

        if not path.is_dir():
            raise RuntimeError(
                f"Protected branch missing: {path}"
            )

        if path.is_symlink():
            raise RuntimeError(
                f"Protected branch is a symlink: {path}"
            )

        actual_shards = len(
            list(path.glob("*.tfrecord-*"))
        )

        if actual_shards != expected_shards:
            raise RuntimeError(
                f"Protected branch {branch} has "
                f"{actual_shards} shards; expected "
                f"{expected_shards}."
            )

        result[branch] = {
            "path": str(path),
            "shards": actual_shards,
            "status": "protected",
        }

    return result


def inventory_candidates() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows: list[dict[str, Any]] = []
    branch_summary: dict[str, Any] = {}

    for branch in CANDIDATE_BRANCHES:
        root = DATA_ROOT / branch

        if not root.is_dir():
            raise RuntimeError(
                f"Approved deletion branch missing: {root}"
            )

        if root.is_symlink():
            raise RuntimeError(
                f"Deletion branch is a symlink: {root}"
            )

        resolved = root.resolve()

        expected = (DATA_ROOT / branch).resolve()

        if resolved != expected:
            raise RuntimeError(
                f"Unexpected resolved path: {root} -> {resolved}"
            )

        branch_files = 0
        branch_bytes = 0

        for directory, subdirs, filenames in os.walk(
            root,
            followlinks=False,
        ):
            directory_path = Path(directory)

            for subdir in subdirs:
                subdir_path = directory_path / subdir

                if subdir_path.is_symlink():
                    raise RuntimeError(
                        f"Symlink directory found: {subdir_path}"
                    )

            for filename in filenames:
                path = directory_path / filename

                if path.is_symlink():
                    raise RuntimeError(
                        f"Symlink file found: {path}"
                    )

                stat = path.stat()

                row = {
                    "branch": branch,
                    "relative_path": str(
                        path.relative_to(DATA_ROOT)
                    ),
                    "absolute_path": str(path),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }

                rows.append(row)

                branch_files += 1
                branch_bytes += stat.st_size

        branch_summary[branch] = {
            "path": str(root),
            "files": branch_files,
            "bytes": branch_bytes,
        }

    rows.sort(
        key=lambda row: (
            row["branch"],
            row["relative_path"],
        )
    )

    return rows, branch_summary


def manifest_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in rows
    )


def preflight() -> dict[str, Any]:
    frozen_selection = verify_frozen_selection()
    protected = verify_protected_branches()

    if QUARANTINE_ROOT.exists():
        raise RuntimeError(
            "Quarantine directory already exists. "
            f"Do not continue automatically: {QUARANTINE_ROOT}"
        )

    rows, branches = inventory_candidates()

    text = manifest_text(rows)

    manifest_sha256 = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    total_files = len(rows)
    total_bytes = sum(
        row["size_bytes"]
        for row in rows
    )

    disk = shutil.disk_usage("/waymo")

    summary = {
        "status": "DRY_RUN_READY",
        "deletion_executed": False,
        "candidate_branches": branches,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "manifest_sha256": manifest_sha256,
        "selection": frozen_selection,
        "protected_branches": protected,
        "disk_before": {
            "total_bytes": disk.total,
            "used_bytes": disk.total - disk.free,
            "free_bytes": disk.free,
        },
        "projected_free_after_deletion_bytes": (
            disk.free + total_bytes
        ),
    }

    atomic_write_text(PRE_DELETE_MANIFEST, text)

    atomic_write_text(
        PRE_DELETE_SUMMARY,
        json.dumps(summary, indent=2) + "\n",
    )

    return summary


def execute_deletion(
    confirmation: str,
) -> None:
    if confirmation != CONFIRMATION_TOKEN:
        raise RuntimeError(
            "Invalid confirmation token."
        )

    if not PRE_DELETE_SUMMARY.is_file():
        raise RuntimeError(
            "Run --dry-run before --execute."
        )

    saved = json.loads(
        PRE_DELETE_SUMMARY.read_text(encoding="utf-8")
    )

    if saved.get("status") != "DRY_RUN_READY":
        raise RuntimeError(
            "Preflight summary is not DRY_RUN_READY."
        )

    # Re-run all safety checks immediately before deletion.
    current = preflight()

    if (
        current["manifest_sha256"]
        != saved["manifest_sha256"]
    ):
        raise RuntimeError(
            "Candidate files changed since the approved dry run."
        )

    if current["total_bytes"] != saved["total_bytes"]:
        raise RuntimeError(
            "Candidate byte total changed since dry run."
        )

    if current["total_files"] != saved["total_files"]:
        raise RuntimeError(
            "Candidate file count changed since dry run."
        )

    QUARANTINE_ROOT.mkdir(
        parents=False,
        exist_ok=False,
    )

    renamed: list[dict[str, str]] = []

    # Atomic rename first. No protected branch is touched.
    for branch in CANDIDATE_BRANCHES:
        source = DATA_ROOT / branch
        destination = QUARANTINE_ROOT / branch

        os.replace(source, destination)

        renamed.append(
            {
                "source": str(source),
                "quarantine": str(destination),
            }
        )

    # Verify training/validation again after atomic renames.
    protected_after_rename = verify_protected_branches()

    for branch in CANDIDATE_BRANCHES:
        quarantined = QUARANTINE_ROOT / branch

        shutil.rmtree(quarantined)

    QUARANTINE_ROOT.rmdir()

    for branch in CANDIDATE_BRANCHES:
        if (DATA_ROOT / branch).exists():
            raise RuntimeError(
                f"Branch still exists after deletion: {branch}"
            )

    protected_final = verify_protected_branches()

    disk_after = shutil.disk_usage("/waymo")

    report = {
        "status": "DELETION_COMPLETE",
        "deletion_executed": True,
        "deleted_branches": list(CANDIDATE_BRANCHES),
        "deleted_files": current["total_files"],
        "deleted_bytes": current["total_bytes"],
        "manifest_sha256": current["manifest_sha256"],
        "manifest_path": str(PRE_DELETE_MANIFEST),
        "renamed_before_deletion": renamed,
        "protected_after_rename": protected_after_rename,
        "protected_final": protected_final,
        "disk_before": current["disk_before"],
        "disk_after": {
            "total_bytes": disk_after.total,
            "used_bytes": disk_after.total - disk_after.free,
            "free_bytes": disk_after.free,
        },
        "timestamp_unix": time.time(),
        "bulk_lidar_download_started": False,
        "paired_migration_started": False,
    }

    atomic_write_text(
        DELETION_REPORT,
        json.dumps(report, indent=2) + "\n",
    )

    print()
    print("APPROVED BRANCH DELETION: COMPLETE")
    print(
        "Deleted:",
        ", ".join(CANDIDATE_BRANCHES),
    )
    print(
        "Freed manifest bytes:",
        human_bytes(current["total_bytes"]),
    )
    print(
        "Free space now:",
        human_bytes(disk_after.free),
    )
    print(
        "Protected training shards:",
        protected_final["training"]["shards"],
    )
    print(
        "Protected validation shards:",
        protected_final["validation"]["shards"],
    )
    print("Report:", DELETION_REPORT)


def main() -> None:
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
    )

    mode.add_argument(
        "--execute",
        action="store_true",
    )

    parser.add_argument(
        "--confirmation",
        default="",
    )

    args = parser.parse_args()

    if args.dry_run:
        summary = preflight()

        print("Deletion preflight: DRY_RUN_READY")
        print(
            "Branches:",
            ", ".join(CANDIDATE_BRANCHES),
        )
        print("Files:", summary["total_files"])
        print(
            "Bytes:",
            summary["total_bytes"],
            f"({human_bytes(summary['total_bytes'])})",
        )
        print(
            "Current free:",
            human_bytes(
                summary["disk_before"]["free_bytes"]
            ),
        )
        print(
            "Projected free:",
            human_bytes(
                summary[
                    "projected_free_after_deletion_bytes"
                ]
            ),
        )
        print(
            "Manifest SHA-256:",
            summary["manifest_sha256"],
        )
        print("Summary:", PRE_DELETE_SUMMARY)
        print("No files were deleted.")

    else:
        execute_deletion(args.confirmation)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(
            f"Guarded deletion failed: {exc}"
        ) from exc