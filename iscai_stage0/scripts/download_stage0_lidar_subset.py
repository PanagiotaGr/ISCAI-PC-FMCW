from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from iscai_stage0.womd_proto_io import iter_scenarios


SHARD = Path(
    "/waymo/data/v1_3_0_scenario/validation/"
    "validation.tfrecord-00000-of-00150"
)

BUCKET = (
    "gs://waymo_open_dataset_motion_v_1_3_0/"
    "uncompressed/lidar_and_camera/validation"
)

OUT_DIR = Path(
    "/waymo/data/womd_lidar_stage0/validation"
)

REPORT = Path(
    "/waymo/iscai_stage0/reports/stage0/"
    "lidar_subset_manifest.json"
)

PRIMARY_ID = "b85e1bd6cc8e74c0"

REQUIRED = {
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
}

TYPE_NAMES = {
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
}

MAX_DOWNLOAD = 2 * 1024**3
MIN_FREE_AFTER = 20 * 1024**3


def classes_at_anchor(scenario) -> set[str]:
    anchor = scenario.current_time_index
    result = set()

    for i, track in enumerate(scenario.tracks):
        if i == scenario.sdc_track_index:
            continue

        if anchor >= len(track.states):
            continue

        if not track.states[anchor].valid:
            continue

        name = TYPE_NAMES.get(track.object_type)

        if name:
            result.add(name)

    return result


def remote_size(uri: str) -> int:
    proc = subprocess.run(
        ["gcloud", "storage", "ls", "-l", uri],
        check=True,
        text=True,
        capture_output=True,
    )

    for line in proc.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+", line)
        if match:
            return int(match.group(1))

    raise RuntimeError(
        f"Could not determine remote size for {uri}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    candidates = []

    for scenario in iter_scenarios(SHARD, limit=100):
        candidates.append(
            (
                scenario.scenario_id,
                classes_at_anchor(scenario),
            )
        )

    by_id = {
        sid: classes
        for sid, classes in candidates
    }

    if PRIMARY_ID not in by_id:
        raise RuntimeError(
            f"Primary scenario {PRIMARY_ID} not found."
        )

    selected = [PRIMARY_ID]
    covered = set(by_id[PRIMARY_ID])

    while not REQUIRED.issubset(covered):
        missing = REQUIRED - covered

        remaining = [
            item
            for item in candidates
            if item[0] not in selected
        ]

        best = max(
            remaining,
            key=lambda item: len(item[1] & missing),
        )

        gain = best[1] & missing

        if not gain:
            raise RuntimeError(
                f"Could not cover classes: {sorted(missing)}"
            )

        selected.append(best[0])
        covered.update(best[1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    objects = []
    total_remote = 0

    for sid in selected:
        uri = f"{BUCKET}/{sid}.tfrecord"
        size = remote_size(uri)

        total_remote += size

        objects.append(
            {
                "scenario_id": sid,
                "classes_at_anchor": sorted(by_id[sid]),
                "uri": uri,
                "remote_bytes": size,
            }
        )

    if total_remote > MAX_DOWNLOAD:
        raise RuntimeError(
            "Selected Stage-0 subset exceeds 2 GiB cap: "
            f"{total_remote / 1024**3:.2f} GiB"
        )

    free = shutil.disk_usage("/waymo").free

    if free - total_remote < MIN_FREE_AFTER:
        raise RuntimeError(
            "Storage safety margin would be violated."
        )

    for item in objects:
        sid = item["scenario_id"]
        path = OUT_DIR / f"{sid}.tfrecord"

        if not path.exists():
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "cp",
                    item["uri"],
                    str(path),
                ],
                check=True,
            )

        item["local_path"] = str(path)
        item["local_bytes"] = path.stat().st_size
        item["sha256"] = sha256(path)

        if item["local_bytes"] != item["remote_bytes"]:
            raise RuntimeError(
                f"Size mismatch for {sid}"
            )

    result = {
        "dataset_release": "WOMD v1.3.0",
        "split": "validation",
        "selection": (
            "deterministic minimal class-covering subset "
            "from first 100 scenarios of validation shard 00000"
        ),
        "required_classes": sorted(REQUIRED),
        "covered_classes": sorted(covered),
        "download_cap_bytes": MAX_DOWNLOAD,
        "total_download_bytes": total_remote,
        "objects": objects,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    REPORT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Selected:", selected)
    print("Covered:", sorted(covered))
    print(
        "Total:",
        f"{total_remote / 1024**2:.1f} MiB",
    )
    print("Manifest:", REPORT)


if __name__ == "__main__":
    main()