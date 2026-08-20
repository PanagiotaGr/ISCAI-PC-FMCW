from pathlib import Path
import json

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)


DATA = Path(
    "/home/agni/waymo/data/paired_womd_lidar_v1_3_0/training/motion"
)


MAX_FILES = 50


counts = {
    "vehicle": 0,
    "pedestrian": 0,
    "cyclist": 0,
    "unknown": 0,
}


scenarios = 0
files_checked = 0


def map_type(track):

    try:
        t = int(track.object_type)

    except Exception:
        return "unknown"

    if t == 1:
        return "vehicle"

    if t == 2:
        return "pedestrian"

    if t == 3:
        return "cyclist"

    return "unknown"



for file in DATA.rglob("*.tfrecord-*"):

    if files_checked >= MAX_FILES:
        break

    files_checked += 1

    try:
        scenario = read_first_scenario(
            file
        )

    except Exception:
        continue


    scenarios += 1


    for track in scenario.tracks:

        counts[
            map_type(track)
        ] += 1



report = {
    "files_checked": files_checked,
    "scenarios_checked": scenarios,
    "counts": counts,
}


Path(
    "reports"
).mkdir(
    exist_ok=True
)


Path(
    "reports/block3_multiclass_dataset_scan.json"
).write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage3 Multiclass Dataset Scan ====="
)

print(
    "files =",
    files_checked
)

print(
    "scenarios =",
    scenarios
)

print(
    "vehicle =",
    counts["vehicle"]
)

print(
    "pedestrian =",
    counts["pedestrian"]
)

print(
    "cyclist =",
    counts["cyclist"]
)
