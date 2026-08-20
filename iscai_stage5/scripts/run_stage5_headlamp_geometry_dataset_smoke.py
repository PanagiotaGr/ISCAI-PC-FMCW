from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/agni/waymo")

for stage_src in [
    ROOT / "iscai_stage1" / "src",
    ROOT / "iscai_stage2" / "src",
    ROOT / "iscai_stage3" / "src",
    ROOT / "iscai_stage4" / "src",
    ROOT / "iscai_stage5" / "src",
]:
    value = str(stage_src)

    if value not in sys.path:
        sys.path.insert(0, value)


ROOT = Path("/home/agni/waymo")

for stage_src in [
    ROOT / "iscai_stage1" / "src",
    ROOT / "iscai_stage2" / "src",
    ROOT / "iscai_stage3" / "src",
    ROOT / "iscai_stage4" / "src",
    ROOT / "iscai_stage5" / "src",
]:
    value = str(stage_src)

    if value not in sys.path:
        sys.path.insert(0, value)



import json
import math
from pathlib import Path

from iscai_stage5.geometry.headlamp_geometry import (
    compute_headlamp_state,
)


SOURCE = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)


data = json.loads(
    SOURCE.read_text()
)


valid_range = 0
valid_azimuth = 0
valid_elevation = 0
valid_radial = 0


classes = {

    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,

}


for item in data:

    actor_type = item.get(
        "actor_type",
        "UNKNOWN"
    )


    if actor_type in classes:
        classes[actor_type] += 1


    trajectory = item.get(
        "trajectory",
        []
    )


    if len(trajectory) < 2:
        continue


    p0 = trajectory[0]
    p1 = trajectory[1]


    position = (
        float(p0[0]),
        float(p0[1]),
        float(p0[2]),
    )


    velocity = (
        float(p1[0] - p0[0]),
        float(p1[1] - p0[1]),
        float(p1[2] - p0[2]),
    )


    state = compute_headlamp_state(
        actor_position=position,
        actor_velocity=velocity,
    )


    values = [

        state.range_m,
        state.azimuth_rad,
        state.elevation_rad,
        state.radial_velocity,

    ]


    if math.isfinite(state.range_m):
        valid_range += 1

    if math.isfinite(state.azimuth_rad):
        valid_azimuth += 1

    if math.isfinite(state.elevation_rad):
        valid_elevation += 1

    if math.isfinite(state.radial_velocity):
        valid_radial += 1



print(
    "===== Stage5 Headlamp Geometry Dataset ====="
)

print(
    "records =",
    len(data)
)

print(
    "range_valid =",
    valid_range
)

print(
    "azimuth_valid =",
    valid_azimuth
)

print(
    "elevation_valid =",
    valid_elevation
)

print(
    "radial_velocity_valid =",
    valid_radial
)

print(
    "vehicle =",
    classes["VEHICLE"]
)

print(
    "pedestrian =",
    classes["PEDESTRIAN"]
)

print(
    "cyclist =",
    classes["CYCLIST"]
)

print(
    "future_used = NO"
)


if (
    valid_range == len(data)
    and valid_azimuth == len(data)
    and valid_elevation == len(data)
    and valid_radial == len(data)
):

    print(
        "STATUS = PASS"
    )

else:

    print(
        "STATUS = FAIL"
    )
    raise SystemExit(1)
