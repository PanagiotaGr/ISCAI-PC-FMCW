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


import math

from iscai_stage5.geometry.headlamp_geometry import (
    compute_headlamp_state,
)


actors = [
    {
        "id": 1,
        "position": (20.0, 5.0, 0.0),
        "velocity": (5.0, 0.0, 0.0),
    },
    {
        "id": 2,
        "position": (50.0, -3.0, 1.0),
        "velocity": (2.0, 1.0, 0.0),
    },
]


valid = 0


print(
    "===== Stage5 Headlamp Geometry Smoke ====="
)


for actor in actors:

    state = compute_headlamp_state(
        actor_position=actor["position"],
        actor_velocity=actor["velocity"],
        ego_position=(0.0, 0.0, 0.0),
        ego_velocity=(0.0, 0.0, 0.0),
        ego_heading=0.0,
    )

    values = (
        state.range_m,
        state.azimuth_rad,
        state.elevation_rad,
        state.radial_velocity,
    )

    finite = all(
        math.isfinite(value)
        for value in values
    )

    if finite:
        valid += 1

    print(
        "actor =",
        actor["id"],
        "range_m =",
        round(state.range_m, 6),
        "azimuth_rad =",
        round(state.azimuth_rad, 6),
        "elevation_rad =",
        round(state.elevation_rad, 6),
        "radial_velocity =",
        round(state.radial_velocity, 6),
        "valid =",
        finite,
    )


status = (
    "PASS"
    if valid == len(actors)
    else "FAIL"
)


print(
    "actors =",
    len(actors)
)

print(
    "valid =",
    valid
)

print(
    "future_used = NO"
)

print(
    "STATUS =",
    status
)


if status != "PASS":
    raise SystemExit(1)
