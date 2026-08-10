from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from iscai_stage0.womd_proto_io import read_first_scenario
from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)
from iscai_stage1.maps.womd_adapter import (
    adapt_causal_womd_map,
)


SID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

OUT_DIR = Path(
    "/home/agni/waymo/iscai_stage1/"
    "artifacts/stage1a/visualizations"
)

HEADLAMP_PNG = OUT_DIR / "pilot_headlamp_centric.png"
ELEVATION_PNG = OUT_DIR / "pilot_elevation_3d.png"

REPORT = Path(
    "/home/agni/waymo/iscai_stage1/"
    "reports/stage1a/block8a_visual_gate.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


scenario = read_first_scenario(MOTION)

if scenario.scenario_id != SID:
    raise RuntimeError(
        f"Unexpected pilot scenario {scenario.scenario_id}"
    )

actors = adapt_causal_womd_scenario(
    scenario
)

maps = adapt_causal_womd_map(
    scenario,
    actors.frames.T_H0_from_W,
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. Headlamp-centric top-down view
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)

for feature in maps.static_features:
    points = feature.points_H0_m

    if not points:
        continue

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    if feature.kind in {
        "crosswalk",
        "speed_bump",
        "driveway",
    } and len(points) >= 3:
        xs = xs + [xs[0]]
        ys = ys + [ys[0]]

    ax.plot(
        xs,
        ys,
        linewidth=0.7,
        alpha=0.55,
    )


anchor_actor_count = 0
receiver_count = 0

for actor in actors.actors:
    history = actor.artifact.history

    causal_H0 = []

    for valid, p_W in zip(
        history.state_valid,
        history.position_W_m,
    ):
        if not valid:
            continue

        causal_H0.append(
            actors.frames.T_H0_from_W.apply_point(
                p_W
            )
        )

    if causal_H0:
        ax.plot(
            [p[0] for p in causal_H0],
            [p[1] for p in causal_H0],
            linewidth=1.2,
        )

    if actor.anchor_center_H0_m is not None:
        anchor_actor_count += 1

        p = actor.anchor_center_H0_m

        if actor.artifact.roles.is_receiver_candidate:
            receiver_count += 1
            marker = "s"
        elif actor.artifact.metadata.is_sdc:
            marker = "x"
        else:
            marker = "o"

        ax.scatter(
            [p[0]],
            [p[1]],
            marker=marker,
            s=35,
        )


# H0 origin / +x forward direction.
ax.scatter(
    [0.0],
    [0.0],
    marker="*",
    s=130,
)

ax.arrow(
    0.0,
    0.0,
    10.0,
    0.0,
    width=0.12,
    length_includes_head=True,
)

ax.set_title(
    f"Stage1A headlamp-centric causal scene\n{SID}"
)
ax.set_xlabel(
    "H0 x [m] — forward"
)
ax.set_ylabel(
    "H0 y [m] — left"
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.grid(True)

fig.tight_layout()

fig.savefig(
    HEADLAMP_PNG,
    dpi=160,
)

plt.close(fig)


# ============================================================
# 2. 3-D / elevation gate
# ============================================================

fig = plt.figure(
    figsize=(11, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

for feature in maps.static_features:
    points = feature.points_H0_m

    if len(points) < 2:
        continue

    ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        [p[2] for p in points],
        linewidth=0.7,
        alpha=0.55,
    )


for actor in actors.actors:
    history = actor.artifact.history

    points_H0 = []

    for valid, p_W in zip(
        history.state_valid,
        history.position_W_m,
    ):
        if not valid:
            continue

        points_H0.append(
            actors.frames.T_H0_from_W.apply_point(
                p_W
            )
        )

    if points_H0:
        ax.plot(
            [p[0] for p in points_H0],
            [p[1] for p in points_H0],
            [p[2] for p in points_H0],
            linewidth=1.2,
        )

    if actor.anchor_center_H0_m is not None:
        p = actor.anchor_center_H0_m

        ax.scatter(
            [p[0]],
            [p[1]],
            [p[2]],
            s=28,
        )


ax.scatter(
    [0.0],
    [0.0],
    [0.0],
    marker="*",
    s=130,
)

ax.set_title(
    f"Stage1A H0 3-D/elevation causal scene\n{SID}"
)

ax.set_xlabel(
    "H0 x [m]"
)
ax.set_ylabel(
    "H0 y [m]"
)
ax.set_zlabel(
    "H0 z [m]"
)

fig.tight_layout()

fig.savefig(
    ELEVATION_PNG,
    dpi=160,
)

plt.close(fig)


# ============================================================
# Gates
# ============================================================

for path in (
    HEADLAMP_PNG,
    ELEVATION_PNG,
):
    if not path.is_file():
        raise RuntimeError(
            f"Visualization not created: {path}"
        )

    if path.stat().st_size < 10_000:
        raise RuntimeError(
            "Visualization unexpectedly small: "
            f"{path} -> {path.stat().st_size} bytes"
        )


report = {
    "status": "PASS",
    "block": "stage1a_block8a_visual_gate",
    "scenario_id": SID,
    "artifact_semantics":
        "causal_womd_annotation_upstream",
    "sensor_realistic": False,
    "causal_only": True,
    "headlamp_frame": {
        "origin": "canonical_H0_headlamp_surrogate",
        "x_axis": "forward",
        "y_axis": "left",
        "z_axis": "up",
    },
    "scene": {
        "actors": len(actors.actors),
        "anchor_valid_actors": anchor_actor_count,
        "receiver_candidates": receiver_count,
        "static_map_features":
            len(maps.static_features),
        "causal_dynamic_frames":
            len(maps.dynamic_frames),
    },
    "visualizations": {
        "headlamp_centric": {
            "path": str(HEADLAMP_PNG),
            "size_bytes":
                HEADLAMP_PNG.stat().st_size,
            "sha256": sha256(
                HEADLAMP_PNG
            ),
        },
        "elevation_3d": {
            "path": str(ELEVATION_PNG),
            "size_bytes":
                ELEVATION_PNG.stat().st_size,
            "sha256": sha256(
                ELEVATION_PNG
            ),
        },
    },
}

REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("===== Stage1A visual gate =====")
print(
    "headlamp_centric =",
    HEADLAMP_PNG,
    HEADLAMP_PNG.stat().st_size,
    "bytes",
)
print(
    "elevation_3d     =",
    ELEVATION_PNG,
    ELEVATION_PNG.stat().st_size,
    "bytes",
)
print(
    "anchor_valid     =",
    anchor_actor_count,
)
print(
    "receiver_candidates =",
    receiver_count,
)
print("causal_only = True")
print("STATUS = PASS")
print("report =", REPORT)
