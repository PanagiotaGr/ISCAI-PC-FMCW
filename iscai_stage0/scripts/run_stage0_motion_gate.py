from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from iscai_stage0.stage0_geometry import (
    box_corners_xy,
    ego_to_global_xy,
    global_to_ego_xy,
)
from iscai_stage0.womd_proto_io import read_first_scenario


TFRECORD = Path(
    "/waymo/data/v1_3_0_scenario/validation/"
    "validation.tfrecord-00000-of-00150"
)

REPORT_DIR = Path(
    "/waymo/iscai_stage0/reports/stage0"
)

REPORT_PATH = REPORT_DIR / "motion_coordinate_audit.json"

GLOBAL_FIGURE = REPORT_DIR / "common_scene_global.png"
EGO_FIGURE = REPORT_DIR / "common_scene_ego.png"


def object_type_name(track) -> str:
    field = track.DESCRIPTOR.fields_by_name["object_type"]
    value = field.enum_type.values_by_number.get(track.object_type)

    if value is None:
        return f"UNKNOWN_{track.object_type}"

    return value.name


def map_feature_kind(feature) -> str | None:
    """Return active MapFeature oneof type without assuming its oneof name."""
    for oneof in feature.DESCRIPTOR.oneofs:
        value = feature.WhichOneof(oneof.name)
        if value is not None:
            return value

    return None


def message_xy_points(message, field_name: str) -> np.ndarray:
    if not hasattr(message, field_name):
        return np.empty((0, 2), dtype=np.float64)

    values = getattr(message, field_name)

    points = [
        [float(point.x), float(point.y)]
        for point in values
    ]

    if not points:
        return np.empty((0, 2), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def get_sdc_anchor(scenario):
    anchor = scenario.current_time_index

    if not 0 <= scenario.sdc_track_index < len(scenario.tracks):
        raise RuntimeError(
            f"Invalid sdc_track_index={scenario.sdc_track_index}"
        )

    sdc_track = scenario.tracks[scenario.sdc_track_index]

    if anchor >= len(sdc_track.states):
        raise RuntimeError("SDC track does not contain anchor state.")

    state = sdc_track.states[anchor]

    if not state.valid:
        raise RuntimeError("SDC anchor state is invalid.")

    return sdc_track, state


def tracks_to_predict_indices(scenario) -> set[int]:
    indices: set[int] = set()

    for item in scenario.tracks_to_predict:
        if hasattr(item, "track_index"):
            indices.add(int(item.track_index))

    return indices


def draw_map(
    ax,
    scenario,
    transform,
) -> None:
    """Draw the map using only fields already confirmed in Stage 0."""
    for feature in scenario.map_features:
        kind = map_feature_kind(feature)

        if kind is None:
            continue

        nested = getattr(feature, kind)

        if kind in {"lane", "road_line", "road_edge"}:
            points = message_xy_points(nested, "polyline")

            if len(points) >= 2:
                points = transform(points)
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    linewidth=0.6,
                    alpha=0.55,
                )

        elif kind in {
            "crosswalk",
            "speed_bump",
            "driveway",
        }:
            points = message_xy_points(nested, "polygon")

            if len(points) >= 3:
                points = transform(points)

                closed = np.vstack([points, points[0]])

                ax.plot(
                    closed[:, 0],
                    closed[:, 1],
                    linewidth=0.6,
                    alpha=0.55,
                )

        elif kind == "stop_sign":
            if hasattr(nested, "position"):
                position = nested.position

                point = transform(
                    np.array(
                        [[position.x, position.y]],
                        dtype=np.float64,
                    )
                )[0]

                ax.scatter(
                    point[0],
                    point[1],
                    marker="x",
                    s=15,
                )


def draw_tracks_and_boxes(
    ax,
    scenario,
    transform,
    heading_offset: float,
) -> None:
    anchor = scenario.current_time_index
    prediction_targets = tracks_to_predict_indices(scenario)

    class_markers = {
        "TYPE_VEHICLE": "o",
        "TYPE_PEDESTRIAN": "^",
        "TYPE_CYCLIST": "s",
    }

    for track_index, track in enumerate(scenario.tracks):
        history = [
            state
            for state in track.states[: anchor + 1]
            if state.valid
        ]

        if not history:
            continue

        history_xy = np.array(
            [
                [state.center_x, state.center_y]
                for state in history
            ],
            dtype=np.float64,
        )

        history_xy = transform(history_xy)

        ax.plot(
            history_xy[:, 0],
            history_xy[:, 1],
            linewidth=1.0,
            alpha=0.8,
        )

        current = track.states[anchor]

        if not current.valid:
            continue

        corners = box_corners_xy(
            center_x=current.center_x,
            center_y=current.center_y,
            length=current.length,
            width=current.width,
            heading=current.heading,
        )

        corners = transform(corners)

        polygon = Polygon(
            corners,
            closed=True,
            fill=False,
            linewidth=(
                1.8
                if track_index == scenario.sdc_track_index
                else 0.8
            ),
        )

        ax.add_patch(polygon)

        center = transform(
            np.array(
                [[current.center_x, current.center_y]],
                dtype=np.float64,
            )
        )[0]

        class_name = object_type_name(track)
        marker = class_markers.get(class_name, ".")

        ax.scatter(
            center[0],
            center[1],
            marker=marker,
            s=(
                35
                if track_index == scenario.sdc_track_index
                else 15
            ),
        )

        if track_index in prediction_targets:
            ax.text(
                center[0],
                center[1],
                f"TTP:{track.id}",
                fontsize=6,
            )

        # Sanity only: relative heading in the current frame.
        _relative_heading = (
            current.heading - heading_offset
        )

        # Keep normalized value available during debugging.
        _relative_heading = math.atan2(
            math.sin(_relative_heading),
            math.cos(_relative_heading),
        )


def save_common_visualizations(
    scenario,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Global frame ----
    fig, ax = plt.subplots(figsize=(10, 10))

    identity = lambda points: np.asarray(
        points,
        dtype=np.float64,
    )

    draw_map(ax, scenario, identity)

    draw_tracks_and_boxes(
        ax,
        scenario,
        identity,
        heading_offset=0.0,
    )

    ax.set_title(
        f"WOMD Stage 0 — global frame\n"
        f"scenario={scenario.scenario_id}"
    )
    ax.set_xlabel("global x")
    ax.set_ylabel("global y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(GLOBAL_FIGURE, dpi=180)
    plt.close(fig)

    # ---- Ego-at-anchor frame ----
    def to_ego(points: np.ndarray) -> np.ndarray:
        return global_to_ego_xy(
            points,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_heading=ego_heading,
        )

    fig, ax = plt.subplots(figsize=(10, 10))

    draw_map(ax, scenario, to_ego)

    draw_tracks_and_boxes(
        ax,
        scenario,
        to_ego,
        heading_offset=ego_heading,
    )

    ax.scatter(
        0.0,
        0.0,
        marker="*",
        s=100,
        label="SDC anchor",
    )

    ax.set_title(
        f"WOMD Stage 0 — ego-at-anchor sanity frame\n"
        f"scenario={scenario.scenario_id}"
    )
    ax.set_xlabel("forward x [m]")
    ax.set_ylabel("left y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(EGO_FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    scenario = read_first_scenario(TFRECORD)

    timestamps = list(scenario.timestamps_seconds)

    if len(timestamps) != 91:
        raise RuntimeError(
            f"Expected 91 timestamps, got {len(timestamps)}"
        )

    anchor = scenario.current_time_index

    if anchor != 10:
        raise RuntimeError(
            f"Expected current_time_index=10, got {anchor}"
        )

    if any(
        len(track.states) != len(timestamps)
        for track in scenario.tracks
    ):
        raise RuntimeError(
            "At least one track is not aligned with scenario timestamps."
        )

    deltas = [
        b - a
        for a, b in zip(timestamps[:-1], timestamps[1:])
    ]

    if any(dt <= 0 for dt in deltas):
        raise RuntimeError("Non-monotonic timestamps detected.")

    _, sdc = get_sdc_anchor(scenario)

    ego_x = float(sdc.center_x)
    ego_y = float(sdc.center_y)
    ego_heading = float(sdc.heading)

    # SDC anchor must become (0, 0).
    sdc_ego = global_to_ego_xy(
        np.array([[ego_x, ego_y]]),
        ego_x,
        ego_y,
        ego_heading,
    )[0]

    sdc_origin_error = float(
        np.linalg.norm(sdc_ego)
    )

    # Check numerical forward/inverse consistency on valid current actors.
    current_global = np.array(
        [
            [track.states[anchor].center_x,
             track.states[anchor].center_y]
            for track in scenario.tracks
            if track.states[anchor].valid
        ],
        dtype=np.float64,
    )

    if len(current_global) == 0:
        raise RuntimeError(
            "No valid current actors available for coordinate audit."
        )

    current_ego = global_to_ego_xy(
        current_global,
        ego_x,
        ego_y,
        ego_heading,
    )

    reconstructed_global = ego_to_global_xy(
        current_ego,
        ego_x,
        ego_y,
        ego_heading,
    )

    reconstruction_errors = np.linalg.norm(
        reconstructed_global - current_global,
        axis=1,
    )

    max_roundtrip_error = float(
        reconstruction_errors.max()
    )

    current_class_counts: dict[str, int] = {}

    for track in scenario.tracks:
        state = track.states[anchor]

        if not state.valid:
            continue

        name = object_type_name(track)

        current_class_counts[name] = (
            current_class_counts.get(name, 0) + 1
        )

    save_common_visualizations(
        scenario,
        ego_x,
        ego_y,
        ego_heading,
    )

    result: dict[str, Any] = {
        "status": "pass",
        "source_file": str(TFRECORD),
        "scenario_id": scenario.scenario_id,

        "window": {
            "timestamps": len(timestamps),
            "current_time_index": anchor,
            "history_steps": anchor,
            "current_steps": 1,
            "future_steps": len(timestamps) - anchor - 1,
        },

        "timestamp_delta_seconds": {
            "min": min(deltas),
            "max": max(deltas),
            "mean": mean(deltas),
            "monotonic": True,
        },

        "sdc_anchor": {
            "track_index": scenario.sdc_track_index,
            "global_x": ego_x,
            "global_y": ego_y,
            "heading": ego_heading,
            "valid": bool(sdc.valid),
        },

        "coordinate_sanity": {
            "sdc_origin_error_m": sdc_origin_error,
            "max_global_ego_global_error_m":
                max_roundtrip_error,
        },

        "current_valid_class_counts":
            current_class_counts,

        "visualizations": {
            "global": str(GLOBAL_FIGURE),
            "ego_at_anchor": str(EGO_FIGURE),
        },

        "note": (
            "Ego-at-anchor conversion is a Stage-0 sanity check. "
            "Production ego/headlamp transforms are Stage 1."
        ),
    }

    if sdc_origin_error > 1e-9:
        raise RuntimeError(
            f"SDC origin sanity failed: {sdc_origin_error}"
        )

    if max_roundtrip_error > 1e-9:
        raise RuntimeError(
            "Coordinate round-trip sanity failed: "
            f"{max_roundtrip_error}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Stage 0 motion coordinate gate: PASS")
    print("scenario_id:", scenario.scenario_id)
    print("SDC origin error:", sdc_origin_error)
    print(
        "max coordinate round-trip error:",
        max_roundtrip_error,
    )
    print("global figure:", GLOBAL_FIGURE)
    print("ego figure:", EGO_FIGURE)
    print("report:", REPORT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(
            f"Stage 0 motion coordinate gate failed: {exc}"
        ) from exc