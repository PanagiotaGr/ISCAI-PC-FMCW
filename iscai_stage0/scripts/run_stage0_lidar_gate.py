from __future__ import annotations

import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import compressed_lidar_pb2

from iscai_stage0.womd_proto_io import (
    iter_scenarios,
    read_first_scenario,
)


ROOT = Path("/waymo/iscai_stage0")
REPORT_DIR = ROOT / "reports/stage0"

MOTION_SHARD = Path(
    "/waymo/data/v1_3_0_scenario/validation/"
    "validation.tfrecord-00000-of-00150"
)

LIDAR_DIR = Path(
    "/waymo/data/womd_lidar_stage0/validation"
)

MANIFEST = REPORT_DIR / "lidar_subset_manifest.json"

REPORT = REPORT_DIR / "lidar_alignment_audit.json"

FIGURE = (
    REPORT_DIR /
    "common_scene_map_boxes_tracks_lidar.png"
)

PRIMARY_ID = "b85e1bd6cc8e74c0"

TYPE_NAMES = {
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
}

REQUIRED_CLASSES = set(TYPE_NAMES.values())

DISTANCE_EDGES = [
    0.0,
    20.0,
    40.0,
    60.0,
    80.0,
    float("inf"),
]


# ------------------------------------------------------------
# Basic transforms
# ------------------------------------------------------------

def as_transform(proto_transform) -> np.ndarray:
    values = np.asarray(
        proto_transform.transform,
        dtype=np.float64,
    )

    if values.size != 16:
        raise RuntimeError(
            f"Expected 16 transform values, got {values.size}"
        )

    transform = values.reshape(4, 4)

    if not np.isfinite(transform).all():
        raise RuntimeError("Non-finite transform.")

    return transform


def transform_points(
    transform: np.ndarray,
    points_xyz: np.ndarray,
) -> np.ndarray:
    points_xyz = np.asarray(
        points_xyz,
        dtype=np.float64,
    )

    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.float64)

    return (
        points_xyz @ transform[:3, :3].T
        + transform[:3, 3]
    )


def wrap_angle(angle: float) -> float:
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def yaw_from_transform(transform: np.ndarray) -> float:
    return math.atan2(
        transform[1, 0],
        transform[0, 0],
    )


# ------------------------------------------------------------
# Official DeltaEncodedData decoding, NumPy version
# ------------------------------------------------------------

def run_length_decode(lengths: np.ndarray) -> np.ndarray:
    total = int(np.sum(lengths))

    decoded = np.empty(total, dtype=np.int64)

    cursor = 0
    current = 1

    for length in lengths:
        length = int(length)

        decoded[cursor:cursor + length] = current

        cursor += length
        current = 1 - current

    return decoded


def decompress_delta(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Empty compressed LiDAR payload.")

    proto = compressed_lidar_pb2.DeltaEncodedData()

    proto.ParseFromString(
        zlib.decompress(data)
    )

    shape = np.asarray(
        proto.metadata.shape,
        dtype=np.int64,
    )

    precision = np.asarray(
        proto.metadata.quant_precision,
        dtype=np.float64,
    )

    if shape.size != 3:
        raise RuntimeError(
            f"Unexpected range-image shape metadata: {shape}"
        )

    if precision.size != shape[2]:
        raise RuntimeError(
            "Quantization precision/channel mismatch."
        )

    mask = run_length_decode(
        np.asarray(proto.mask, dtype=np.int64)
    )

    residual = np.asarray(
        proto.residual,
        dtype=np.int64,
    )

    reconstructed = np.cumsum(residual)

    valid_mask = mask > 0

    if int(valid_mask.sum()) != len(reconstructed):
        raise RuntimeError(
            "Delta residual/mask length mismatch."
        )

    mask[valid_mask] = reconstructed

    image = np.reshape(
        mask,
        (
            int(shape[2]),
            int(shape[0]),
            int(shape[1]),
        ),
    )

    image = np.transpose(
        image,
        (1, 2, 0),
    ).astype(np.float64)

    image *= precision.reshape(1, 1, -1)

    return image


# ------------------------------------------------------------
# Waymo geometry equations, NumPy implementation
# ------------------------------------------------------------

def rpy_rotation(
    roll: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    """
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    matching Waymo transform_utils.
    """
    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    shape = roll.shape + (3, 3)

    rotation = np.empty(
        shape,
        dtype=np.float64,
    )

    rotation[..., 0, 0] = cy * cp
    rotation[..., 0, 1] = cy * sp * sr - sy * cr
    rotation[..., 0, 2] = cy * sp * cr + sy * sr

    rotation[..., 1, 0] = sy * cp
    rotation[..., 1, 1] = sy * sp * sr + cy * cr
    rotation[..., 1, 2] = sy * sp * cr - cy * sr

    rotation[..., 2, 0] = -sp
    rotation[..., 2, 1] = cp * sr
    rotation[..., 2, 2] = cp * cr

    return rotation


def beam_inclinations(
    calibration,
    height: int,
) -> np.ndarray:
    if calibration.beam_inclinations:
        values = np.asarray(
            calibration.beam_inclinations,
            dtype=np.float64,
        )
    else:
        minimum = calibration.beam_inclination_min
        maximum = calibration.beam_inclination_max

        diff = maximum - minimum

        values = (
            (
                0.5 + np.arange(
                    height,
                    dtype=np.float64,
                )
            )
            / float(height)
            * diff
            + minimum
        )

    # Official WOMD utility reverses inclinations here.
    return values[::-1]


def range_image_to_vehicle(
    range_image: np.ndarray,
    calibration,
    *,
    pixel_pose_rotation: np.ndarray | None,
    pixel_pose_translation: np.ndarray | None,
    frame_pose: np.ndarray | None,
) -> np.ndarray:

    height, width = range_image.shape[:2]

    ranges = range_image[..., 0]

    extrinsic = as_transform(
        calibration.extrinsic
    )

    inclination = beam_inclinations(
        calibration,
        height,
    )

    azimuth_correction = math.atan2(
        extrinsic[1, 0],
        extrinsic[0, 0],
    )

    ratios = (
        np.arange(
            width,
            0,
            -1,
            dtype=np.float64,
        )
        - 0.5
    ) / float(width)

    azimuth = (
        (ratios * 2.0 - 1.0) * np.pi
        - azimuth_correction
    )

    azimuth = azimuth[None, :]
    inclination = inclination[:, None]

    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)

    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)

    x = cos_az * cos_inc * ranges
    y = sin_az * cos_inc * ranges
    z = sin_inc * ranges

    sensor_points = np.stack(
        [x, y, z],
        axis=-1,
    )

    # Sensor -> vehicle
    vehicle_points = np.einsum(
        "ij,hwj->hwi",
        extrinsic[:3, :3],
        sensor_points,
    )

    vehicle_points += extrinsic[:3, 3]

    # TOP LiDAR motion compensation:
    # vehicle(pixel time) -> global -> vehicle(frame time)
    if pixel_pose_rotation is not None:
        if (
            pixel_pose_translation is None
            or frame_pose is None
        ):
            raise RuntimeError(
                "Incomplete TOP LiDAR pose information."
            )

        global_points = np.einsum(
            "hwij,hwj->hwi",
            pixel_pose_rotation,
            vehicle_points,
        )

        global_points += pixel_pose_translation

        world_to_vehicle = np.linalg.inv(
            frame_pose
        )

        vehicle_points = np.einsum(
            "ij,hwj->hwi",
            world_to_vehicle[:3, :3],
            global_points,
        )

        vehicle_points += (
            world_to_vehicle[:3, 3]
        )

    mask = ranges > 0.0

    return vehicle_points[mask]


def extract_frame_points(frame):
    if len(frame.lasers) == 0:
        raise RuntimeError(
            "LiDAR frame contains zero lasers."
        )

    if (
        len(frame.lasers)
        != len(frame.laser_calibrations)
    ):
        raise RuntimeError(
            "lasers/calibrations length mismatch."
        )

    frame_pose = as_transform(frame.pose)

    calibrations = {
        int(calibration.name): calibration
        for calibration
        in frame.laser_calibrations
    }

    all_points = []

    laser_stats = []

    for laser in frame.lasers:
        name = int(laser.name)

        if name not in calibrations:
            raise RuntimeError(
                f"No calibration for laser {name}"
            )

        calibration = calibrations[name]

        is_top = (
            name == dataset_pb2.LaserName.TOP
        )

        pixel_rotation = None
        pixel_translation = None

        if is_top:
            compressed_pose = (
                laser
                .ri_return1
                .range_image_pose_delta_compressed
            )

            if not compressed_pose:
                raise RuntimeError(
                    "TOP LiDAR return1 has no pixel pose."
                )

            pose = decompress_delta(
                compressed_pose
            )

            if pose.shape[-1] != 6:
                raise RuntimeError(
                    "Expected TOP pixel pose with 6 channels, "
                    f"got {pose.shape}"
                )

            pixel_rotation = rpy_rotation(
                pose[..., 0],
                pose[..., 1],
                pose[..., 2],
            )

            pixel_translation = pose[..., 3:6]

        return_counts = []

        for range_return in (
            laser.ri_return1,
            laser.ri_return2,
        ):
            compressed = (
                range_return
                .range_image_delta_compressed
            )

            if not compressed:
                return_counts.append(0)
                continue

            image = decompress_delta(
                compressed
            )

            if (
                image.ndim != 3
                or image.shape[-1] < 3
            ):
                raise RuntimeError(
                    f"Bad range image shape: {image.shape}"
                )

            points = range_image_to_vehicle(
                image,
                calibration,
                pixel_pose_rotation=(
                    pixel_rotation
                    if is_top
                    else None
                ),
                pixel_pose_translation=(
                    pixel_translation
                    if is_top
                    else None
                ),
                frame_pose=(
                    frame_pose
                    if is_top
                    else None
                ),
            )

            return_counts.append(len(points))

            if len(points):
                all_points.append(points)

        laser_stats.append(
            {
                "laser_name": name,
                "return1_points":
                    return_counts[0],
                "return2_points":
                    return_counts[1],
            }
        )

    if not all_points:
        raise RuntimeError(
            "Frame decoded successfully but yielded zero points."
        )

    return (
        np.concatenate(all_points, axis=0),
        frame_pose,
        laser_stats,
    )


# ------------------------------------------------------------
# Motion lookup
# ------------------------------------------------------------

def load_motion_scenarios(
    scenario_ids: set[str],
) -> dict[str, object]:

    found = {}

    for scenario in iter_scenarios(
        MOTION_SHARD,
        limit=None,
    ):
        sid = scenario.scenario_id

        if sid in scenario_ids:
            found[sid] = scenario

        if len(found) == len(scenario_ids):
            break

    missing = scenario_ids - set(found)

    if missing:
        raise RuntimeError(
            f"Motion scenarios not found: {sorted(missing)}"
        )

    return found


# ------------------------------------------------------------
# Density inside WOMD boxes
# ------------------------------------------------------------

def points_in_box(
    points_global: np.ndarray,
    state,
) -> int:

    dx = points_global[:, 0] - state.center_x
    dy = points_global[:, 1] - state.center_y
    dz = points_global[:, 2] - state.center_z

    c = math.cos(state.heading)
    s = math.sin(state.heading)

    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy

    inside = (
        (np.abs(local_x) <= state.length / 2.0)
        & (np.abs(local_y) <= state.width / 2.0)
        & (np.abs(dz) <= state.height / 2.0)
    )

    return int(np.count_nonzero(inside))


def distance_bin(distance: float) -> str:
    for low, high in zip(
        DISTANCE_EDGES[:-1],
        DISTANCE_EDGES[1:],
    ):
        if low <= distance < high:
            if math.isinf(high):
                return f"{int(low)}+m"

            return (
                f"{int(low)}-{int(high)}m"
            )

    raise RuntimeError("Invalid distance.")


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

def feature_kind(feature):
    for oneof in feature.DESCRIPTOR.oneofs:
        value = feature.WhichOneof(oneof.name)

        if value is not None:
            return value

    return None


def proto_points_xyz(
    message,
    field: str,
) -> np.ndarray:

    values = getattr(message, field)

    if not values:
        return np.empty((0, 3))

    return np.asarray(
        [
            [
                float(point.x),
                float(point.y),
                float(point.z),
            ]
            for point in values
        ],
        dtype=np.float64,
    )


def box_corners_global(state) -> np.ndarray:
    l = state.length / 2.0
    w = state.width / 2.0

    local = np.asarray(
        [
            [l, w],
            [l, -w],
            [-l, -w],
            [-l, w],
        ],
        dtype=np.float64,
    )

    c = math.cos(state.heading)
    s = math.sin(state.heading)

    rotation = np.asarray(
        [
            [c, -s],
            [s, c],
        ]
    )

    xy = local @ rotation.T

    xy[:, 0] += state.center_x
    xy[:, 1] += state.center_y

    z = np.full(
        (4, 1),
        state.center_z,
    )

    return np.concatenate(
        [xy, z],
        axis=1,
    )


def save_visualization(
    scenario,
    anchor_points_vehicle: np.ndarray,
    frame_pose: np.ndarray,
) -> None:

    world_to_vehicle = np.linalg.inv(
        frame_pose
    )

    anchor = scenario.current_time_index

    fig, ax = plt.subplots(
        figsize=(11, 11)
    )

    # LiDAR
    points = anchor_points_vehicle

    in_view = (
        (np.abs(points[:, 0]) <= 120.0)
        & (np.abs(points[:, 1]) <= 120.0)
    )

    points = points[in_view]

    # visualization only: bounded rendering size
    if len(points) > 120_000:
        step = math.ceil(
            len(points) / 120_000
        )
        points = points[::step]

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=0.4,
        alpha=0.25,
        label="LiDAR",
    )

    # Map
    for feature in scenario.map_features:
        kind = feature_kind(feature)

        if kind is None:
            continue

        nested = getattr(feature, kind)

        if kind in {
            "lane",
            "road_line",
            "road_edge",
        }:
            global_xyz = proto_points_xyz(
                nested,
                "polyline",
            )

            if len(global_xyz) >= 2:
                local = transform_points(
                    world_to_vehicle,
                    global_xyz,
                )

                ax.plot(
                    local[:, 0],
                    local[:, 1],
                    linewidth=0.5,
                    alpha=0.55,
                )

        elif kind in {
            "crosswalk",
            "speed_bump",
            "driveway",
        }:
            global_xyz = proto_points_xyz(
                nested,
                "polygon",
            )

            if len(global_xyz) >= 3:
                local = transform_points(
                    world_to_vehicle,
                    global_xyz,
                )

                local = np.vstack(
                    [local, local[0]]
                )

                ax.plot(
                    local[:, 0],
                    local[:, 1],
                    linewidth=0.6,
                    alpha=0.6,
                )

        elif kind == "stop_sign":
            position = nested.position

            point = transform_points(
                world_to_vehicle,
                np.asarray(
                    [[
                        position.x,
                        position.y,
                        position.z,
                    ]]
                ),
            )[0]

            ax.scatter(
                point[0],
                point[1],
                marker="x",
                s=16,
            )

    # Causal tracks and anchor boxes
    for track_index, track in enumerate(
        scenario.tracks
    ):
        history = [
            state
            for state
            in track.states[:anchor + 1]
            if state.valid
        ]

        if history:
            global_xyz = np.asarray(
                [
                    [
                        state.center_x,
                        state.center_y,
                        state.center_z,
                    ]
                    for state in history
                ],
                dtype=np.float64,
            )

            local = transform_points(
                world_to_vehicle,
                global_xyz,
            )

            ax.plot(
                local[:, 0],
                local[:, 1],
                linewidth=1.0,
                alpha=0.8,
            )

        state = track.states[anchor]

        if not state.valid:
            continue

        corners = box_corners_global(
            state
        )

        local = transform_points(
            world_to_vehicle,
            corners,
        )

        polygon = Polygon(
            local[:, :2],
            closed=True,
            fill=False,
            linewidth=(
                2.0
                if track_index
                == scenario.sdc_track_index
                else 0.8
            ),
        )

        ax.add_patch(polygon)

    ax.scatter(
        0.0,
        0.0,
        marker="*",
        s=100,
        label="LiDAR vehicle-frame origin",
    )

    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("vehicle forward x [m]")
    ax.set_ylabel("vehicle left y [m]")

    ax.set_title(
        "WOMD Stage 0 — map + tracks + boxes + LiDAR\n"
        f"scenario={scenario.scenario_id}, "
        f"frame={anchor}"
    )

    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE,
        dpi=180,
    )

    plt.close(fig)


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

def summarize_density(samples):
    grouped = defaultdict(list)

    for sample in samples:
        key = (
            sample["class"],
            sample["distance_bin"],
        )

        grouped[key].append(
            sample["points_in_box"]
        )

    output = {}

    for (
        class_name,
        bin_name,
    ), values in sorted(grouped.items()):

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        output.setdefault(
            class_name,
            {},
        )[bin_name] = {
            "boxes": int(len(array)),
            "mean_points":
                float(np.mean(array)),
            "median_points":
                float(np.median(array)),
            "min_points":
                int(np.min(array)),
            "max_points":
                int(np.max(array)),
        }

    return output


# ------------------------------------------------------------
# Main gate
# ------------------------------------------------------------

def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = json.loads(
        MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    scenario_ids = {
        item["scenario_id"]
        for item in manifest["objects"]
    }

    motion = load_motion_scenarios(
        scenario_ids
    )

    per_scenario = []

    density_samples = []

    all_classes_seen = set()

    primary_visualization = None

    for sid in sorted(scenario_ids):
        scenario = motion[sid]

        lidar_path = (
            LIDAR_DIR /
            f"{sid}.tfrecord"
        )

        if not lidar_path.is_file():
            raise RuntimeError(
                f"Missing LiDAR sidecar: {lidar_path}"
            )

        lidar = read_first_scenario(
            lidar_path
        )

        # Sidecars may contain only compressed LiDAR.
        if (
            lidar.scenario_id
            and lidar.scenario_id != sid
        ):
            raise RuntimeError(
                f"scenario_id mismatch: "
                f"{sid} vs {lidar.scenario_id}"
            )

        expected_frames = (
            scenario.current_time_index + 1
        )

        actual_frames = len(
            lidar.compressed_frame_laser_data
        )

        if actual_frames != expected_frames:
            raise RuntimeError(
                f"{sid}: expected {expected_frames} "
                f"LiDAR frames, got {actual_frames}"
            )

        # If sidecar also includes timestamps, compare directly.
        sidecar_timestamp_check = None

        if lidar.timestamps_seconds:
            lidar_timestamps = np.asarray(
                lidar.timestamps_seconds,
                dtype=np.float64,
            )

            motion_timestamps = np.asarray(
                scenario.timestamps_seconds[
                    :actual_frames
                ],
                dtype=np.float64,
            )

            if len(lidar_timestamps) != actual_frames:
                raise RuntimeError(
                    f"{sid}: sidecar timestamp count mismatch."
                )

            max_dt_error = float(
                np.max(
                    np.abs(
                        lidar_timestamps
                        - motion_timestamps
                    )
                )
            )

            if max_dt_error > 1e-6:
                raise RuntimeError(
                    f"{sid}: timestamp mismatch "
                    f"{max_dt_error}s"
                )

            sidecar_timestamp_check = {
                "embedded": True,
                "max_abs_error_seconds":
                    max_dt_error,
            }

        else:
            sidecar_timestamp_check = {
                "embedded": False,
                "alignment": (
                    "index correspondence defined by "
                    "Scenario.compressed_frame_laser_data"
                ),
            }

        total_points = 0

        frame_point_counts = []

        sdc_offsets = []

        heading_offsets = []

        first_frame_lasers = None

        for frame_index, frame in enumerate(
            lidar.compressed_frame_laser_data
        ):
            (
                points_vehicle,
                frame_pose,
                laser_stats,
            ) = extract_frame_points(frame)

            if first_frame_lasers is None:
                first_frame_lasers = laser_stats

            total_points += len(
                points_vehicle
            )

            frame_point_counts.append(
                int(len(points_vehicle))
            )

            # Validate homogeneous pose.
            if not np.allclose(
                frame_pose[3],
                [0, 0, 0, 1],
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"{sid}: invalid pose last row."
                )

            rotation = frame_pose[:3, :3]

            if not np.allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=2e-3,
            ):
                raise RuntimeError(
                    f"{sid}: non-orthonormal frame pose."
                )

            # Motion ↔ LiDAR coordinate-frame sanity via SDC.
            sdc_state = scenario.tracks[
                scenario.sdc_track_index
            ].states[frame_index]

            if not sdc_state.valid:
                raise RuntimeError(
                    f"{sid}: SDC invalid at LiDAR frame "
                    f"{frame_index}"
                )

            world_to_vehicle = np.linalg.inv(
                frame_pose
            )

            sdc_global = np.asarray(
                [[
                    sdc_state.center_x,
                    sdc_state.center_y,
                    sdc_state.center_z,
                ]],
                dtype=np.float64,
            )

            sdc_vehicle = transform_points(
                world_to_vehicle,
                sdc_global,
            )[0]

            sdc_offsets.append(
                sdc_vehicle.tolist()
            )

            heading_offsets.append(
                wrap_angle(
                    sdc_state.heading
                    - yaw_from_transform(
                        frame_pose
                    )
                )
            )

            # Convert the FULL LiDAR cloud to WOMD global frame.
            points_global = transform_points(
                frame_pose,
                points_vehicle,
            )

            # Point density per class/distance.
            for track_index, track in enumerate(
                scenario.tracks
            ):
                if (
                    track_index
                    == scenario.sdc_track_index
                ):
                    continue

                class_name = TYPE_NAMES.get(
                    track.object_type
                )

                if class_name is None:
                    continue

                state = track.states[
                    frame_index
                ]

                if not state.valid:
                    continue

                center_vehicle = transform_points(
                    world_to_vehicle,
                    np.asarray(
                        [[
                            state.center_x,
                            state.center_y,
                            state.center_z,
                        ]]
                    ),
                )[0]

                distance = float(
                    np.hypot(
                        center_vehicle[0],
                        center_vehicle[1],
                    )
                )

                count = points_in_box(
                    points_global,
                    state,
                )

                all_classes_seen.add(
                    class_name
                )

                density_samples.append(
                    {
                        "scenario_id": sid,
                        "frame_index":
                            frame_index,
                        "track_id":
                            int(track.id),
                        "class":
                            class_name,
                        "distance_m":
                            distance,
                        "distance_bin":
                            distance_bin(distance),
                        "points_in_box":
                            count,
                    }
                )

            if (
                sid == PRIMARY_ID
                and frame_index
                == scenario.current_time_index
            ):
                primary_visualization = (
                    scenario,
                    points_vehicle.copy(),
                    frame_pose.copy(),
                )

        offsets = np.asarray(
            sdc_offsets,
            dtype=np.float64,
        )

        heading_offsets_np = np.asarray(
            heading_offsets,
            dtype=np.float64,
        )

        offset_std = np.std(
            offsets,
            axis=0,
        )

        heading_std = float(
            np.std(
                heading_offsets_np
            )
        )

        # This should be approximately a rigid,
        # time-stable relation between frame pose
        # origin and SDC box reference.
        coordinate_consistency = (
            float(
                np.max(
                    offset_std[:2]
                )
            ) < 0.75
            and heading_std < 0.15
        )

        if not coordinate_consistency:
            raise RuntimeError(
                f"{sid}: motion/LiDAR coordinate "
                "relationship is not time-stable. "
                f"xy std={offset_std[:2]}, "
                f"heading std={heading_std}"
            )

        per_scenario.append(
            {
                "scenario_id": sid,

                "expected_lidar_frames":
                    expected_frames,

                "actual_lidar_frames":
                    actual_frames,

                "timestamp_alignment":
                    sidecar_timestamp_check,

                "frame_point_counts":
                    frame_point_counts,

                "total_points":
                    int(total_points),

                "laser_structure_first_frame":
                    first_frame_lasers,

                "sdc_center_in_lidar_vehicle_frame": {
                    "mean_xyz_m":
                        np.mean(
                            offsets,
                            axis=0,
                        ).tolist(),

                    "std_xyz_m":
                        offset_std.tolist(),
                },

                "sdc_heading_minus_pose_yaw": {
                    "mean_rad":
                        float(
                            np.mean(
                                heading_offsets_np
                            )
                        ),

                    "std_rad":
                        heading_std,
                },

                "coordinate_consistency":
                    True,
            }
        )

    if primary_visualization is None:
        raise RuntimeError(
            "Primary LiDAR visualization frame not found."
        )

    save_visualization(
        *primary_visualization
    )

    missing_classes = (
        REQUIRED_CLASSES
        - all_classes_seen
    )

    if missing_classes:
        raise RuntimeError(
            "Density audit does not cover classes: "
            f"{sorted(missing_classes)}"
        )

    if not density_samples:
        raise RuntimeError(
            "No point-density samples generated."
        )

    result = {
        "status": "pass",

        "dataset_release": "WOMD v1.3.0",

        "split": "validation",

        "scenario_matching": {
            "method":
                "exact scenario_id sidecar filename",
            "scenario_count":
                len(scenario_ids),
            "all_matched":
                True,
        },

        "lidar_frame_alignment": {
            "contract": (
                "compressed_frame_laser_data[i] "
                "corresponds to timestamps_seconds[i] "
                "for i <= current_time_index"
            ),
            "all_frame_counts_match":
                True,
        },

        "decompression": {
            "method":
                "zlib + DeltaEncodedData, NumPy",
            "tensorflow_required":
                False,
        },

        "coordinate_alignment": {
            "frame_pose_validated":
                True,
            "laser_extrinsics_used":
                True,
            "top_lidar_pixel_pose_used":
                True,
            "sdc_pose_relationship_time_stable":
                True,
        },

        "per_scenario":
            per_scenario,

        "point_density": {
            "definition": (
                "all available LiDAR sensors, "
                "both returns, count of decoded "
                "points inside each valid WOMD 3D box"
            ),

            "distance_definition": (
                "planar range from LiDAR vehicle-frame "
                "origin to actor box center"
            ),

            "samples":
                len(density_samples),

            "summary_by_class_and_distance":
                summarize_density(
                    density_samples
                ),

            "raw_samples":
                density_samples,
        },

        "visualization": {
            "path": str(FIGURE),
            "contains": [
                "LiDAR points",
                "map",
                "causal tracks",
                "3D-box ground-plane footprints",
            ],
        },
    }

    REPORT.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Stage 0 WOMD-LiDAR gate: PASS")
    print(
        "Matched scenarios:",
        len(scenario_ids),
    )
    print(
        "Classes:",
        sorted(all_classes_seen),
    )
    print(
        "Density samples:",
        len(density_samples),
    )
    print(
        "Visualization:",
        FIGURE,
    )
    print(
        "Report:",
        REPORT,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(
            f"Stage 0 LiDAR gate failed: {exc}"
        ) from exc