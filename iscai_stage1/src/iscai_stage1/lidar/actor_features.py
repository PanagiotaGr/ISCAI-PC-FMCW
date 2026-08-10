from __future__ import annotations

import math
from typing import Any

import numpy as np
from waymo_open_dataset import dataset_pb2

from iscai_stage1.lidar.contracts import (
    ActorLidarFrameFeatures,
    ActorLidarPointSummary,
    CausalActorLidarArtifact,
    DecodedActorLidarPoint,
    ScalarStats,
)
from iscai_stage1.lidar.stage0_decoder_bridge import (
    load_frozen_stage0_lidar_decoder,
)


def _stats(values: np.ndarray) -> ScalarStats:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "Scalar statistics require a non-empty 1-D array."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Scalar statistics require finite values."
        )

    return ScalarStats(
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        median=float(np.median(values)),
        variance=float(np.var(values)),
    )


def summarize_actor_lidar_points(
    points: tuple[DecodedActorLidarPoint, ...],
) -> ActorLidarPointSummary:
    """Summarize already-associated actor LiDAR points in H0.

    This preserves the earlier unit-level statistics contract.
    It is separate from the causal extractor's actor-local
    spatial-spread field.
    """
    if not points:
        return ActorLidarPointSummary(
            point_count=0,
            has_points=False,
            range_stats=None,
            intensity_stats=None,
            elongation_stats=None,
            spatial_std_H0_m=None,
        )

    xyz = np.asarray(
        [point.point_H0_m for point in points],
        dtype=np.float64,
    )

    range_m = np.asarray(
        [point.range_m for point in points],
        dtype=np.float64,
    )

    intensity = np.asarray(
        [point.intensity for point in points],
        dtype=np.float64,
    )

    elongation = np.asarray(
        [point.elongation for point in points],
        dtype=np.float64,
    )

    if xyz.shape != (len(points), 3):
        raise ValueError(
            "Each LiDAR point must have exactly 3 H0 coordinates."
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "LiDAR H0 coordinates must be finite."
        )

    if not np.all(np.isfinite(range_m)):
        raise ValueError(
            "LiDAR ranges must be finite."
        )

    if not np.all(np.isfinite(intensity)):
        raise ValueError(
            "LiDAR intensities must be finite."
        )

    if not np.all(np.isfinite(elongation)):
        raise ValueError(
            "LiDAR elongations must be finite."
        )

    if np.any(range_m < 0.0):
        raise ValueError(
            "LiDAR range cannot be negative."
        )

    spatial_std = tuple(
        float(value)
        for value in np.std(
            xyz,
            axis=0,
        )
    )

    return ActorLidarPointSummary(
        point_count=len(points),
        has_points=True,
        range_stats=_stats(range_m),
        intensity_stats=_stats(intensity),
        elongation_stats=_stats(elongation),
        spatial_std_H0_m=spatial_std,
    )


def _actor_box_mask(
    points_global: np.ndarray,
    state: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact Stage-0 same-time oriented-box convention.

    Returns:
      mask: assigned points
      local_xyz: actor-oriented coordinates for all points
    """
    dx = points_global[:, 0] - float(state.center_x)
    dy = points_global[:, 1] - float(state.center_y)
    dz = points_global[:, 2] - float(state.center_z)

    c = math.cos(float(state.heading))
    s = math.sin(float(state.heading))

    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy

    local_xyz = np.stack(
        (local_x, local_y, dz),
        axis=1,
    )

    mask = (
        (np.abs(local_x) <= float(state.length) / 2.0)
        & (np.abs(local_y) <= float(state.width) / 2.0)
        & (np.abs(dz) <= float(state.height) / 2.0)
    )

    return mask, local_xyz


def decode_frame_points_with_channels(
    frame: Any,
    frozen_decoder: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Decode all lasers and both returns.

    Returns:
      points_vehicle: Nx3
      points_global: Nx3
      range_m: N
      intensity: N
      elongation: N

    Geometry is delegated to the frozen Stage-0 decoder.
    """
    if len(frame.lasers) == 0:
        raise RuntimeError(
            "LiDAR frame contains zero lasers."
        )

    if len(frame.lasers) != len(
        frame.laser_calibrations
    ):
        raise RuntimeError(
            "lasers/calibrations length mismatch."
        )

    frame_pose = frozen_decoder.as_transform(
        frame.pose
    )

    calibrations = {
        int(calibration.name): calibration
        for calibration in frame.laser_calibrations
    }

    point_chunks = []
    range_chunks = []
    intensity_chunks = []
    elongation_chunks = []

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

            pose = frozen_decoder.decompress_delta(
                compressed_pose
            )

            if pose.shape[-1] != 6:
                raise RuntimeError(
                    "Expected TOP pixel pose with six "
                    f"channels, got {pose.shape}"
                )

            pixel_rotation = (
                frozen_decoder.rpy_rotation(
                    pose[..., 0],
                    pose[..., 1],
                    pose[..., 2],
                )
            )

            pixel_translation = pose[..., 3:6]

        for range_return in (
            laser.ri_return1,
            laser.ri_return2,
        ):
            compressed = (
                range_return
                .range_image_delta_compressed
            )

            if not compressed:
                continue

            image = frozen_decoder.decompress_delta(
                compressed
            )

            if image.ndim != 3:
                raise RuntimeError(
                    f"Bad range-image shape {image.shape}"
                )

            if image.shape[-1] < 3:
                raise RuntimeError(
                    "Range image lacks mandatory "
                    "range/intensity/elongation channels: "
                    f"{image.shape}"
                )

            valid = image[..., 0] > 0.0

            points_vehicle = (
                frozen_decoder.range_image_to_vehicle(
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
            )

            raw_range = image[..., 0][valid]
            raw_intensity = image[..., 1][valid]
            raw_elongation = image[..., 2][valid]

            expected = int(np.count_nonzero(valid))

            if len(points_vehicle) != expected:
                raise RuntimeError(
                    "Frozen geometry/channel alignment "
                    "mismatch: "
                    f"points={len(points_vehicle)}, "
                    f"valid_pixels={expected}"
                )

            point_chunks.append(
                np.asarray(
                    points_vehicle,
                    dtype=np.float64,
                )
            )
            range_chunks.append(
                np.asarray(
                    raw_range,
                    dtype=np.float64,
                )
            )
            intensity_chunks.append(
                np.asarray(
                    raw_intensity,
                    dtype=np.float64,
                )
            )
            elongation_chunks.append(
                np.asarray(
                    raw_elongation,
                    dtype=np.float64,
                )
            )

    if not point_chunks:
        raise RuntimeError(
            "Decoded LiDAR frame yielded zero points."
        )

    points_vehicle = np.concatenate(
        point_chunks,
        axis=0,
    )

    range_m = np.concatenate(
        range_chunks
    )
    intensity = np.concatenate(
        intensity_chunks
    )
    elongation = np.concatenate(
        elongation_chunks
    )

    n = len(points_vehicle)

    if not (
        len(range_m)
        == len(intensity)
        == len(elongation)
        == n
    ):
        raise RuntimeError(
            "Decoded point/channel length mismatch."
        )

    # Stage-0 frame pose maps the same decoded vehicle-frame
    # points into the WOMD/global frame used by motion boxes.
    points_global = frozen_decoder.transform_points(
        frame_pose,
        points_vehicle,
    )

    return (
        points_vehicle,
        points_global,
        range_m,
        intensity,
        elongation,
    )


def summarize_actor_frame(
    *,
    time_index: int,
    timestamp_s: float,
    state: Any,
    points_global: np.ndarray,
    range_m: np.ndarray,
    intensity: np.ndarray,
    elongation: np.ndarray,
    frozen_decoder: Any,
) -> ActorLidarFrameFeatures:
    if not bool(state.valid):
        return ActorLidarFrameFeatures(
            time_index=time_index,
            timestamp_s=timestamp_s,
            actor_state_valid=False,
            point_count=0,
            range_m=None,
            intensity=None,
            elongation=None,
            spatial_std_actor_m=None,
        )

    mask, local_xyz = _actor_box_mask(
        points_global,
        state,
    )

    count = int(np.count_nonzero(mask))

    # Regression against the exact frozen Stage-0 box counter.
    stage0_count = int(
        frozen_decoder.points_in_box(
            points_global,
            state,
        )
    )

    if count != stage0_count:
        raise RuntimeError(
            "Stage1/Stage0 oracle-box assignment mismatch: "
            f"stage1={count}, stage0={stage0_count}"
        )

    if count == 0:
        return ActorLidarFrameFeatures(
            time_index=time_index,
            timestamp_s=timestamp_s,
            actor_state_valid=True,
            point_count=0,
            range_m=None,
            intensity=None,
            elongation=None,
            spatial_std_actor_m=None,
        )

    selected_local = local_xyz[mask]

    spatial_std = tuple(
        float(value)
        for value in np.std(
            selected_local,
            axis=0,
        )
    )

    return ActorLidarFrameFeatures(
        time_index=time_index,
        timestamp_s=timestamp_s,
        actor_state_valid=True,
        point_count=count,
        range_m=_stats(range_m[mask]),
        intensity=_stats(intensity[mask]),
        elongation=_stats(elongation[mask]),
        spatial_std_actor_m=spatial_std,
    )


def extract_causal_actor_lidar(
    *,
    motion_scenario: Any,
    lidar_sidecar_scenario: Any,
) -> tuple[CausalActorLidarArtifact, ...]:
    anchor = int(
        motion_scenario.current_time_index
    )

    if len(
        lidar_sidecar_scenario
        .compressed_frame_laser_data
    ) != anchor + 1:
        raise RuntimeError(
            "Expected exactly anchor+1 causal LiDAR frames."
        )

    if len(
        motion_scenario.timestamps_seconds
    ) <= anchor:
        raise RuntimeError(
            "Incomplete causal motion timestamps."
        )

    frozen = (
        load_frozen_stage0_lidar_decoder()
    )

    per_actor: list[list[ActorLidarFrameFeatures]] = [
        []
        for _ in motion_scenario.tracks
    ]

    for time_index in range(anchor + 1):
        frame = (
            lidar_sidecar_scenario
            .compressed_frame_laser_data[
                time_index
            ]
        )

        (
            _points_vehicle,
            points_global,
            range_m,
            intensity,
            elongation,
        ) = decode_frame_points_with_channels(
            frame,
            frozen,
        )

        timestamp_s = float(
            motion_scenario.timestamps_seconds[
                time_index
            ]
        )

        for track_index, track in enumerate(
            motion_scenario.tracks
        ):
            if len(track.states) <= time_index:
                raise RuntimeError(
                    "Motion track lacks causal state: "
                    f"track={track_index}, "
                    f"time={time_index}"
                )

            feature = summarize_actor_frame(
                time_index=time_index,
                timestamp_s=timestamp_s,
                state=track.states[time_index],
                points_global=points_global,
                range_m=range_m,
                intensity=intensity,
                elongation=elongation,
                frozen_decoder=frozen,
            )

            per_actor[track_index].append(
                feature
            )

    result = []

    for track_index, track in enumerate(
        motion_scenario.tracks
    ):
        result.append(
            CausalActorLidarArtifact(
                scenario_id=str(
                    motion_scenario.scenario_id
                ),
                track_index=track_index,
                track_id=str(track.id),
                frames=tuple(
                    per_actor[track_index]
                ),
            )
        )

    return tuple(result)
