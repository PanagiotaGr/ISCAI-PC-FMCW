"""Frozen Stage-1A contracts.

These quantities are causal canonicalized WOMD annotation-derived upstream
artifacts. They are NOT the sensor-realistic inputs of the main predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias


Vec3: TypeAlias = tuple[float, float, float]
Mat3: TypeAlias = tuple[Vec3, Vec3, Vec3]

ObjectClass: TypeAlias = Literal[
    "TYPE_UNSET",
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
    "TYPE_OTHER",
]


ARTIFACT_SEMANTICS = "causal_womd_annotation_upstream"
SENSOR_REALISTIC = False

ASSOCIATION_MODE = "oracle_womd_track_id"
LIDAR_ACTOR_ASSIGNMENT_MODE = "oracle_causal_box"

HEADLAMP_SURROGATE_MODE = "front_face_midpoint_surrogate"
RECEIVER_GEOMETRY_MODE = "centroid_baseline"

ZERO_VEC3: Vec3 = (0.0, 0.0, 0.0)
ZERO_MAT3: Mat3 = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class Stage1ArtifactSemantics:
    """Metadata that must accompany Stage-1 causal artifacts."""

    artifact_semantics: str = ARTIFACT_SEMANTICS
    sensor_realistic: bool = SENSOR_REALISTIC
    association_mode: str = ASSOCIATION_MODE
    lidar_actor_assignment_mode: str = LIDAR_ACTOR_ASSIGNMENT_MODE


@dataclass(frozen=True)
class HeadlampSurrogateConfig:
    """Configurable H-frame extrinsic relative to the SDC body frame.

    Baseline:
        translation = [L_SDC / 2, 0, 0]
        rotation = identity

    Positive longitudinal_inset_m moves the surrogate rearward from
    the nominal front-face midpoint.
    """

    mode: str = HEADLAMP_SURROGATE_MODE
    longitudinal_inset_m: float = 0.0
    lateral_offset_m: float = 0.0
    vertical_offset_m: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0

    def translation_in_sdc_m(self, sdc_length_m: float) -> Vec3:
        values = (
            sdc_length_m,
            self.longitudinal_inset_m,
            self.lateral_offset_m,
            self.vertical_offset_m,
            self.roll_rad,
            self.pitch_rad,
            self.yaw_rad,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Headlamp configuration contains non-finite values.")
        if sdc_length_m <= 0.0:
            raise ValueError("SDC length must be positive.")

        return (
            0.5 * sdc_length_m - self.longitudinal_inset_m,
            self.lateral_offset_m,
            self.vertical_offset_m,
        )


@dataclass(frozen=True)
class ReceiverGeometryConfig:
    """Receiver body-frame offset/extrinsic model.

    This covariance is receiver-OFFSET covariance, not receiver-position,
    measurement, or predictive covariance.
    """

    mode: str = RECEIVER_GEOMETRY_MODE
    offset_mean_body_m: Vec3 = ZERO_VEC3
    offset_covariance_body_m2: Mat3 = ZERO_MAT3


@dataclass(frozen=True)
class ReceiverGeometryH0:
    """Persisted Stage-1 receiver geometry fields."""

    receiver_offset_mean_H0: Vec3
    receiver_offset_covariance_H0: Mat3
    receiver_point_mean_H0: Vec3
    receiver_geometry_mode: str
    receiver_geometry_valid: bool