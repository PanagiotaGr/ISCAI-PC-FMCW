from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from iscai_stage3.association import (
    AssociatedDetection,
    AssociatedTrack,
)

from iscai_stage3.observations import (
    MeasurementSnapshot,
)


Vec3 = tuple[
    float,
    float,
    float,
]

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _transpose(matrix):
    return tuple(
        tuple(
            matrix[j][i]
            for j in range(len(matrix))
        )
        for i in range(len(matrix[0]))
    )


def _matmul(a, b):
    rows = len(a)
    inner = len(b)
    cols = len(b[0])

    if any(
        len(row) != inner
        for row in a
    ):
        raise ValueError(
            "Matrix dimension mismatch."
        )

    return tuple(
        tuple(
            sum(
                a[i][k] * b[k][j]
                for k in range(inner)
            )
            for j in range(cols)
        )
        for i in range(rows)
    )


def _symmetrize3(matrix) -> Matrix3:
    return tuple(
        tuple(
            0.5 * (
                matrix[i][j]
                +
                matrix[j][i]
            )
            for j in range(3)
        )
        for i in range(3)
    )  # type: ignore[return-value]


def spherical_position_Ht(
    measurement: MeasurementSnapshot,
) -> Vec3:
    """
    Stage2 measurement-space position in the
    dynamic headlamp frame H_t.

    x forward, y left, z up.
    """

    r = measurement.range_m
    az = measurement.azimuth_rad
    el = measurement.elevation_rad

    ce = cos(el)
    se = sin(el)

    ca = cos(az)
    sa = sin(az)

    return (
        r * ce * ca,
        r * ce * sa,
        r * se,
    )


def position_jacobian_Ht(
    measurement: MeasurementSnapshot,
):
    """
    Jacobian:

        p_Ht = f([r, vr, az, el])

    Radial velocity does not directly enter
    instantaneous Cartesian position, so its
    column is zero.
    """

    r = measurement.range_m
    az = measurement.azimuth_rad
    el = measurement.elevation_rad

    ce = cos(el)
    se = sin(el)

    ca = cos(az)
    sa = sin(az)

    return (
        (
            ce * ca,
            0.0,
            -r * ce * sa,
            -r * se * ca,
        ),
        (
            ce * sa,
            0.0,
            r * ce * ca,
            -r * se * sa,
        ),
        (
            se,
            0.0,
            0.0,
            r * ce,
        ),
    )


def position_covariance_Ht(
    measurement: MeasurementSnapshot,
) -> Matrix3:
    """
    Propagate the complete Stage2 4x4
    measurement covariance through the
    spherical->Cartesian position mapping:

        Sigma_p = J R_t J^T
    """

    J = position_jacobian_Ht(
        measurement
    )

    R = measurement.covariance_4x4

    result = _matmul(
        _matmul(
            J,
            R,
        ),
        _transpose(J),
    )

    result = _symmetrize3(
        result
    )

    for i in range(3):
        if result[i][i] < -1e-12:
            raise ValueError(
                "Propagated position covariance "
                "has a negative diagonal."
            )

    return result


def point_Ht_to_H0(
    point_Ht: Vec3,
    *,
    T_Ht_from_W,
    T_H0_from_W,
) -> Vec3:
    """
    Convert a point from dynamic H_t into
    canonical anchor headlamp H_0.

    Uses the frozen Stage1 transforms:

        H_t -> W -> H_0
    """

    T_W_from_Ht = (
        T_Ht_from_W.inverse()
    )

    point_W = (
        T_W_from_Ht.apply_point(
            point_Ht
        )
    )

    point_H0 = (
        T_H0_from_W.apply_point(
            point_W
        )
    )

    return tuple(
        float(x)
        for x in point_H0
    )  # type: ignore[return-value]


def vector_Ht_to_H0(
    vector_Ht: Vec3,
    *,
    T_Ht_from_W,
    T_H0_from_W,
) -> Vec3:
    """
    Rotation-only vector transform H_t -> H_0.
    Translation must not affect covariance.
    """

    T_W_from_Ht = (
        T_Ht_from_W.inverse()
    )

    vector_W = (
        T_W_from_Ht.apply_vector(
            vector_Ht
        )
    )

    vector_H0 = (
        T_H0_from_W.apply_vector(
            vector_W
        )
    )

    return tuple(
        float(x)
        for x in vector_H0
    )  # type: ignore[return-value]


def rotation_H0_from_Ht(
    *,
    T_Ht_from_W,
    T_H0_from_W,
) -> Matrix3:
    """
    Recover R_H0_from_Ht through the frozen
    RigidTransform vector API, without
    reimplementing Stage1 transform internals.
    """

    basis = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    columns = tuple(
        vector_Ht_to_H0(
            e,
            T_Ht_from_W=T_Ht_from_W,
            T_H0_from_W=T_H0_from_W,
        )
        for e in basis
    )

    return tuple(
        tuple(
            columns[j][i]
            for j in range(3)
        )
        for i in range(3)
    )  # type: ignore[return-value]


def covariance_Ht_to_H0(
    covariance_Ht: Matrix3,
    *,
    T_Ht_from_W,
    T_H0_from_W,
) -> Matrix3:
    rotation = rotation_H0_from_Ht(
        T_Ht_from_W=T_Ht_from_W,
        T_H0_from_W=T_H0_from_W,
    )

    result = _matmul(
        _matmul(
            rotation,
            covariance_Ht,
        ),
        _transpose(rotation),
    )

    return _symmetrize3(
        result
    )


@dataclass(frozen=True)
class FrameTransformContext:
    """
    Transform context aligned with the Stage3
    AlgorithmObservationSequence frame order.

    T_H0_from_W:
        frozen anchor headlamp transform.

    T_Ht_from_W_by_frame:
        one Stage1 dynamic headlamp transform
        for each Stage2 frame.
    """

    T_H0_from_W: object

    T_Ht_from_W_by_frame: tuple[
        object | None,
        ...
    ]

    def transform_for_frame(
        self,
        frame_index: int,
    ):
        if frame_index < 0:
            raise IndexError(
                "Negative frame index."
            )

        if frame_index >= len(
            self.T_Ht_from_W_by_frame
        ):
            raise IndexError(
                "No dynamic headlamp transform "
                "for requested frame."
            )

        transform = (
            self.T_Ht_from_W_by_frame[
                frame_index
            ]
        )

        if transform is None:
            raise ValueError(
                "Dynamic headlamp transform "
                "is unavailable for this frame."
            )

        return transform


def context_from_stage1_frames(
    *,
    anchor_frames,
    dynamic_frames,
) -> FrameTransformContext:
    """
    Thin bridge to the frozen Stage1 API.

    Expected frozen fields:
      AnchorFrames.T_H0_from_W
      DynamicHeadlampFrame.T_Ht_from_W
    """

    if not hasattr(
        anchor_frames,
        "T_H0_from_W",
    ):
        raise TypeError(
            "Anchor frame object does not expose "
            "T_H0_from_W."
        )

    transforms = []

    for frame in dynamic_frames:
        if frame is None:
            transforms.append(None)
            continue

        if not hasattr(
            frame,
            "T_Ht_from_W",
        ):
            raise TypeError(
                "Dynamic headlamp frame does not "
                "expose T_Ht_from_W."
            )

        transforms.append(
            frame.T_Ht_from_W
        )

    return FrameTransformContext(
        T_H0_from_W=(
            anchor_frames.T_H0_from_W
        ),
        T_Ht_from_W_by_frame=tuple(
            transforms
        ),
    )


@dataclass(frozen=True)
class CartesianObservation:
    """
    One associated Stage2 detection expressed
    in the single canonical anchor headlamp
    frame H0.
    """

    track_id: str
    frame_index: int
    timestamp_s: float

    detection_key: object

    position_H0_m: Vec3

    position_covariance_H0_m2: Matrix3

    measurement: MeasurementSnapshot


def cartesianize_associated_detection(
    associated: AssociatedDetection,
    *,
    context: FrameTransformContext,
) -> CartesianObservation:
    measurement = (
        associated.detection
    )

    point_Ht = spherical_position_Ht(
        measurement
    )

    covariance_Ht = (
        position_covariance_Ht(
            measurement
        )
    )

    T_Ht_from_W = (
        context.transform_for_frame(
            associated.frame_index
        )
    )

    point_H0 = point_Ht_to_H0(
        point_Ht,
        T_Ht_from_W=T_Ht_from_W,
        T_H0_from_W=context.T_H0_from_W,
    )

    covariance_H0 = (
        covariance_Ht_to_H0(
            covariance_Ht,
            T_Ht_from_W=T_Ht_from_W,
            T_H0_from_W=(
                context.T_H0_from_W
            ),
        )
    )

    return CartesianObservation(
        track_id=associated.track_id,
        frame_index=(
            associated.frame_index
        ),
        timestamp_s=(
            associated.timestamp_s
        ),
        detection_key=(
            measurement.detection_key
        ),
        position_H0_m=point_H0,
        position_covariance_H0_m2=(
            covariance_H0
        ),
        measurement=measurement,
    )


def cartesianize_associated_track(
    track: AssociatedTrack,
    *,
    context: FrameTransformContext,
) -> tuple[
    CartesianObservation,
    ...
]:
    result = tuple(
        cartesianize_associated_detection(
            item,
            context=context,
        )
        for item in track.detections
    )

    previous_time = None

    for observation in result:
        if (
            previous_time is not None
            and observation.timestamp_s
            <= previous_time
        ):
            raise ValueError(
                "Track timestamps must be "
                "strictly increasing."
            )

        previous_time = (
            observation.timestamp_s
        )

    return result


def point_H0_to_Ht(
    point_H0: Vec3,
    *,
    T_Ht_from_W,
    T_H0_from_W,
) -> Vec3:
    """
    Convert a point from canonical H0 into
    the dynamic measurement frame Ht:

        H0 -> W -> Ht
    """

    T_W_from_H0 = (
        T_H0_from_W.inverse()
    )

    point_W = (
        T_W_from_H0.apply_point(
            point_H0
        )
    )

    point_Ht = (
        T_Ht_from_W.apply_point(
            point_W
        )
    )

    return tuple(
        float(x)
        for x in point_Ht
    )  # type: ignore[return-value]


def vector_H0_to_Ht(
    vector_H0: Vec3,
    *,
    T_Ht_from_W,
    T_H0_from_W,
) -> Vec3:
    """
    Rotation-only vector conversion H0 -> Ht.
    """

    T_W_from_H0 = (
        T_H0_from_W.inverse()
    )

    vector_W = (
        T_W_from_H0.apply_vector(
            vector_H0
        )
    )

    vector_Ht = (
        T_Ht_from_W.apply_vector(
            vector_W
        )
    )

    return tuple(
        float(x)
        for x in vector_Ht
    )  # type: ignore[return-value]


def rotation_Ht_from_H0(
    *,
    T_Ht_from_W,
    T_H0_from_W,
) -> Matrix3:
    """
    Recover R_Ht_from_H0 using only the
    frozen Stage1 public transform API.
    """

    basis = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    columns = tuple(
        vector_H0_to_Ht(
            e,
            T_Ht_from_W=T_Ht_from_W,
            T_H0_from_W=T_H0_from_W,
        )
        for e in basis
    )

    return tuple(
        tuple(
            columns[j][i]
            for j in range(3)
        )
        for i in range(3)
    )  # type: ignore[return-value]


def sensor_origin_H0(
    *,
    frame_index: int,
    context: FrameTransformContext,
) -> Vec3:
    """
    Position of dynamic headlamp Ht origin,
    expressed in canonical H0.
    """

    T_Ht_from_W = (
        context.transform_for_frame(
            frame_index
        )
    )

    T_W_from_Ht = (
        T_Ht_from_W.inverse()
    )

    origin_W = (
        T_W_from_Ht.apply_point(
            (0.0, 0.0, 0.0)
        )
    )

    origin_H0 = (
        context.T_H0_from_W.apply_point(
            origin_W
        )
    )

    return tuple(
        float(x)
        for x in origin_H0
    )  # type: ignore[return-value]


def sensor_velocity_H0(
    *,
    frame_index: int,
    frame_timestamps_s: tuple[
        float,
        ...
    ],
    context: FrameTransformContext,
) -> Vec3:
    """
    Strict-adjacent causal headlamp velocity,
    reconstructed from frozen Stage1 poses.

    This mirrors the causal Stage2 semantics:
    current pose minus immediately previous
    pose, never a future-centered difference.
    """

    if frame_index <= 0:
        raise ValueError(
            "Sensor velocity requires a "
            "previous frame."
        )

    if frame_index >= len(
        frame_timestamps_s
    ):
        raise IndexError(
            "No timestamp for frame."
        )

    current_t = float(
        frame_timestamps_s[
            frame_index
        ]
    )

    previous_t = float(
        frame_timestamps_s[
            frame_index - 1
        ]
    )

    dt = current_t - previous_t

    if dt <= 0.0:
        raise ValueError(
            "Frame timestamps must strictly "
            "increase."
        )

    current = sensor_origin_H0(
        frame_index=frame_index,
        context=context,
    )

    previous = sensor_origin_H0(
        frame_index=frame_index - 1,
        context=context,
    )

    return tuple(
        (
            current[i]
            -
            previous[i]
        )
        / dt
        for i in range(3)
    )  # type: ignore[return-value]
