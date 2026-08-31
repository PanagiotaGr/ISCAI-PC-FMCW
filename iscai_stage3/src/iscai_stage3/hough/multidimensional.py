from __future__ import annotations

from dataclasses import dataclass, replace
from math import (
    cos,
    exp,
    floor,
    hypot,
    isfinite,
    sin,
    sqrt,
)

import numpy as np

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
    covariance_Ht_to_H0,
    point_Ht_to_H0,
    position_covariance_Ht,
    sensor_velocity_H0,
    spherical_position_Ht,
    vector_H0_to_Ht,
)

from iscai_stage3.observations import (
    frame_detections,
    frame_timestamp_s,
    snapshot_from_detection,
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


@dataclass(frozen=True)
class HoughConfig:
    """
    Sparse time-aware Multidimensional Hough.

    Parameter space:

        [x_anchor, y_anchor, vx, vy]

    The anchor time is the last causal input
    frame.

    Numerical choices are declared Stage3
    baseline parameters and are frozen before
    formal validation.
    """

    velocity_min_mps: float = -30.0
    velocity_max_mps: float = 30.0
    velocity_step_mps: float = 2.0

    anchor_position_bin_m: float = 1.0

    position_vote_neighbor_bins: int = 1

    position_vote_floor_m: float = 0.75

    radial_velocity_floor_mps: float = 2.0
    radial_velocity_gate_sigma: float = 4.0

    minimum_vote_weight: float = 1e-8

    minimum_vote_frames: int = 3

    maximum_peaks: int = 16

    nms_position_bins: int = 2
    nms_velocity_bins: int = 1

    support_position_floor_m: float = 1.0
    support_spatial_gate_sigma: float = 3.0
    support_radial_gate_sigma: float = 4.0

    minimum_support_frames: int = 3
    minimum_time_span_s: float = 0.2
    maximum_missing_gap_frames: int = 2

    duplicate_support_overlap: float = 0.60
    duplicate_velocity_gate_mps: float = 3.0

    def __post_init__(self) -> None:
        numeric = (
            self.velocity_min_mps,
            self.velocity_max_mps,
            self.velocity_step_mps,
            self.anchor_position_bin_m,
            self.position_vote_floor_m,
            self.radial_velocity_floor_mps,
            self.radial_velocity_gate_sigma,
            self.minimum_vote_weight,
            self.support_position_floor_m,
            self.support_spatial_gate_sigma,
            self.support_radial_gate_sigma,
            self.minimum_time_span_s,
            self.duplicate_support_overlap,
            self.duplicate_velocity_gate_mps,
        )

        if not all(
            isfinite(float(x))
            for x in numeric
        ):
            raise ValueError(
                "Hough numeric configuration "
                "must be finite."
            )

        if (
            self.velocity_max_mps
            <=
            self.velocity_min_mps
        ):
            raise ValueError(
                "Hough velocity range is invalid."
            )

        if self.velocity_step_mps <= 0.0:
            raise ValueError(
                "Hough velocity step must "
                "be positive."
            )

        if self.anchor_position_bin_m <= 0.0:
            raise ValueError(
                "Hough position-bin width "
                "must be positive."
            )

        if self.position_vote_floor_m <= 0.0:
            raise ValueError(
                "Position voting floor "
                "must be positive."
            )

        if self.radial_velocity_floor_mps <= 0.0:
            raise ValueError(
                "Radial-velocity floor "
                "must be positive."
            )

        if self.radial_velocity_gate_sigma <= 0.0:
            raise ValueError(
                "Radial gate must be positive."
            )

        if self.minimum_vote_weight < 0.0:
            raise ValueError(
                "Minimum vote weight cannot "
                "be negative."
            )

        integer_positive = (
            self.minimum_vote_frames,
            self.maximum_peaks,
            self.minimum_support_frames,
        )

        if any(
            int(x) <= 0
            for x in integer_positive
        ):
            raise ValueError(
                "Required Hough counts must "
                "be positive."
            )

        integer_nonnegative = (
            self.position_vote_neighbor_bins,
            self.nms_position_bins,
            self.nms_velocity_bins,
            self.maximum_missing_gap_frames,
        )

        if any(
            int(x) < 0
            for x in integer_nonnegative
        ):
            raise ValueError(
                "Hough integer gates cannot "
                "be negative."
            )

        if not (
            0.0
            <=
            self.duplicate_support_overlap
            <=
            1.0
        ):
            raise ValueError(
                "Duplicate overlap must lie "
                "in [0,1]."
            )


@dataclass(frozen=True)
class HoughFrameContext:
    transforms: FrameTransformContext

    frame_timestamps_s: tuple[
        float,
        ...
    ]

    def __post_init__(self) -> None:
        if len(
            self.frame_timestamps_s
        ) != len(
            self.transforms
            .T_Ht_from_W_by_frame
        ):
            raise ValueError(
                "Hough timestamps/transforms "
                "must have equal length."
            )

        previous = None

        for value in self.frame_timestamps_s:
            current = float(value)

            if not isfinite(current):
                raise ValueError(
                    "Hough timestamps must "
                    "be finite."
                )

            if (
                previous is not None
                and current <= previous
            ):
                raise ValueError(
                    "Hough timestamps must "
                    "strictly increase."
                )

            previous = current


@dataclass(frozen=True)
class HoughDetection:
    """
    Raw algorithm-facing detection.

    detection_index is only its local position
    within one unlabeled frame. It is not a
    cross-frame identity.
    """

    frame_index: int
    detection_index: int

    timestamp_s: float

    position_H0_m: Vec3

    position_covariance_H0_m2: Matrix3

    radial_velocity_mps: float
    radial_velocity_variance_m2_s2: float

    measurement: object


@dataclass(frozen=True)
class HoughFrame:
    frame_index: int
    timestamp_s: float

    detections: tuple[
        HoughDetection,
        ...
    ]


@dataclass(frozen=True)
class HoughPeak:
    """
    One local maximum in the sparse 4D
    accumulator.
    """

    x_anchor_bin: int
    y_anchor_bin: int

    vx_bin: int
    vy_bin: int

    x_anchor_m: float
    y_anchor_m: float

    vx_mps: float
    vy_mps: float

    score: float

    vote_frames: tuple[
        int,
        ...
    ]


@dataclass(frozen=True)
class HoughSupportRef:
    frame_index: int
    detection_index: int

    spatial_mahalanobis_sq: float

    radial_normalized_residual: float


@dataclass(frozen=True)
class HoughTrack:
    track_id: str

    anchor_frame_index: int
    anchor_timestamp_s: float

    anchor_position_H0_m: Vec3

    velocity_H0_mps: Vec3

    support_refs: tuple[
        HoughSupportRef,
        ...
    ]

    coverage: float
    time_span_s: float

    mean_residual_m: float

    accumulator_score: float

    source_peak: HoughPeak

    truth_used: bool = False
    actor_identity_used: bool = False
    detection_key_matching_used: bool = False
    actor_class_used: bool = False
    future_information_used: bool = False

    def __post_init__(self) -> None:
        if (
            self.truth_used
            or
            self.actor_identity_used
            or
            self.detection_key_matching_used
            or
            self.actor_class_used
            or
            self.future_information_used
        ):
            raise ValueError(
                "Hough track contains "
                "prohibited information."
            )


@dataclass(frozen=True)
class HoughResult:
    scenario_id: str

    anchor_frame_index: int
    anchor_timestamp_s: float

    accumulator_cells: int

    peaks: tuple[
        HoughPeak,
        ...
    ]

    tracks: tuple[
        HoughTrack,
        ...
    ]

    input_mode: str = (
        "raw_unlabeled_stage2_track_before_detect"
    )

    parameter_space: str = (
        "x_anchor_y_anchor_vx_vy"
    )

    truth_used: bool = False
    estimated_association_used: bool = False
    future_information_used: bool = False


def _nearest_bin(
    value: float,
    width: float,
) -> int:
    scaled = (
        value
        /
        width
    )

    return int(
        floor(
            scaled
            +
            0.5
        )
    )


def _velocity_values(
    config: HoughConfig,
) -> tuple[
    float,
    ...
]:
    count = int(
        floor(
            (
                config.velocity_max_mps
                -
                config.velocity_min_mps
            )
            /
            config.velocity_step_mps
            +
            0.5
        )
    )

    values = tuple(
        config.velocity_min_mps
        +
        index
        *
        config.velocity_step_mps
        for index in range(
            count + 1
        )
    )

    if not values:
        raise ValueError(
            "Empty Hough velocity grid."
        )

    return values


def build_raw_hough_frames(
    sequence: AlgorithmObservationSequence,
    *,
    context: HoughFrameContext,
) -> tuple[
    HoughFrame,
    ...
]:
    """
    Convert raw Stage2 unlabeled detections
    directly into H0.

    No Stage3 association layer is called.
    """

    if len(
        sequence.frames
    ) != len(
        context.frame_timestamps_s
    ):
        raise ValueError(
            "Sequence/context frame counts "
            "do not match."
        )

    result = []

    for frame_index, frame in enumerate(
        sequence.frames
    ):
        timestamp = float(
            frame_timestamp_s(
                frame
            )
        )

        if abs(
            timestamp
            -
            context.frame_timestamps_s[
                frame_index
            ]
        ) > 1e-9:
            raise ValueError(
                "Hough context timestamp "
                "does not match Stage2 frame."
            )

        T_Ht_from_W = (
            context.transforms
            .transform_for_frame(
                frame_index
            )
        )

        converted = []

        for detection_index, detection in enumerate(
            frame_detections(
                frame
            )
        ):
            measurement = (
                snapshot_from_detection(
                    detection
                )
            )

            position_Ht = (
                spherical_position_Ht(
                    measurement
                )
            )

            covariance_Ht = (
                position_covariance_Ht(
                    measurement
                )
            )

            position_H0 = (
                point_Ht_to_H0(
                    position_Ht,
                    T_Ht_from_W=(
                        T_Ht_from_W
                    ),
                    T_H0_from_W=(
                        context.transforms
                        .T_H0_from_W
                    ),
                )
            )

            covariance_H0 = (
                covariance_Ht_to_H0(
                    covariance_Ht,
                    T_Ht_from_W=(
                        T_Ht_from_W
                    ),
                    T_H0_from_W=(
                        context.transforms
                        .T_H0_from_W
                    ),
                )
            )

            R = np.asarray(
                measurement
                .covariance_4x4,
                dtype=np.float64,
            )

            radial_variance = float(
                max(
                    R[1, 1],
                    0.0,
                )
            )

            converted.append(
                HoughDetection(
                    frame_index=frame_index,
                    detection_index=(
                        detection_index
                    ),
                    timestamp_s=timestamp,
                    position_H0_m=(
                        position_H0
                    ),
                    position_covariance_H0_m2=(
                        covariance_H0
                    ),
                    radial_velocity_mps=(
                        measurement
                        .radial_velocity_mps
                    ),
                    radial_velocity_variance_m2_s2=(
                        radial_variance
                    ),
                    measurement=measurement,
                )
            )

        result.append(
            HoughFrame(
                frame_index=frame_index,
                timestamp_s=timestamp,
                detections=tuple(
                    converted
                ),
            )
        )

    return tuple(result)


def _radial_normalized_residual(
    detection: HoughDetection,
    *,
    candidate_velocity_H0_mps: Vec3,
    context: HoughFrameContext,
    config: HoughConfig,
) -> float:
    """
    Compare candidate world/H0 velocity to
    measured Stage2 radial velocity.

    The first frame has no strict-adjacent
    headlamp velocity, so radial consistency
    is deliberately not used there.
    """

    if detection.frame_index <= 0:
        return 0.0

    sensor_velocity = (
        sensor_velocity_H0(
            frame_index=(
                detection.frame_index
            ),
            frame_timestamps_s=(
                context.frame_timestamps_s
            ),
            context=context.transforms,
        )
    )

    T_Ht_from_W = (
        context.transforms
        .transform_for_frame(
            detection.frame_index
        )
    )

    actor_velocity_Ht = np.asarray(
        vector_H0_to_Ht(
            candidate_velocity_H0_mps,
            T_Ht_from_W=T_Ht_from_W,
            T_H0_from_W=(
                context.transforms
                .T_H0_from_W
            ),
        ),
        dtype=np.float64,
    )

    sensor_velocity_Ht = np.asarray(
        vector_H0_to_Ht(
            sensor_velocity,
            T_Ht_from_W=T_Ht_from_W,
            T_H0_from_W=(
                context.transforms
                .T_H0_from_W
            ),
        ),
        dtype=np.float64,
    )

    measurement = detection.measurement

    az = float(
        measurement.azimuth_rad
    )

    el = float(
        measurement.elevation_rad
    )

    ce = cos(el)

    unit_los = np.asarray(
        [
            ce * cos(az),
            ce * sin(az),
            sin(el),
        ],
        dtype=np.float64,
    )

    predicted_radial = float(
        unit_los
        @
        (
            actor_velocity_Ht
            -
            sensor_velocity_Ht
        )
    )

    sigma = sqrt(
        detection
        .radial_velocity_variance_m2_s2
        +
        config
        .radial_velocity_floor_mps
        ** 2
    )

    return (
        detection.radial_velocity_mps
        -
        predicted_radial
    ) / sigma


def _position_vote_weights(
    *,
    anchor_position_xy: np.ndarray,
    covariance_xy: np.ndarray,
    config: HoughConfig,
):
    floor_variance = (
        config
        .position_vote_floor_m
        ** 2
    )

    effective = (
        covariance_xy
        +
        floor_variance
        *
        np.eye(
            2,
            dtype=np.float64,
        )
    )

    determinant = float(
        np.linalg.det(
            effective
        )
    )

    if determinant <= 0.0:
        inverse = np.linalg.pinv(
            effective
        )

        determinant = max(
            float(
                np.prod(
                    np.maximum(
                        np.linalg.eigvalsh(
                            effective
                        ),
                        1e-12,
                    )
                )
            ),
            1e-12,
        )

    else:
        inverse = np.linalg.inv(
            effective
        )

    # Equals approximately 1 at the declared
    # uncertainty floor and decreases as
    # measurement uncertainty grows.
    confidence = min(
        1.0,
        floor_variance
        /
        sqrt(
            determinant
        ),
    )

    center_x_bin = _nearest_bin(
        float(anchor_position_xy[0]),
        config.anchor_position_bin_m,
    )

    center_y_bin = _nearest_bin(
        float(anchor_position_xy[1]),
        config.anchor_position_bin_m,
    )

    radius = (
        config
        .position_vote_neighbor_bins
    )

    for dx in range(
        -radius,
        radius + 1,
    ):
        for dy in range(
            -radius,
            radius + 1,
        ):
            x_bin = (
                center_x_bin
                +
                dx
            )

            y_bin = (
                center_y_bin
                +
                dy
            )

            center = np.asarray(
                [
                    x_bin
                    *
                    config
                    .anchor_position_bin_m,

                    y_bin
                    *
                    config
                    .anchor_position_bin_m,
                ],
                dtype=np.float64,
            )

            residual = (
                anchor_position_xy
                -
                center
            )

            mahalanobis_sq = float(
                residual
                @
                inverse
                @
                residual
            )

            weight = (
                confidence
                *
                exp(
                    -0.5
                    *
                    mahalanobis_sq
                )
            )

            yield (
                x_bin,
                y_bin,
                weight,
            )


def build_sparse_accumulator(
    frames: tuple[
        HoughFrame,
        ...
    ],
    *,
    context: HoughFrameContext,
    config: HoughConfig,
):
    """
    Sparse 4D accumulator:

        [x_anchor_bin,
         y_anchor_bin,
         vx_bin,
         vy_bin]

    Each cell stores at most one effective vote
    per input frame. Multiple clutter points in
    one frame therefore cannot inflate temporal
    support by themselves.
    """

    if not frames:
        raise ValueError(
            "Hough requires at least one frame."
        )

    anchor_time = (
        frames[-1]
        .timestamp_s
    )

    velocities = _velocity_values(
        config
    )

    accumulator: dict[
        tuple[int, int, int, int],
        dict[int, float],
    ] = {}

    for frame in frames:
        delta_to_anchor = (
            anchor_time
            -
            frame.timestamp_s
        )

        if delta_to_anchor < -1e-12:
            raise ValueError(
                "Hough received future-ordered "
                "frames."
            )

        for detection in frame.detections:
            covariance_xy = np.asarray(
                [
                    [
                        detection
                        .position_covariance_H0_m2[0][0],

                        detection
                        .position_covariance_H0_m2[0][1],
                    ],
                    [
                        detection
                        .position_covariance_H0_m2[1][0],

                        detection
                        .position_covariance_H0_m2[1][1],
                    ],
                ],
                dtype=np.float64,
            )

            for vx_index, vx in enumerate(
                velocities
            ):
                for vy_index, vy in enumerate(
                    velocities
                ):
                    candidate_velocity = (
                        float(vx),
                        float(vy),
                        0.0,
                    )

                    radial_normalized = (
                        _radial_normalized_residual(
                            detection,
                            candidate_velocity_H0_mps=(
                                candidate_velocity
                            ),
                            context=context,
                            config=config,
                        )
                    )

                    if abs(
                        radial_normalized
                    ) > (
                        config
                        .radial_velocity_gate_sigma
                    ):
                        continue

                    sigma_vr = sqrt(
                        detection
                        .radial_velocity_variance_m2_s2
                        +
                        config
                        .radial_velocity_floor_mps
                        ** 2
                    )

                    radial_confidence = min(
                        1.0,
                        config
                        .radial_velocity_floor_mps
                        /
                        sigma_vr,
                    )

                    radial_weight = (
                        radial_confidence
                        *
                        exp(
                            -0.5
                            *
                            radial_normalized
                            *
                            radial_normalized
                        )
                    )

                    anchor_xy = np.asarray(
                        [
                            detection
                            .position_H0_m[0]
                            +
                            vx
                            *
                            delta_to_anchor,

                            detection
                            .position_H0_m[1]
                            +
                            vy
                            *
                            delta_to_anchor,
                        ],
                        dtype=np.float64,
                    )

                    for (
                        x_bin,
                        y_bin,
                        spatial_weight,
                    ) in _position_vote_weights(
                        anchor_position_xy=(
                            anchor_xy
                        ),
                        covariance_xy=(
                            covariance_xy
                        ),
                        config=config,
                    ):
                        weight = (
                            spatial_weight
                            *
                            radial_weight
                        )

                        if (
                            weight
                            <
                            config
                            .minimum_vote_weight
                        ):
                            continue

                        key = (
                            x_bin,
                            y_bin,
                            vx_index,
                            vy_index,
                        )

                        frame_votes = (
                            accumulator
                            .setdefault(
                                key,
                                {},
                            )
                        )

                        previous = (
                            frame_votes.get(
                                frame.frame_index,
                                0.0,
                            )
                        )

                        if weight > previous:
                            frame_votes[
                                frame.frame_index
                            ] = weight

    return accumulator


def extract_hough_peaks(
    accumulator,
    *,
    config: HoughConfig,
) -> tuple[
    HoughPeak,
    ...
]:
    velocities = _velocity_values(
        config
    )

    candidates = []

    for key, frame_weights in (
        accumulator.items()
    ):
        if len(
            frame_weights
        ) < (
            config
            .minimum_vote_frames
        ):
            continue

        (
            x_bin,
            y_bin,
            vx_index,
            vy_index,
        ) = key

        score = float(
            sum(
                frame_weights.values()
            )
        )

        candidates.append(
            HoughPeak(
                x_anchor_bin=x_bin,
                y_anchor_bin=y_bin,
                vx_bin=vx_index,
                vy_bin=vy_index,
                x_anchor_m=(
                    x_bin
                    *
                    config
                    .anchor_position_bin_m
                ),
                y_anchor_m=(
                    y_bin
                    *
                    config
                    .anchor_position_bin_m
                ),
                vx_mps=float(
                    velocities[
                        vx_index
                    ]
                ),
                vy_mps=float(
                    velocities[
                        vy_index
                    ]
                ),
                score=score,
                vote_frames=tuple(
                    sorted(
                        frame_weights
                    )
                ),
            )
        )

    candidates.sort(
        key=lambda peak: (
            -peak.score,
            -len(
                peak.vote_frames
            ),
            peak.x_anchor_bin,
            peak.y_anchor_bin,
            peak.vx_bin,
            peak.vy_bin,
        )
    )

    selected = []

    for candidate in candidates:
        suppressed = False

        for accepted in selected:
            if (
                abs(
                    candidate.x_anchor_bin
                    -
                    accepted.x_anchor_bin
                )
                <=
                config.nms_position_bins
                and
                abs(
                    candidate.y_anchor_bin
                    -
                    accepted.y_anchor_bin
                )
                <=
                config.nms_position_bins
                and
                abs(
                    candidate.vx_bin
                    -
                    accepted.vx_bin
                )
                <=
                config.nms_velocity_bins
                and
                abs(
                    candidate.vy_bin
                    -
                    accepted.vy_bin
                )
                <=
                config.nms_velocity_bins
            ):
                suppressed = True
                break

        if suppressed:
            continue

        selected.append(
            candidate
        )

        if len(
            selected
        ) >= config.maximum_peaks:
            break

    return tuple(selected)


def _spatial_mahalanobis_sq(
    detection: HoughDetection,
    *,
    predicted_xy: np.ndarray,
    config: HoughConfig,
) -> float:
    observed = np.asarray(
        detection
        .position_H0_m[0:2],
        dtype=np.float64,
    )

    residual = (
        observed
        -
        predicted_xy
    )

    covariance = np.asarray(
        [
            [
                detection
                .position_covariance_H0_m2[0][0],

                detection
                .position_covariance_H0_m2[0][1],
            ],
            [
                detection
                .position_covariance_H0_m2[1][0],

                detection
                .position_covariance_H0_m2[1][1],
            ],
        ],
        dtype=np.float64,
    )

    covariance += (
        config
        .support_position_floor_m
        ** 2
        *
        np.eye(
            2,
            dtype=np.float64,
        )
    )

    try:
        solved = np.linalg.solve(
            covariance,
            residual,
        )

    except np.linalg.LinAlgError:
        solved = (
            np.linalg.pinv(
                covariance
            )
            @
            residual
        )

    return float(
        residual
        @
        solved
    )


def _select_peak_support(
    peak: HoughPeak,
    *,
    frames: tuple[
        HoughFrame,
        ...
    ],
    context: HoughFrameContext,
    config: HoughConfig,
) -> tuple[
    HoughSupportRef,
    ...
]:
    anchor_time = (
        frames[-1]
        .timestamp_s
    )

    result = []

    candidate_velocity = (
        peak.vx_mps,
        peak.vy_mps,
        0.0,
    )

    for frame in frames:
        dt = (
            frame.timestamp_s
            -
            anchor_time
        )

        predicted_xy = np.asarray(
            [
                peak.x_anchor_m
                +
                peak.vx_mps
                *
                dt,

                peak.y_anchor_m
                +
                peak.vy_mps
                *
                dt,
            ],
            dtype=np.float64,
        )

        best = None

        for detection in frame.detections:
            spatial = (
                _spatial_mahalanobis_sq(
                    detection,
                    predicted_xy=(
                        predicted_xy
                    ),
                    config=config,
                )
            )

            if spatial > (
                config
                .support_spatial_gate_sigma
                ** 2
            ):
                continue

            radial = (
                _radial_normalized_residual(
                    detection,
                    candidate_velocity_H0_mps=(
                        candidate_velocity
                    ),
                    context=context,
                    config=config,
                )
            )

            if abs(radial) > (
                config
                .support_radial_gate_sigma
            ):
                continue

            cost = (
                spatial
                +
                radial
                *
                radial
            )

            candidate = (
                cost,
                detection.detection_index,
                spatial,
                radial,
            )

            if (
                best is None
                or
                candidate
                <
                best
            ):
                best = candidate

        if best is not None:
            (
                _cost,
                detection_index,
                spatial,
                radial,
            ) = best

            result.append(
                HoughSupportRef(
                    frame_index=(
                        frame.frame_index
                    ),
                    detection_index=(
                        detection_index
                    ),
                    spatial_mahalanobis_sq=(
                        float(spatial)
                    ),
                    radial_normalized_residual=(
                        float(radial)
                    ),
                )
            )

    return tuple(result)


def _support_is_valid(
    support: tuple[
        HoughSupportRef,
        ...
    ],
    *,
    frames: tuple[
        HoughFrame,
        ...
    ],
    config: HoughConfig,
) -> bool:
    if len(
        support
    ) < (
        config
        .minimum_support_frames
    ):
        return False

    frame_indices = tuple(
        item.frame_index
        for item in support
    )

    frame_lookup = {
        frame.frame_index: frame
        for frame in frames
    }

    first_time = (
        frame_lookup[
            frame_indices[0]
        ].timestamp_s
    )

    last_time = (
        frame_lookup[
            frame_indices[-1]
        ].timestamp_s
    )

    if (
        last_time
        -
        first_time
    ) < (
        config
        .minimum_time_span_s
    ):
        return False

    for previous, current in zip(
        frame_indices[:-1],
        frame_indices[1:],
    ):
        missing = (
            current
            -
            previous
            -
            1
        )

        if (
            missing
            >
            config
            .maximum_missing_gap_frames
        ):
            return False

    return True


def _weighted_linear_fit(
    times: np.ndarray,
    values: np.ndarray,
    variances: np.ndarray,
    *,
    anchor_time: float,
    floor_variance: float,
):
    dt = (
        times
        -
        anchor_time
    )

    A = np.column_stack(
        [
            np.ones_like(
                dt
            ),
            dt,
        ]
    )

    weights = (
        1.0
        /
        (
            variances
            +
            floor_variance
        )
    )

    normal = (
        A.T
        @
        (
            weights[
                :,
                None
            ]
            *
            A
        )
    )

    rhs = (
        A.T
        @
        (
            weights
            *
            values
        )
    )

    try:
        coefficients = np.linalg.solve(
            normal,
            rhs,
        )

    except np.linalg.LinAlgError:
        coefficients = (
            np.linalg.pinv(
                normal
            )
            @
            rhs
        )

    return (
        float(
            coefficients[0]
        ),
        float(
            coefficients[1]
        ),
    )


def _refine_track(
    peak: HoughPeak,
    support: tuple[
        HoughSupportRef,
        ...
    ],
    *,
    frames: tuple[
        HoughFrame,
        ...
    ],
    config: HoughConfig,
) -> HoughTrack:
    frame_lookup = {
        frame.frame_index: frame
        for frame in frames
    }

    detections = []

    for ref in support:
        frame = frame_lookup[
            ref.frame_index
        ]

        detections.append(
            frame.detections[
                ref.detection_index
            ]
        )

    anchor_frame = (
        frames[-1]
    )

    anchor_time = (
        anchor_frame.timestamp_s
    )

    times = np.asarray(
        [
            detection.timestamp_s
            for detection in detections
        ],
        dtype=np.float64,
    )

    positions = np.asarray(
        [
            detection.position_H0_m
            for detection in detections
        ],
        dtype=np.float64,
    )

    covariances = np.asarray(
        [
            detection
            .position_covariance_H0_m2
            for detection in detections
        ],
        dtype=np.float64,
    )

    intercepts = []
    velocities = []

    floor_variance = (
        config
        .support_position_floor_m
        ** 2
    )

    for axis in range(3):
        intercept, velocity = (
            _weighted_linear_fit(
                times,
                positions[
                    :,
                    axis
                ],
                covariances[
                    :,
                    axis,
                    axis
                ],
                anchor_time=(
                    anchor_time
                ),
                floor_variance=(
                    floor_variance
                ),
            )
        )

        intercepts.append(
            intercept
        )

        velocities.append(
            velocity
        )

    residuals = []

    for detection in detections:
        dt = (
            detection.timestamp_s
            -
            anchor_time
        )

        predicted = np.asarray(
            intercepts,
            dtype=np.float64,
        ) + (
            dt
            *
            np.asarray(
                velocities,
                dtype=np.float64,
            )
        )

        observed = np.asarray(
            detection
            .position_H0_m,
            dtype=np.float64,
        )

        residuals.append(
            float(
                np.linalg.norm(
                    observed
                    -
                    predicted
                )
            )
        )

    frame_indices = tuple(
        ref.frame_index
        for ref in support
    )

    frame_lookup = {
        frame.frame_index: frame
        for frame in frames
    }

    first_time = (
        frame_lookup[
            frame_indices[0]
        ].timestamp_s
    )

    last_time = (
        frame_lookup[
            frame_indices[-1]
        ].timestamp_s
    )

    coverage = (
        len(
            set(
                frame_indices
            )
        )
        /
        len(frames)
    )

    return HoughTrack(
        track_id="UNASSIGNED",
        anchor_frame_index=(
            anchor_frame.frame_index
        ),
        anchor_timestamp_s=(
            anchor_time
        ),
        anchor_position_H0_m=tuple(
            intercepts
        ),  # type: ignore[arg-type]
        velocity_H0_mps=tuple(
            velocities
        ),  # type: ignore[arg-type]
        support_refs=support,
        coverage=float(
            coverage
        ),
        time_span_s=float(
            last_time
            -
            first_time
        ),
        mean_residual_m=float(
            np.mean(
                residuals
            )
        ),
        accumulator_score=(
            peak.score
        ),
        source_peak=peak,
        truth_used=False,
        actor_identity_used=False,
        detection_key_matching_used=False,
        actor_class_used=False,
        future_information_used=False,
    )


def _support_pairs(
    track: HoughTrack,
):
    return {
        (
            ref.frame_index,
            ref.detection_index,
        )
        for ref in track.support_refs
    }


def _deduplicate_tracks(
    tracks: tuple[
        HoughTrack,
        ...
    ],
    *,
    config: HoughConfig,
) -> tuple[
    HoughTrack,
    ...
]:
    ordered = sorted(
        tracks,
        key=lambda track: (
            -len(
                track.support_refs
            ),
            track.mean_residual_m,
            -track.accumulator_score,
            track.anchor_position_H0_m,
            track.velocity_H0_mps,
        ),
    )

    accepted = []

    for candidate in ordered:
        candidate_support = (
            _support_pairs(
                candidate
            )
        )

        duplicate = False

        for existing in accepted:
            existing_support = (
                _support_pairs(
                    existing
                )
            )

            intersection = len(
                candidate_support
                &
                existing_support
            )

            denominator = max(
                1,
                min(
                    len(
                        candidate_support
                    ),
                    len(
                        existing_support
                    ),
                ),
            )

            overlap = (
                intersection
                /
                denominator
            )

            velocity_distance = sqrt(
                sum(
                    (
                        candidate
                        .velocity_H0_mps[i]
                        -
                        existing
                        .velocity_H0_mps[i]
                    )
                    ** 2
                    for i in range(3)
                )
            )

            if (
                overlap
                >=
                config
                .duplicate_support_overlap
                and
                velocity_distance
                <=
                config
                .duplicate_velocity_gate_mps
            ):
                duplicate = True
                break

        if not duplicate:
            accepted.append(
                candidate
            )

    return tuple(
        replace(
            track,
            track_id=f"H{index:03d}",
        )
        for index, track in enumerate(
            accepted,
            start=1,
        )
    )


def run_multidimensional_hough(
    sequence: AlgorithmObservationSequence,
    *,
    context: HoughFrameContext,
    config: HoughConfig | None = None,
) -> HoughResult:
    """
    Main Stage3 MHT/Hough branch.

    This is track-before-detect relative to the
    Stage3 estimated-association branch:
    it operates directly on raw unlabeled
    Stage2 frames.
    """

    if config is None:
        config = HoughConfig()

    frames = build_raw_hough_frames(
        sequence,
        context=context,
    )

    if len(frames) < (
        config
        .minimum_support_frames
    ):
        return HoughResult(
            scenario_id=(
                sequence.scenario_id
            ),
            anchor_frame_index=(
                len(frames) - 1
            ),
            anchor_timestamp_s=(
                frames[-1].timestamp_s
                if frames
                else float("nan")
            ),
            accumulator_cells=0,
            peaks=(),
            tracks=(),
            truth_used=False,
            estimated_association_used=False,
            future_information_used=False,
        )

    accumulator = (
        build_sparse_accumulator(
            frames,
            context=context,
            config=config,
        )
    )

    peaks = extract_hough_peaks(
        accumulator,
        config=config,
    )

    tracks = []

    for peak in peaks:
        support = (
            _select_peak_support(
                peak,
                frames=frames,
                context=context,
                config=config,
            )
        )

        # Part-A-style AND-logic:
        #
        # accumulator peak
        # AND sufficient temporal support
        # AND sufficient temporal span
        # AND bounded miss gap
        # AND spatial/radial consistency.

        if not _support_is_valid(
            support,
            frames=frames,
            config=config,
        ):
            continue

        track = _refine_track(
            peak,
            support,
            frames=frames,
            config=config,
        )

        tracks.append(
            track
        )

    tracks = _deduplicate_tracks(
        tuple(tracks),
        config=config,
    )

    return HoughResult(
        scenario_id=(
            sequence.scenario_id
        ),
        anchor_frame_index=(
            frames[-1]
            .frame_index
        ),
        anchor_timestamp_s=(
            frames[-1]
            .timestamp_s
        ),
        accumulator_cells=len(
            accumulator
        ),
        peaks=peaks,
        tracks=tracks,
        truth_used=False,
        estimated_association_used=False,
        future_information_used=False,
    )
