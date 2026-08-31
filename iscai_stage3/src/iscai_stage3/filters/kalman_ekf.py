from __future__ import annotations

from dataclasses import dataclass
from math import (
    atan2,
    isfinite,
    sqrt,
)

import numpy as np

from iscai_stage3.association import (
    AssociatedTrack,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
    cartesianize_associated_track,
    point_H0_to_Ht,
    rotation_Ht_from_H0,
    sensor_velocity_H0,
    vector_H0_to_Ht,
)

from iscai_stage3.observations import (
    MeasurementSnapshot,
    wrap_angle_rad,
)

from iscai_stage3.state import (
    estimate_causal_cv_state,
)


@dataclass(frozen=True)
class KalmanConfig:
    """
    CV-process EKF baseline.

    q_accel is a declared Stage3 process-noise
    modeling parameter, not a measured sensor
    quantity and not a Part-A claim.
    """

    acceleration_spectral_density_m2_s3: float = 4.0

    innovation_jitter: float = 1e-9

    min_range_m: float = 1e-6
    min_horizontal_range_m: float = 1e-6

    def __post_init__(self) -> None:
        values = (
            self.acceleration_spectral_density_m2_s3,
            self.innovation_jitter,
            self.min_range_m,
            self.min_horizontal_range_m,
        )

        if not all(
            isfinite(float(x))
            for x in values
        ):
            raise ValueError(
                "Kalman config values must "
                "be finite."
            )

        if (
            self.acceleration_spectral_density_m2_s3
            < 0.0
        ):
            raise ValueError(
                "Acceleration spectral density "
                "cannot be negative."
            )

        if self.innovation_jitter < 0.0:
            raise ValueError(
                "Innovation jitter cannot "
                "be negative."
            )

        if (
            self.min_range_m <= 0.0
            or
            self.min_horizontal_range_m <= 0.0
        ):
            raise ValueError(
                "Geometry floors must be "
                "positive."
            )


@dataclass(frozen=True)
class KalmanFrameContext:
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
                "Frame timestamps and dynamic "
                "transforms must have the same "
                "length."
            )

        previous = None

        for value in self.frame_timestamps_s:
            current = float(value)

            if not isfinite(current):
                raise ValueError(
                    "Frame timestamps must "
                    "be finite."
                )

            if (
                previous is not None
                and current <= previous
            ):
                raise ValueError(
                    "Frame timestamps must "
                    "strictly increase."
                )

            previous = current


@dataclass(frozen=True)
class KalmanState:
    """
    State order:
        [px, py, pz, vx, vy, vz] in H0.
    """

    timestamp_s: float

    mean_6: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
    ]

    covariance_6x6: tuple[
        tuple[float, ...],
        ...
    ]

    def __post_init__(self) -> None:
        if len(self.mean_6) != 6:
            raise ValueError(
                "Kalman mean must have "
                "six elements."
            )

        if not all(
            isfinite(float(x))
            for x in self.mean_6
        ):
            raise ValueError(
                "Kalman mean contains "
                "non-finite values."
            )

        if len(self.covariance_6x6) != 6:
            raise ValueError(
                "Kalman covariance must "
                "be 6x6."
            )

        if any(
            len(row) != 6
            for row in self.covariance_6x6
        ):
            raise ValueError(
                "Kalman covariance must "
                "be 6x6."
            )


@dataclass(frozen=True)
class KalmanUpdate:
    frame_index: int
    timestamp_s: float

    innovation: tuple[
        float,
        float,
        float,
        float,
    ]

    normalized_innovation_squared: float

    prior: KalmanState
    posterior: KalmanState

    measurement_covariance_used: bool = True


@dataclass(frozen=True)
class KalmanTrackResult:
    track_id: str

    initial_state: KalmanState

    updates: tuple[
        KalmanUpdate,
        ...
    ]

    current_state: KalmanState

    process_model: str = "CV"
    measurement_model: str = (
        "nonlinear_range_radial_velocity_"
        "azimuth_elevation"
    )

    truth_used: bool = False
    annotated_velocity_used: bool = False
    future_information_used: bool = False

    def __post_init__(self) -> None:
        if self.truth_used:
            raise ValueError(
                "Kalman baseline cannot "
                "use evaluator truth."
            )

        if self.annotated_velocity_used:
            raise ValueError(
                "Kalman baseline cannot use "
                "annotated velocity."
            )

        if self.future_information_used:
            raise ValueError(
                "Kalman baseline cannot use "
                "future information."
            )


def _as_np_covariance(
    covariance,
) -> np.ndarray:
    result = np.asarray(
        covariance,
        dtype=np.float64,
    )

    if result.shape != (6, 6):
        raise ValueError(
            "Expected a 6x6 covariance."
        )

    return result


def _state_from_arrays(
    *,
    timestamp_s: float,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> KalmanState:
    covariance = 0.5 * (
        covariance
        +
        covariance.T
    )

    return KalmanState(
        timestamp_s=float(
            timestamp_s
        ),
        mean_6=tuple(
            float(x)
            for x in mean
        ),  # type: ignore[arg-type]
        covariance_6x6=tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in covariance
        ),
    )


def cv_transition_matrix(
    dt_s: float,
) -> np.ndarray:
    dt = float(dt_s)

    if dt <= 0.0:
        raise ValueError(
            "Kalman dt must be positive."
        )

    F = np.eye(
        6,
        dtype=np.float64,
    )

    F[0:3, 3:6] = (
        dt
        *
        np.eye(
            3,
            dtype=np.float64,
        )
    )

    return F


def cv_process_covariance(
    dt_s: float,
    *,
    acceleration_spectral_density_m2_s3: float,
) -> np.ndarray:
    dt = float(dt_s)
    q = float(
        acceleration_spectral_density_m2_s3
    )

    if dt <= 0.0:
        raise ValueError(
            "Kalman dt must be positive."
        )

    if q < 0.0:
        raise ValueError(
            "Process spectral density "
            "cannot be negative."
        )

    I3 = np.eye(
        3,
        dtype=np.float64,
    )

    Q = np.block(
        [
            [
                (dt ** 3) / 3.0 * I3,
                (dt ** 2) / 2.0 * I3,
            ],
            [
                (dt ** 2) / 2.0 * I3,
                dt * I3,
            ],
        ]
    )

    return q * Q


def initialize_kalman_state(
    cartesian_observations,
) -> KalmanState:
    """
    Initialize from the first two causal noisy
    H0 observations using exactly the same
    backward-difference estimator as CV.

    Radial velocity is not double-counted here;
    it begins contributing in EKF updates.
    """

    if len(cartesian_observations) < 2:
        raise ValueError(
            "Kalman initialization requires "
            "two causal observations."
        )

    causal = estimate_causal_cv_state(
        tuple(
            cartesian_observations[:2]
        )
    )

    return KalmanState(
        timestamp_s=causal.timestamp_s,
        mean_6=(
            *causal.position_H0_m,
            *causal.velocity_H0_mps,
        ),
        covariance_6x6=(
            causal.covariance_6x6
        ),
    )


def predict_kalman_state(
    state: KalmanState,
    *,
    target_timestamp_s: float,
    config: KalmanConfig,
) -> KalmanState:
    dt = (
        float(target_timestamp_s)
        -
        state.timestamp_s
    )

    if dt <= 0.0:
        raise ValueError(
            "Prediction timestamp must "
            "be later than state timestamp."
        )

    F = cv_transition_matrix(
        dt
    )

    Q = cv_process_covariance(
        dt,
        acceleration_spectral_density_m2_s3=(
            config
            .acceleration_spectral_density_m2_s3
        ),
    )

    x = np.asarray(
        state.mean_6,
        dtype=np.float64,
    )

    P = _as_np_covariance(
        state.covariance_6x6
    )

    predicted_x = F @ x

    predicted_P = (
        F @ P @ F.T
        +
        Q
    )

    return _state_from_arrays(
        timestamp_s=target_timestamp_s,
        mean=predicted_x,
        covariance=predicted_P,
    )


def _measurement_geometry(
    *,
    mean_6: np.ndarray,
    frame_index: int,
    context: KalmanFrameContext,
    config: KalmanConfig,
):
    position_H0 = tuple(
        float(x)
        for x in mean_6[0:3]
    )

    velocity_H0 = tuple(
        float(x)
        for x in mean_6[3:6]
    )

    T_Ht_from_W = (
        context.transforms
        .transform_for_frame(
            frame_index
        )
    )

    position_Ht = np.asarray(
        point_H0_to_Ht(
            position_H0,
            T_Ht_from_W=T_Ht_from_W,
            T_H0_from_W=(
                context.transforms
                .T_H0_from_W
            ),
        ),
        dtype=np.float64,
    )

    actor_velocity_Ht = np.asarray(
        vector_H0_to_Ht(
            velocity_H0,
            T_Ht_from_W=T_Ht_from_W,
            T_H0_from_W=(
                context.transforms
                .T_H0_from_W
            ),
        ),
        dtype=np.float64,
    )

    sensor_velocity_h0 = (
        sensor_velocity_H0(
            frame_index=frame_index,
            frame_timestamps_s=(
                context.frame_timestamps_s
            ),
            context=context.transforms,
        )
    )

    sensor_velocity_Ht = np.asarray(
        vector_H0_to_Ht(
            sensor_velocity_h0,
            T_Ht_from_W=T_Ht_from_W,
            T_H0_from_W=(
                context.transforms
                .T_H0_from_W
            ),
        ),
        dtype=np.float64,
    )

    relative_velocity_Ht = (
        actor_velocity_Ht
        -
        sensor_velocity_Ht
    )

    rotation = np.asarray(
        rotation_Ht_from_H0(
            T_Ht_from_W=T_Ht_from_W,
            T_H0_from_W=(
                context.transforms
                .T_H0_from_W
            ),
        ),
        dtype=np.float64,
    )

    x, y, z = position_Ht

    horizontal = sqrt(
        x * x
        +
        y * y
    )

    range_m = sqrt(
        x * x
        +
        y * y
        +
        z * z
    )

    if range_m < config.min_range_m:
        raise ValueError(
            "Target range too small for "
            "spherical EKF measurement."
        )

    if (
        horizontal
        <
        config.min_horizontal_range_m
    ):
        raise ValueError(
            "Target is too close to the "
            "spherical azimuth singularity."
        )

    unit_los = (
        position_Ht
        /
        range_m
    )

    radial_velocity = float(
        unit_los
        @
        relative_velocity_Ht
    )

    azimuth = atan2(
        y,
        x,
    )

    elevation = atan2(
        z,
        horizontal,
    )

    measurement = np.asarray(
        [
            range_m,
            radial_velocity,
            azimuth,
            elevation,
        ],
        dtype=np.float64,
    )

    return (
        measurement,
        position_Ht,
        relative_velocity_Ht,
        rotation,
    )


def measurement_model_and_jacobian(
    *,
    state: KalmanState,
    frame_index: int,
    context: KalmanFrameContext,
    config: KalmanConfig,
):
    """
    Nonlinear Stage2 measurement model:

        z = [r, vr, az, el]

    Measurement is predicted in Ht while the
    EKF state remains in H0.
    """

    mean = np.asarray(
        state.mean_6,
        dtype=np.float64,
    )

    (
        measurement,
        q,
        relative_velocity,
        rotation,
    ) = _measurement_geometry(
        mean_6=mean,
        frame_index=frame_index,
        context=context,
        config=config,
    )

    x, y, z = q

    horizontal_sq = (
        x * x
        +
        y * y
    )

    horizontal = sqrt(
        horizontal_sq
    )

    range_sq = (
        horizontal_sq
        +
        z * z
    )

    range_m = sqrt(
        range_sq
    )

    unit_los = (
        q
        /
        range_m
    )

    radial_velocity = (
        measurement[1]
    )

    dr_dq = unit_los

    dvr_dq = (
        relative_velocity
        -
        radial_velocity
        * unit_los
    ) / range_m

    dvr_du = unit_los

    daz_dq = np.asarray(
        [
            -y / horizontal_sq,
            x / horizontal_sq,
            0.0,
        ],
        dtype=np.float64,
    )

    del_dq = np.asarray(
        [
            -z * x
            /
            (
                horizontal
                *
                range_sq
            ),

            -z * y
            /
            (
                horizontal
                *
                range_sq
            ),

            horizontal
            /
            range_sq,
        ],
        dtype=np.float64,
    )

    H = np.zeros(
        (4, 6),
        dtype=np.float64,
    )

    H[0, 0:3] = (
        dr_dq @ rotation
    )

    H[1, 0:3] = (
        dvr_dq @ rotation
    )

    H[1, 3:6] = (
        dvr_du @ rotation
    )

    H[2, 0:3] = (
        daz_dq @ rotation
    )

    H[3, 0:3] = (
        del_dq @ rotation
    )

    return measurement, H


def measurement_vector(
    measurement: MeasurementSnapshot,
) -> np.ndarray:
    return np.asarray(
        [
            measurement.range_m,
            measurement.radial_velocity_mps,
            measurement.azimuth_rad,
            measurement.elevation_rad,
        ],
        dtype=np.float64,
    )


def update_kalman_state(
    prior: KalmanState,
    *,
    measurement: MeasurementSnapshot,
    frame_index: int,
    context: KalmanFrameContext,
    config: KalmanConfig,
) -> KalmanUpdate:
    if abs(
        prior.timestamp_s
        -
        context.frame_timestamps_s[
            frame_index
        ]
    ) > 1e-9:
        raise ValueError(
            "Prior timestamp is not aligned "
            "with the measurement frame."
        )

    expected, H = (
        measurement_model_and_jacobian(
            state=prior,
            frame_index=frame_index,
            context=context,
            config=config,
        )
    )

    observed = measurement_vector(
        measurement
    )

    innovation = (
        observed
        -
        expected
    )

    innovation[2] = wrap_angle_rad(
        float(
            innovation[2]
        )
    )

    P = _as_np_covariance(
        prior.covariance_6x6
    )

    R = np.asarray(
        measurement.covariance_4x4,
        dtype=np.float64,
    )

    if R.shape != (4, 4):
        raise ValueError(
            "Stage2 measurement covariance "
            "must be 4x4."
        )

    S = (
        H @ P @ H.T
        +
        R
    )

    if config.innovation_jitter > 0.0:
        S = (
            S
            +
            config.innovation_jitter
            *
            np.eye(
                4,
                dtype=np.float64,
            )
        )

    PHt = (
        P @ H.T
    )

    try:
        K = np.linalg.solve(
            S.T,
            PHt.T,
        ).T

        solved_innovation = (
            np.linalg.solve(
                S,
                innovation,
            )
        )

    except np.linalg.LinAlgError:
        S_pinv = np.linalg.pinv(
            S
        )

        K = (
            PHt @ S_pinv
        )

        solved_innovation = (
            S_pinv @ innovation
        )

    x_prior = np.asarray(
        prior.mean_6,
        dtype=np.float64,
    )

    x_posterior = (
        x_prior
        +
        K @ innovation
    )

    I = np.eye(
        6,
        dtype=np.float64,
    )

    I_KH = (
        I
        -
        K @ H
    )

    # Joseph-form covariance update.
    P_posterior = (
        I_KH
        @
        P
        @
        I_KH.T
        +
        K
        @
        R
        @
        K.T
    )

    P_posterior = 0.5 * (
        P_posterior
        +
        P_posterior.T
    )

    nis = float(
        innovation
        @
        solved_innovation
    )

    posterior = _state_from_arrays(
        timestamp_s=prior.timestamp_s,
        mean=x_posterior,
        covariance=P_posterior,
    )

    return KalmanUpdate(
        frame_index=frame_index,
        timestamp_s=prior.timestamp_s,
        innovation=tuple(
            float(x)
            for x in innovation
        ),  # type: ignore[arg-type]
        normalized_innovation_squared=nis,
        prior=prior,
        posterior=posterior,
        measurement_covariance_used=True,
    )


def filter_associated_track(
    track: AssociatedTrack,
    *,
    context: KalmanFrameContext,
    config: KalmanConfig | None = None,
) -> KalmanTrackResult:
    if config is None:
        config = KalmanConfig()

    if len(track.detections) < 2:
        raise ValueError(
            "Kalman track requires at least "
            "two associated detections."
        )

    cartesian = (
        cartesianize_associated_track(
            track,
            context=context.transforms,
        )
    )

    initial = initialize_kalman_state(
        cartesian
    )

    state = initial

    updates = []

    # First two measurements initialized the
    # causal state. Starting at measurement 3,
    # run the genuine nonlinear EKF recursion.
    for associated in track.detections[2:]:
        target_timestamp = (
            associated.timestamp_s
        )

        prior = predict_kalman_state(
            state,
            target_timestamp_s=(
                target_timestamp
            ),
            config=config,
        )

        update = update_kalman_state(
            prior,
            measurement=(
                associated.detection
            ),
            frame_index=(
                associated.frame_index
            ),
            context=context,
            config=config,
        )

        updates.append(
            update
        )

        state = update.posterior

    return KalmanTrackResult(
        track_id=track.track_id,
        initial_state=initial,
        updates=tuple(updates),
        current_state=state,
        truth_used=False,
        annotated_velocity_used=False,
        future_information_used=False,
    )
