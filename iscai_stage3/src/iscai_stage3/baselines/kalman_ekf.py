from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from iscai_stage2.observations.contracts import (
    PcfmcwLikeObservation,
)


@dataclass(frozen=True)
class EKFState:
    """
    Cartesian CV state in dynamic headlamp frame Ht.

    state order:
        [x, y, z, vx, vy, vz]
    """

    timestamp_s: float

    mean: tuple[
        float, float, float,
        float, float, float,
    ]

    covariance: tuple[
        tuple[float, ...],
        ...
    ]


def _as_covariance6(
    value: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    if value.shape != (6, 6):
        raise ValueError(
            "State covariance must be 6x6."
        )

    value = 0.5 * (
        value + value.T
    )

    if not np.all(
        np.isfinite(value)
    ):
        raise ValueError(
            "State covariance must be finite."
        )

    return tuple(
        tuple(
            float(x)
            for x in row
        )
        for row in value
    )


def _wrap_pi(
    angle_rad: float,
) -> float:
    return (
        angle_rad + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def measurement_function(
    state_mean,
) -> np.ndarray:
    """
    Cartesian state -> Stage2 measurement coordinates.

    output order:
        [range, radial_velocity, azimuth, elevation]
    """

    state = np.asarray(
        state_mean,
        dtype=np.float64,
    )

    if state.shape != (6,):
        raise ValueError(
            "State mean must have shape (6,)."
        )

    x, y, z, vx, vy, vz = state

    rho2 = (
        x * x
        + y * y
    )

    r2 = (
        rho2
        + z * z
    )

    if r2 <= 1e-18:
        raise ValueError(
            "Measurement undefined at zero range."
        )

    rho = math.sqrt(rho2)
    r = math.sqrt(r2)

    if rho <= 1e-12:
        raise ValueError(
            "Azimuth undefined on vertical axis."
        )

    radial_velocity = (
        x * vx
        + y * vy
        + z * vz
    ) / r

    azimuth = math.atan2(
        y,
        x,
    )

    elevation = math.atan2(
        z,
        rho,
    )

    return np.asarray(
        (
            r,
            radial_velocity,
            azimuth,
            elevation,
        ),
        dtype=np.float64,
    )


def measurement_jacobian(
    state_mean,
) -> np.ndarray:
    """
    Analytic Jacobian H = dh/dx.

    h(x) =
        [range, radial_velocity, azimuth, elevation]
    """

    state = np.asarray(
        state_mean,
        dtype=np.float64,
    )

    if state.shape != (6,):
        raise ValueError(
            "State mean must have shape (6,)."
        )

    x, y, z, vx, vy, vz = state

    rho2 = (
        x * x
        + y * y
    )

    r2 = (
        rho2
        + z * z
    )

    if r2 <= 1e-18:
        raise ValueError(
            "Jacobian undefined at zero range."
        )

    rho = math.sqrt(rho2)
    r = math.sqrt(r2)

    if rho <= 1e-12:
        raise ValueError(
            "Jacobian undefined on vertical axis."
        )

    dot_pv = (
        x * vx
        + y * vy
        + z * vz
    )

    radial_velocity = (
        dot_pv / r
    )

    H = np.zeros(
        (4, 6),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # range
    # --------------------------------------------------------

    H[0, 0] = x / r
    H[0, 1] = y / r
    H[0, 2] = z / r

    # --------------------------------------------------------
    # radial velocity
    #
    # vr = p dot v / ||p||
    # --------------------------------------------------------

    H[1, 0] = (
        vx / r
        - radial_velocity * x / r2
    )

    H[1, 1] = (
        vy / r
        - radial_velocity * y / r2
    )

    H[1, 2] = (
        vz / r
        - radial_velocity * z / r2
    )

    H[1, 3] = x / r
    H[1, 4] = y / r
    H[1, 5] = z / r

    # --------------------------------------------------------
    # azimuth
    # --------------------------------------------------------

    H[2, 0] = -y / rho2
    H[2, 1] = x / rho2

    # --------------------------------------------------------
    # elevation
    # --------------------------------------------------------

    denominator = (
        r2 * rho
    )

    H[3, 0] = (
        -x * z / denominator
    )

    H[3, 1] = (
        -y * z / denominator
    )

    H[3, 2] = (
        rho / r2
    )

    return H


def initialize_ekf_from_measurement(
    observation: PcfmcwLikeObservation,
    *,
    tangential_velocity_variance: float = 100.0,
) -> EKFState:
    """
    Initialize from one causal Stage2 measurement.

    Only the measured radial component of velocity is known.
    Therefore the initial velocity mean is placed along LOS,
    while velocity covariance remains deliberately broad.
    """

    if not observation.measurement_valid:
        raise ValueError(
            "Cannot initialize from invalid measurement."
        )

    if observation.measured_fmcw:
        raise ValueError(
            "Measured-FMCW claim is forbidden."
        )

    r = float(
        observation.range_m
    )

    az = float(
        observation.azimuth_rad
    )

    el = float(
        observation.elevation_rad
    )

    vr = float(
        observation.radial_velocity_mps
    )

    cos_el = math.cos(el)

    direction = np.asarray(
        (
            cos_el * math.cos(az),
            cos_el * math.sin(az),
            math.sin(el),
        ),
        dtype=np.float64,
    )

    position = (
        r * direction
    )

    velocity = (
        vr * direction
    )

    mean = np.concatenate(
        (
            position,
            velocity,
        )
    )

    # Broad initial Cartesian covariance.
    # Subsequent measurement updates use the full Stage2 R_t.
    P = np.zeros(
        (6, 6),
        dtype=np.float64,
    )

    P[0:3, 0:3] = (
        np.eye(3)
        * max(
            1.0,
            observation.covariance.matrix[0][0],
        )
    )

    P[3:6, 3:6] = (
        np.eye(3)
        * tangential_velocity_variance
    )

    return EKFState(
        timestamp_s=float(
            observation.timestamp_s
        ),
        mean=tuple(
            float(x)
            for x in mean
        ),
        covariance=_as_covariance6(P),
    )


def predict_ekf_cv(
    state: EKFState,
    *,
    timestamp_s: float,
    acceleration_noise_variance: float = 1.0,
) -> EKFState:
    """
    CV Kalman prediction:

        x^- = F x
        P^- = F P F^T + Q

    Q uses a white-acceleration model.
    """

    dt = (
        float(timestamp_s)
        - state.timestamp_s
    )

    if dt <= 0.0:
        raise ValueError(
            "Prediction timestamp must increase."
        )

    if (
        not math.isfinite(
            acceleration_noise_variance
        )
        or acceleration_noise_variance < 0.0
    ):
        raise ValueError(
            "Acceleration-noise variance must be "
            "finite and non-negative."
        )

    F = np.eye(
        6,
        dtype=np.float64,
    )

    for axis in range(3):
        F[
            axis,
            axis + 3,
        ] = dt

    q = float(
        acceleration_noise_variance
    )

    Q = np.zeros(
        (6, 6),
        dtype=np.float64,
    )

    for axis in range(3):

        position_index = axis
        velocity_index = (
            axis + 3
        )

        Q[
            position_index,
            position_index,
        ] = (
            0.25
            * dt ** 4
            * q
        )

        Q[
            position_index,
            velocity_index,
        ] = (
            0.5
            * dt ** 3
            * q
        )

        Q[
            velocity_index,
            position_index,
        ] = (
            0.5
            * dt ** 3
            * q
        )

        Q[
            velocity_index,
            velocity_index,
        ] = (
            dt ** 2
            * q
        )

    x = np.asarray(
        state.mean,
        dtype=np.float64,
    )

    P = np.asarray(
        state.covariance,
        dtype=np.float64,
    )

    predicted_mean = (
        F @ x
    )

    predicted_covariance = (
        F @ P @ F.T
        + Q
    )

    return EKFState(
        timestamp_s=float(
            timestamp_s
        ),
        mean=tuple(
            float(value)
            for value
            in predicted_mean
        ),
        covariance=_as_covariance6(
            predicted_covariance
        ),
    )


def update_ekf_with_stage2_measurement(
    *,
    predicted: EKFState,
    observation: PcfmcwLikeObservation,
) -> EKFState:
    """
    EKF correction with the complete frozen Stage2 measurement:

        z = [range, radial_velocity, azimuth, elevation]

    and the complete Stage2 covariance R_t.

    No truth sidecar.
    No WOMD future state.
    """

    if not observation.measurement_valid:
        raise ValueError(
            "Cannot update with invalid measurement."
        )

    if observation.measured_fmcw:
        raise ValueError(
            "Measured-FMCW claim is forbidden."
        )

    if abs(
        observation.timestamp_s
        - predicted.timestamp_s
    ) > 1e-9:
        raise ValueError(
            "Measurement/prediction timestamp mismatch."
        )

    x = np.asarray(
        predicted.mean,
        dtype=np.float64,
    )

    P = np.asarray(
        predicted.covariance,
        dtype=np.float64,
    )

    z = np.asarray(
        observation.measurement_vector(),
        dtype=np.float64,
    )

    R = np.asarray(
        observation.covariance.matrix,
        dtype=np.float64,
    )

    if R.shape != (4, 4):
        raise ValueError(
            "Stage2 measurement covariance must be 4x4."
        )

    z_pred = measurement_function(
        x
    )

    H = measurement_jacobian(
        x
    )

    innovation = (
        z - z_pred
    )

    # Azimuth innovation must respect wrapping.
    innovation[2] = _wrap_pi(
        float(
            innovation[2]
        )
    )

    S = (
        H @ P @ H.T
        + R
    )

    try:
        # Equivalent to:
        # K = P H^T S^-1
        K = np.linalg.solve(
            S.T,
            (
                P @ H.T
            ).T,
        ).T

    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Innovation covariance is singular."
        ) from exc

    updated_mean = (
        x + K @ innovation
    )

    identity = np.eye(
        6,
        dtype=np.float64,
    )

    # Joseph stabilized covariance update.
    A = (
        identity
        - K @ H
    )

    updated_covariance = (
        A @ P @ A.T
        + K @ R @ K.T
    )

    return EKFState(
        timestamp_s=predicted.timestamp_s,
        mean=tuple(
            float(value)
            for value
            in updated_mean
        ),
        covariance=_as_covariance6(
            updated_covariance
        ),
    )
