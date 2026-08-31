from __future__ import annotations

from dataclasses import dataclass
from math import (
    atan2,
    cos,
    isfinite,
    log,
    pi,
    sin,
    sqrt,
)

import numpy as np

from iscai_stage3.association import (
    AssociatedTrack,
)

from iscai_stage3.filters.kalman_ekf import (
    KalmanConfig,
    KalmanFrameContext,
    KalmanState,
    cv_process_covariance,
    measurement_model_and_jacobian,
    measurement_vector,
)

from iscai_stage3.geometry import (
    cartesianize_associated_detection,
)

from iscai_stage3.observations import (
    MeasurementSnapshot,
    wrap_angle_rad,
)

from iscai_stage3.state import (
    estimate_causal_ca_state,
    estimate_causal_ctrv_state,
    estimate_causal_cv_state,
)


MODE_NAMES = (
    "CV",
    "CA",
    "CTRV",
)


@dataclass(frozen=True)
class IMMConfig:
    """
    Full interacting multiple-model filter.

    Transition matrix convention:

        transition_matrix[i][j]
        =
        P(mode_j at k | mode_i at k-1)

    All process-noise parameters are declared
    Stage3 baseline assumptions, not measured
    sensor facts.
    """

    transition_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (0.90, 0.05, 0.05),
        (0.05, 0.90, 0.05),
        (0.05, 0.05, 0.90),
    )

    initial_mode_probabilities: tuple[
        float,
        float,
        float,
    ] = (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    )

    cv_acceleration_spectral_density_m2_s3: float = 4.0

    ca_jerk_spectral_density_m2_s5: float = 4.0

    ctrv_linear_acceleration_spectral_density_m2_s3: float = 4.0

    ctrv_turn_rate_random_walk_rad2_s3: float = 0.05

    mode_switch_acceleration_variance_m2_s4: float = 9.0

    mode_switch_turn_rate_variance_rad2_s2: float = 0.09

    initial_acceleration_variance_m2_s4: float = 9.0

    initial_turn_rate_variance_rad2_s2: float = 0.09

    innovation_jitter: float = 1e-9

    ctrv_turn_rate_epsilon_radps: float = 1e-6

    prediction_step_s: float = 0.1

    def __post_init__(self) -> None:
        transition = np.asarray(
            self.transition_matrix,
            dtype=np.float64,
        )

        if transition.shape != (3, 3):
            raise ValueError(
                "IMM transition matrix must "
                "be 3x3."
            )

        if not np.all(
            np.isfinite(transition)
        ):
            raise ValueError(
                "IMM transition matrix must "
                "be finite."
            )

        if np.any(transition < 0.0):
            raise ValueError(
                "IMM transition probabilities "
                "cannot be negative."
            )

        if not np.allclose(
            transition.sum(axis=1),
            1.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Each IMM transition-matrix "
                "row must sum to one."
            )

        probabilities = np.asarray(
            self.initial_mode_probabilities,
            dtype=np.float64,
        )

        if probabilities.shape != (3,):
            raise ValueError(
                "IMM needs three initial mode "
                "probabilities."
            )

        if np.any(probabilities < 0.0):
            raise ValueError(
                "Initial mode probabilities "
                "cannot be negative."
            )

        if not np.isclose(
            probabilities.sum(),
            1.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Initial mode probabilities "
                "must sum to one."
            )

        numeric = (
            self.cv_acceleration_spectral_density_m2_s3,
            self.ca_jerk_spectral_density_m2_s5,
            self.ctrv_linear_acceleration_spectral_density_m2_s3,
            self.ctrv_turn_rate_random_walk_rad2_s3,
            self.mode_switch_acceleration_variance_m2_s4,
            self.mode_switch_turn_rate_variance_rad2_s2,
            self.initial_acceleration_variance_m2_s4,
            self.initial_turn_rate_variance_rad2_s2,
            self.innovation_jitter,
            self.ctrv_turn_rate_epsilon_radps,
            self.prediction_step_s,
        )

        if not all(
            isfinite(float(x))
            for x in numeric
        ):
            raise ValueError(
                "IMM numeric configuration "
                "must be finite."
            )

        if any(
            float(x) < 0.0
            for x in numeric[:-2]
        ):
            raise ValueError(
                "IMM process variances/noises "
                "cannot be negative."
            )

        if self.innovation_jitter < 0.0:
            raise ValueError(
                "Innovation jitter cannot "
                "be negative."
            )

        if (
            self.ctrv_turn_rate_epsilon_radps
            <= 0.0
        ):
            raise ValueError(
                "CTRV epsilon must be positive."
            )

        if self.prediction_step_s <= 0.0:
            raise ValueError(
                "IMM prediction step must "
                "be positive."
            )


@dataclass(frozen=True)
class IMMModeState:
    mode: str

    timestamp_s: float

    mean_10: tuple[
        float,
        ...,
    ]

    covariance_10x10: tuple[
        tuple[float, ...],
        ...
    ]

    def __post_init__(self) -> None:
        if self.mode not in MODE_NAMES:
            raise ValueError(
                f"Unknown IMM mode {self.mode!r}."
            )

        if len(self.mean_10) != 10:
            raise ValueError(
                "IMM mode mean must contain "
                "10 values."
            )

        if len(
            self.covariance_10x10
        ) != 10:
            raise ValueError(
                "IMM covariance must be 10x10."
            )

        if any(
            len(row) != 10
            for row in self.covariance_10x10
        ):
            raise ValueError(
                "IMM covariance must be 10x10."
            )

        values = (
            self.timestamp_s,
            *self.mean_10,
        )

        if not all(
            isfinite(float(x))
            for x in values
        ):
            raise ValueError(
                "IMM state contains "
                "non-finite values."
            )

        P = np.asarray(
            self.covariance_10x10,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(P)
        ):
            raise ValueError(
                "IMM covariance contains "
                "non-finite values."
            )

        if not np.allclose(
            P,
            P.T,
            atol=1e-8,
        ):
            raise ValueError(
                "IMM covariance must "
                "be symmetric."
            )

        if np.min(
            np.diag(P)
        ) < -1e-10:
            raise ValueError(
                "IMM covariance contains "
                "negative diagonal."
            )


@dataclass(frozen=True)
class IMMCombinedState:
    timestamp_s: float

    mean_10: tuple[
        float,
        ...,
    ]

    covariance_10x10: tuple[
        tuple[float, ...],
        ...
    ]

    mode_probabilities: tuple[
        float,
        float,
        float,
    ]


@dataclass(frozen=True)
class IMMModeUpdate:
    mode: str

    prior: IMMModeState
    posterior: IMMModeState

    innovation: tuple[
        float,
        float,
        float,
        float,
    ]

    normalized_innovation_squared: float

    log_likelihood: float

    measurement_covariance_used: bool = True


@dataclass(frozen=True)
class IMMUpdateStep:
    frame_index: int
    timestamp_s: float

    prior_mode_probabilities: tuple[
        float,
        float,
        float,
    ]

    predicted_mode_probabilities: tuple[
        float,
        float,
        float,
    ]

    # Matrix [source_mode][destination_mode].
    mixing_probabilities: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    mode_updates: tuple[
        IMMModeUpdate,
        IMMModeUpdate,
        IMMModeUpdate,
    ]

    posterior_mode_probabilities: tuple[
        float,
        float,
        float,
    ]

    combined_posterior: IMMCombinedState


@dataclass(frozen=True)
class IMMTrackResult:
    track_id: str

    initial_mode_states: tuple[
        IMMModeState,
        IMMModeState,
        IMMModeState,
    ]

    initial_mode_probabilities: tuple[
        float,
        float,
        float,
    ]

    updates: tuple[
        IMMUpdateStep,
        ...
    ]

    current_mode_states: tuple[
        IMMModeState,
        IMMModeState,
        IMMModeState,
    ]

    current_mode_probabilities: tuple[
        float,
        float,
        float,
    ]

    combined_current_state: IMMCombinedState

    truth_used: bool = False
    annotated_velocity_used: bool = False
    annotated_heading_used: bool = False
    future_information_used: bool = False

    def __post_init__(self) -> None:
        if (
            self.truth_used
            or
            self.annotated_velocity_used
            or
            self.annotated_heading_used
            or
            self.future_information_used
        ):
            raise ValueError(
                "IMM result contains prohibited "
                "information."
            )


def _state_from_arrays(
    *,
    mode: str,
    timestamp_s: float,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> IMMModeState:
    covariance = 0.5 * (
        covariance
        +
        covariance.T
    )

    return IMMModeState(
        mode=mode,
        timestamp_s=float(
            timestamp_s
        ),
        mean_10=tuple(
            float(x)
            for x in mean
        ),
        covariance_10x10=tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in covariance
        ),
    )


def _covariance_10(
    state: IMMModeState,
) -> np.ndarray:
    return np.asarray(
        state.covariance_10x10,
        dtype=np.float64,
    )


def _mean_10(
    state: IMMModeState,
) -> np.ndarray:
    return np.asarray(
        state.mean_10,
        dtype=np.float64,
    )


def _embed_covariance(
    *,
    base: np.ndarray,
    acceleration_variance: float,
    turn_rate_variance: float,
) -> np.ndarray:
    P = np.zeros(
        (10, 10),
        dtype=np.float64,
    )

    rows, cols = base.shape

    P[:rows, :cols] = base

    if rows < 9:
        P[6:9, 6:9] = (
            acceleration_variance
            *
            np.eye(
                3,
                dtype=np.float64,
            )
        )

    P[9, 9] = (
        turn_rate_variance
    )

    return P


def initialize_imm_mode_states(
    cartesian_observations,
    *,
    config: IMMConfig,
) -> tuple[
    IMMModeState,
    IMMModeState,
    IMMModeState,
]:
    if len(
        cartesian_observations
    ) < 3:
        raise ValueError(
            "IMM initialization requires "
            "three causal observations."
        )

    first_three = tuple(
        cartesian_observations[:3]
    )

    cv = estimate_causal_cv_state(
        first_three
    )

    ca = estimate_causal_ca_state(
        first_three
    )

    ctrv = estimate_causal_ctrv_state(
        first_three
    )

    # CV mode.
    cv_mean = np.zeros(
        10,
        dtype=np.float64,
    )

    cv_mean[0:3] = (
        cv.position_H0_m
    )

    cv_mean[3:6] = (
        cv.velocity_H0_mps
    )

    cv_P = _embed_covariance(
        base=np.asarray(
            cv.covariance_6x6,
            dtype=np.float64,
        ),
        acceleration_variance=(
            config
            .initial_acceleration_variance_m2_s4
        ),
        turn_rate_variance=(
            config
            .initial_turn_rate_variance_rad2_s2
        ),
    )

    # CA mode.
    ca_mean = np.zeros(
        10,
        dtype=np.float64,
    )

    ca_mean[0:3] = (
        ca.position_H0_m
    )

    ca_mean[3:6] = (
        ca.velocity_H0_mps
    )

    ca_mean[6:9] = (
        ca.acceleration_H0_mps2
    )

    ca_P = _embed_covariance(
        base=np.asarray(
            ca.covariance_9x9,
            dtype=np.float64,
        ),
        acceleration_variance=(
            config
            .initial_acceleration_variance_m2_s4
        ),
        turn_rate_variance=(
            config
            .initial_turn_rate_variance_rad2_s2
        ),
    )

    # CTRV mode.
    ctrv_mean = np.zeros(
        10,
        dtype=np.float64,
    )

    ctrv_mean[0:3] = (
        ctrv.position_H0_m
    )

    ctrv_mean[3] = (
        ctrv.planar_speed_mps
        *
        cos(
            ctrv.heading_rad
        )
    )

    ctrv_mean[4] = (
        ctrv.planar_speed_mps
        *
        sin(
            ctrv.heading_rad
        )
    )

    ctrv_mean[5] = (
        ctrv.vertical_velocity_mps
    )

    ctrv_mean[9] = (
        ctrv.turn_rate_radps
    )

    ctrv_P = _embed_covariance(
        base=np.asarray(
            cv.covariance_6x6,
            dtype=np.float64,
        ),
        acceleration_variance=(
            config
            .initial_acceleration_variance_m2_s4
        ),
        turn_rate_variance=(
            config
            .initial_turn_rate_variance_rad2_s2
        ),
    )

    timestamp = (
        first_three[-1]
        .timestamp_s
    )

    return (
        _state_from_arrays(
            mode="CV",
            timestamp_s=timestamp,
            mean=cv_mean,
            covariance=cv_P,
        ),
        _state_from_arrays(
            mode="CA",
            timestamp_s=timestamp,
            mean=ca_mean,
            covariance=ca_P,
        ),
        _state_from_arrays(
            mode="CTRV",
            timestamp_s=timestamp,
            mean=ctrv_mean,
            covariance=ctrv_P,
        ),
    )


def interact_mode_states(
    states: tuple[
        IMMModeState,
        IMMModeState,
        IMMModeState,
    ],
    mode_probabilities: tuple[
        float,
        float,
        float,
    ],
    *,
    config: IMMConfig,
):
    if tuple(
        state.mode
        for state in states
    ) != MODE_NAMES:
        raise ValueError(
            "IMM states must be ordered "
            "CV / CA / CTRV."
        )

    timestamps = {
        state.timestamp_s
        for state in states
    }

    if len(timestamps) != 1:
        raise ValueError(
            "IMM mode states must be "
            "time-aligned."
        )

    probabilities = np.asarray(
        mode_probabilities,
        dtype=np.float64,
    )

    if (
        probabilities.shape != (3,)
        or
        np.any(probabilities < 0.0)
        or
        not np.isclose(
            probabilities.sum(),
            1.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Invalid IMM mode probabilities."
        )

    transition = np.asarray(
        config.transition_matrix,
        dtype=np.float64,
    )

    # c_j = sum_i mu_i * p_ij
    predicted_probabilities = (
        probabilities
        @
        transition
    )

    if np.any(
        predicted_probabilities <= 0.0
    ):
        raise ValueError(
            "IMM predicted mode probability "
            "must be positive."
        )

    mixing = (
        probabilities[:, None]
        *
        transition
        /
        predicted_probabilities[
            None,
            :
        ]
    )

    mixed_states = []

    means = tuple(
        _mean_10(state)
        for state in states
    )

    covariances = tuple(
        _covariance_10(state)
        for state in states
    )

    timestamp = states[0].timestamp_s

    for destination in range(3):
        weights = mixing[
            :,
            destination
        ]

        mixed_mean = sum(
            weights[source]
            *
            means[source]
            for source in range(3)
        )

        mixed_covariance = np.zeros(
            (10, 10),
            dtype=np.float64,
        )

        for source in range(3):
            delta = (
                means[source]
                -
                mixed_mean
            )

            mixed_covariance += (
                weights[source]
                *
                (
                    covariances[source]
                    +
                    np.outer(
                        delta,
                        delta,
                    )
                )
            )

        mixed_states.append(
            _state_from_arrays(
                mode=MODE_NAMES[
                    destination
                ],
                timestamp_s=timestamp,
                mean=mixed_mean,
                covariance=(
                    mixed_covariance
                ),
            )
        )

    return (
        tuple(mixed_states),
        tuple(
            float(x)
            for x in predicted_probabilities
        ),
        tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in mixing
        ),
    )


def _cv_mode_mean(
    mean: np.ndarray,
    dt: float,
) -> np.ndarray:
    result = mean.copy()

    result[0:3] = (
        mean[0:3]
        +
        dt
        *
        mean[3:6]
    )

    result[3:6] = (
        mean[3:6]
    )

    result[6:9] = 0.0
    result[9] = 0.0

    return result


def _ca_mode_mean(
    mean: np.ndarray,
    dt: float,
) -> np.ndarray:
    result = mean.copy()

    result[0:3] = (
        mean[0:3]
        +
        dt
        *
        mean[3:6]
        +
        0.5
        *
        dt
        *
        dt
        *
        mean[6:9]
    )

    result[3:6] = (
        mean[3:6]
        +
        dt
        *
        mean[6:9]
    )

    result[6:9] = (
        mean[6:9]
    )

    result[9] = 0.0

    return result


def _ctrv_mode_mean(
    mean: np.ndarray,
    dt: float,
    *,
    epsilon: float,
) -> np.ndarray:
    result = mean.copy()

    x, y, z = (
        mean[0],
        mean[1],
        mean[2],
    )

    vx, vy, vz = (
        mean[3],
        mean[4],
        mean[5],
    )

    omega = float(
        mean[9]
    )

    speed = sqrt(
        vx * vx
        +
        vy * vy
    )

    if speed <= 1e-12:
        result[0] = x
        result[1] = y
        result[3] = 0.0
        result[4] = 0.0

    else:
        heading = atan2(
            vy,
            vx,
        )

        if abs(omega) <= epsilon:
            result[0] = (
                x
                +
                vx * dt
            )

            result[1] = (
                y
                +
                vy * dt
            )

            result[3] = vx
            result[4] = vy

        else:
            future_heading = (
                heading
                +
                omega
                *
                dt
            )

            result[0] = (
                x
                +
                speed
                /
                omega
                *
                (
                    sin(
                        future_heading
                    )
                    -
                    sin(heading)
                )
            )

            result[1] = (
                y
                +
                speed
                /
                omega
                *
                (
                    -cos(
                        future_heading
                    )
                    +
                    cos(heading)
                )
            )

            result[3] = (
                speed
                *
                cos(
                    future_heading
                )
            )

            result[4] = (
                speed
                *
                sin(
                    future_heading
                )
            )

    result[2] = (
        z
        +
        vz
        *
        dt
    )

    result[5] = vz

    result[6:9] = 0.0

    result[9] = omega

    return result


def _cv_jacobian(
    dt: float,
) -> np.ndarray:
    F = np.zeros(
        (10, 10),
        dtype=np.float64,
    )

    F[0:3, 0:3] = np.eye(3)
    F[0:3, 3:6] = dt * np.eye(3)
    F[3:6, 3:6] = np.eye(3)

    return F


def _ca_jacobian(
    dt: float,
) -> np.ndarray:
    F = np.zeros(
        (10, 10),
        dtype=np.float64,
    )

    I = np.eye(
        3,
        dtype=np.float64,
    )

    F[0:3, 0:3] = I
    F[0:3, 3:6] = dt * I
    F[0:3, 6:9] = (
        0.5
        *
        dt
        *
        dt
        *
        I
    )

    F[3:6, 3:6] = I
    F[3:6, 6:9] = dt * I

    F[6:9, 6:9] = I

    return F


def _numeric_jacobian(
    function,
    mean: np.ndarray,
) -> np.ndarray:
    F = np.zeros(
        (10, 10),
        dtype=np.float64,
    )

    for index in range(10):
        step = (
            1e-6
            *
            max(
                1.0,
                abs(float(
                    mean[index]
                )),
            )
        )

        plus = mean.copy()
        minus = mean.copy()

        plus[index] += step
        minus[index] -= step

        F[:, index] = (
            function(plus)
            -
            function(minus)
        ) / (
            2.0 * step
        )

    return F


def _ca_process_covariance(
    dt: float,
    *,
    jerk_spectral_density: float,
) -> np.ndarray:
    q = float(
        jerk_spectral_density
    )

    Q = np.zeros(
        (10, 10),
        dtype=np.float64,
    )

    block = np.asarray(
        [
            [
                dt ** 5 / 20.0,
                dt ** 4 / 8.0,
                dt ** 3 / 6.0,
            ],
            [
                dt ** 4 / 8.0,
                dt ** 3 / 3.0,
                dt ** 2 / 2.0,
            ],
            [
                dt ** 3 / 6.0,
                dt ** 2 / 2.0,
                dt,
            ],
        ],
        dtype=np.float64,
    )

    for axis in range(3):
        indices = (
            axis,
            axis + 3,
            axis + 6,
        )

        for i in range(3):
            for j in range(3):
                Q[
                    indices[i],
                    indices[j],
                ] = (
                    q
                    *
                    block[i, j]
                )

    return Q


def _mode_process_covariance(
    mode: str,
    dt: float,
    *,
    config: IMMConfig,
) -> np.ndarray:
    Q = np.zeros(
        (10, 10),
        dtype=np.float64,
    )

    scale = (
        dt
        /
        config.prediction_step_s
    )

    if mode == "CV":
        Q[0:6, 0:6] = (
            cv_process_covariance(
                dt,
                acceleration_spectral_density_m2_s3=(
                    config
                    .cv_acceleration_spectral_density_m2_s3
                ),
            )
        )

        Q[6:9, 6:9] = (
            config
            .mode_switch_acceleration_variance_m2_s4
            *
            scale
            *
            np.eye(3)
        )

        Q[9, 9] = (
            config
            .mode_switch_turn_rate_variance_rad2_s2
            *
            scale
        )

    elif mode == "CA":
        Q += _ca_process_covariance(
            dt,
            jerk_spectral_density=(
                config
                .ca_jerk_spectral_density_m2_s5
            ),
        )

        Q[9, 9] = (
            config
            .mode_switch_turn_rate_variance_rad2_s2
            *
            scale
        )

    elif mode == "CTRV":
        Q[0:6, 0:6] = (
            cv_process_covariance(
                dt,
                acceleration_spectral_density_m2_s3=(
                    config
                    .ctrv_linear_acceleration_spectral_density_m2_s3
                ),
            )
        )

        Q[6:9, 6:9] = (
            config
            .mode_switch_acceleration_variance_m2_s4
            *
            scale
            *
            np.eye(3)
        )

        Q[9, 9] = (
            config
            .ctrv_turn_rate_random_walk_rad2_s3
            *
            dt
        )

    else:
        raise ValueError(
            f"Unknown IMM mode {mode!r}."
        )

    return Q


def predict_imm_mode_state(
    state: IMMModeState,
    *,
    target_timestamp_s: float,
    config: IMMConfig,
) -> IMMModeState:
    dt = (
        float(target_timestamp_s)
        -
        state.timestamp_s
    )

    if dt <= 0.0:
        raise ValueError(
            "IMM prediction timestamp "
            "must increase."
        )

    mean = _mean_10(
        state
    )

    P = _covariance_10(
        state
    )

    if state.mode == "CV":
        predicted_mean = (
            _cv_mode_mean(
                mean,
                dt,
            )
        )

        F = _cv_jacobian(
            dt
        )

    elif state.mode == "CA":
        predicted_mean = (
            _ca_mode_mean(
                mean,
                dt,
            )
        )

        F = _ca_jacobian(
            dt
        )

    elif state.mode == "CTRV":
        def propagate(value):
            return _ctrv_mode_mean(
                value,
                dt,
                epsilon=(
                    config
                    .ctrv_turn_rate_epsilon_radps
                ),
            )

        predicted_mean = propagate(
            mean
        )

        F = _numeric_jacobian(
            propagate,
            mean,
        )

    else:
        raise ValueError(
            f"Unknown IMM mode {state.mode!r}."
        )

    Q = _mode_process_covariance(
        state.mode,
        dt,
        config=config,
    )

    predicted_P = (
        F
        @
        P
        @
        F.T
        +
        Q
    )

    return _state_from_arrays(
        mode=state.mode,
        timestamp_s=target_timestamp_s,
        mean=predicted_mean,
        covariance=predicted_P,
    )


def update_imm_mode_state(
    prior: IMMModeState,
    *,
    measurement: MeasurementSnapshot,
    frame_index: int,
    context: KalmanFrameContext,
    config: IMMConfig,
) -> IMMModeUpdate:
    """
    All IMM modes use the exact same Stage2
    nonlinear measurement model and full R_t.
    """

    first_six_covariance = (
        _covariance_10(prior)[
            0:6,
            0:6,
        ]
    )

    kalman_view = KalmanState(
        timestamp_s=prior.timestamp_s,
        mean_6=tuple(
            prior.mean_10[0:6]
        ),  # type: ignore[arg-type]
        covariance_6x6=tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in first_six_covariance
        ),
    )

    kalman_config = KalmanConfig(
        acceleration_spectral_density_m2_s3=0.0,
        innovation_jitter=(
            config.innovation_jitter
        ),
    )

    expected, H6 = (
        measurement_model_and_jacobian(
            state=kalman_view,
            frame_index=frame_index,
            context=context,
            config=kalman_config,
        )
    )

    H = np.zeros(
        (4, 10),
        dtype=np.float64,
    )

    H[:, 0:6] = H6

    observed = measurement_vector(
        measurement
    )

    innovation = (
        observed
        -
        expected
    )

    innovation[2] = (
        wrap_angle_rad(
            float(
                innovation[2]
            )
        )
    )

    P = _covariance_10(
        prior
    )

    R = np.asarray(
        measurement.covariance_4x4,
        dtype=np.float64,
    )

    if R.shape != (4, 4):
        raise ValueError(
            "Stage2 R_t must be 4x4."
        )

    S = (
        H
        @
        P
        @
        H.T
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
        P
        @
        H.T
    )

    try:
        solved_PHt = np.linalg.solve(
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

        solved_PHt = (
            PHt
            @
            S_pinv
        )

        solved_innovation = (
            S_pinv
            @
            innovation
        )

    K = solved_PHt

    posterior_mean = (
        _mean_10(prior)
        +
        K
        @
        innovation
    )

    I = np.eye(
        10,
        dtype=np.float64,
    )

    I_KH = (
        I
        -
        K
        @
        H
    )

    posterior_P = (
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

    posterior_P = 0.5 * (
        posterior_P
        +
        posterior_P.T
    )

    nis = float(
        innovation
        @
        solved_innovation
    )

    sign, logdet = np.linalg.slogdet(
        S
    )

    if sign <= 0.0:
        eigenvalues = np.linalg.eigvalsh(
            0.5
            *
            (
                S
                +
                S.T
            )
        )

        eigenvalues = np.maximum(
            eigenvalues,
            1e-15,
        )

        logdet = float(
            np.log(
                eigenvalues
            ).sum()
        )

    log_likelihood = (
        -0.5
        *
        (
            nis
            +
            float(logdet)
            +
            4.0
            *
            log(
                2.0
                *
                pi
            )
        )
    )

    posterior = _state_from_arrays(
        mode=prior.mode,
        timestamp_s=prior.timestamp_s,
        mean=posterior_mean,
        covariance=posterior_P,
    )

    return IMMModeUpdate(
        mode=prior.mode,
        prior=prior,
        posterior=posterior,
        innovation=tuple(
            float(x)
            for x in innovation
        ),  # type: ignore[arg-type]
        normalized_innovation_squared=nis,
        log_likelihood=(
            log_likelihood
        ),
        measurement_covariance_used=True,
    )


def combine_imm_states(
    states: tuple[
        IMMModeState,
        IMMModeState,
        IMMModeState,
    ],
    mode_probabilities: tuple[
        float,
        float,
        float,
    ],
) -> IMMCombinedState:
    probabilities = np.asarray(
        mode_probabilities,
        dtype=np.float64,
    )

    if (
        probabilities.shape != (3,)
        or
        np.any(probabilities < 0.0)
        or
        not np.isclose(
            probabilities.sum(),
            1.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Invalid IMM probabilities."
        )

    timestamps = {
        state.timestamp_s
        for state in states
    }

    if len(timestamps) != 1:
        raise ValueError(
            "IMM states must be time-aligned."
        )

    means = tuple(
        _mean_10(state)
        for state in states
    )

    covariances = tuple(
        _covariance_10(state)
        for state in states
    )

    combined_mean = sum(
        probabilities[i]
        *
        means[i]
        for i in range(3)
    )

    combined_covariance = np.zeros(
        (10, 10),
        dtype=np.float64,
    )

    for i in range(3):
        delta = (
            means[i]
            -
            combined_mean
        )

        combined_covariance += (
            probabilities[i]
            *
            (
                covariances[i]
                +
                np.outer(
                    delta,
                    delta,
                )
            )
        )

    combined_covariance = 0.5 * (
        combined_covariance
        +
        combined_covariance.T
    )

    return IMMCombinedState(
        timestamp_s=states[0].timestamp_s,
        mean_10=tuple(
            float(x)
            for x in combined_mean
        ),
        covariance_10x10=tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in combined_covariance
        ),
        mode_probabilities=tuple(
            float(x)
            for x in probabilities
        ),  # type: ignore[arg-type]
    )


def imm_update_step(
    states: tuple[
        IMMModeState,
        IMMModeState,
        IMMModeState,
    ],
    mode_probabilities: tuple[
        float,
        float,
        float,
    ],
    *,
    measurement: MeasurementSnapshot,
    frame_index: int,
    context: KalmanFrameContext,
    target_timestamp_s: float,
    config: IMMConfig,
) -> IMMUpdateStep:
    (
        mixed_states,
        predicted_probabilities,
        mixing_probabilities,
    ) = interact_mode_states(
        states,
        mode_probabilities,
        config=config,
    )

    predicted_states = tuple(
        predict_imm_mode_state(
            state,
            target_timestamp_s=(
                target_timestamp_s
            ),
            config=config,
        )
        for state in mixed_states
    )

    mode_updates = tuple(
        update_imm_mode_state(
            state,
            measurement=measurement,
            frame_index=frame_index,
            context=context,
            config=config,
        )
        for state in predicted_states
    )

    log_weights = np.asarray(
        [
            log(
                max(
                    predicted_probabilities[i],
                    1e-300,
                )
            )
            +
            mode_updates[
                i
            ].log_likelihood
            for i in range(3)
        ],
        dtype=np.float64,
    )

    maximum = float(
        np.max(
            log_weights
        )
    )

    weights = np.exp(
        log_weights
        -
        maximum
    )

    normalization = float(
        weights.sum()
    )

    if (
        not isfinite(normalization)
        or
        normalization <= 0.0
    ):
        raise ValueError(
            "IMM likelihood normalization "
            "failed."
        )

    posterior_probabilities_array = (
        weights
        /
        normalization
    )

    posterior_probabilities = tuple(
        float(x)
        for x in posterior_probabilities_array
    )

    posterior_states = tuple(
        update.posterior
        for update in mode_updates
    )

    combined = combine_imm_states(
        posterior_states,
        posterior_probabilities,
    )

    return IMMUpdateStep(
        frame_index=frame_index,
        timestamp_s=target_timestamp_s,
        prior_mode_probabilities=(
            mode_probabilities
        ),
        predicted_mode_probabilities=(
            predicted_probabilities
        ),
        mixing_probabilities=(
            mixing_probabilities
        ),
        mode_updates=mode_updates,  # type: ignore[arg-type]
        posterior_mode_probabilities=(
            posterior_probabilities
        ),  # type: ignore[arg-type]
        combined_posterior=combined,
    )


def filter_associated_track_imm(
    track: AssociatedTrack,
    *,
    context: KalmanFrameContext,
    config: IMMConfig | None = None,
) -> IMMTrackResult:
    if config is None:
        config = IMMConfig()

    if len(track.detections) < 3:
        raise ValueError(
            "IMM requires at least three "
            "associated detections."
        )

    initial_cartesian = tuple(
        cartesianize_associated_detection(
            item,
            context=context.transforms,
        )
        for item in track.detections[:3]
    )

    initial_states = (
        initialize_imm_mode_states(
            initial_cartesian,
            config=config,
        )
    )

    probabilities = (
        config.initial_mode_probabilities
    )

    states = initial_states

    updates = []

    for associated in track.detections[3:]:
        step = imm_update_step(
            states,
            probabilities,
            measurement=(
                associated.detection
            ),
            frame_index=(
                associated.frame_index
            ),
            context=context,
            target_timestamp_s=(
                associated.timestamp_s
            ),
            config=config,
        )

        updates.append(
            step
        )

        states = tuple(
            update.posterior
            for update in step.mode_updates
        )  # type: ignore[assignment]

        probabilities = (
            step
            .posterior_mode_probabilities
        )

    combined = combine_imm_states(
        states,
        probabilities,
    )

    return IMMTrackResult(
        track_id=track.track_id,
        initial_mode_states=initial_states,
        initial_mode_probabilities=(
            config.initial_mode_probabilities
        ),
        updates=tuple(updates),
        current_mode_states=states,
        current_mode_probabilities=(
            probabilities
        ),
        combined_current_state=combined,
        truth_used=False,
        annotated_velocity_used=False,
        annotated_heading_used=False,
        future_information_used=False,
    )
