from __future__ import annotations

from dataclasses import dataclass
import math


from iscai_stage3.baselines.imm_core import (
    IMMProbabilities,
    imm_mixing_probabilities,
    update_imm_probabilities,
)

from iscai_stage3.baselines.imm_evaluator import (
    IMMModeMeasurement,
    evaluate_mode,
)

from iscai_stage3.baselines.mode_prediction import (
    predict_ca_measurement,
    predict_ctrv_measurement,
    predict_cv_measurement,
)


TRANSITION_MATRIX = (
    (0.90, 0.05, 0.05),
    (0.05, 0.90, 0.05),
    (0.05, 0.05, 0.90),
)


@dataclass(frozen=True)
class IMMRealStepResult:
    probabilities: IMMProbabilities

    cv_log_likelihood: float
    ca_log_likelihood: float
    ctrv_log_likelihood: float


def _stable_relative_likelihoods(
    *,
    cv_log_likelihood: float,
    ca_log_likelihood: float,
    ctrv_log_likelihood: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Convert log-likelihoods to relative likelihoods
    without exponential underflow.

    Subtracting the common maximum does not change
    the normalized Bayesian mode probabilities.
    """

    logs = (
        float(cv_log_likelihood),
        float(ca_log_likelihood),
        float(ctrv_log_likelihood),
    )

    if not all(
        math.isfinite(value)
        for value in logs
    ):
        raise ValueError(
            "IMM log-likelihoods must be finite."
        )

    maximum = max(logs)

    relative = tuple(
        math.exp(
            value - maximum
        )
        for value in logs
    )

    # At least the maximum element must be exp(0) = 1.
    if max(relative) <= 0.0:
        raise RuntimeError(
            "Stable IMM likelihood conversion failed."
        )

    return relative


def imm_real_step(
    *,
    probabilities: IMMProbabilities,

    position,
    velocity,
    acceleration,

    heading_rad: float,
    yaw_rate_radps: float,

    measurement,
    covariance,

    dt: float,
) -> IMMRealStepResult:
    """
    One causal IMM mode-probability update.

    Inputs:
        strictly past motion state
        current Stage2 measurement z_t
        current Stage2 measurement covariance R_t

    No future state.
    No evaluator truth sidecar.
    """

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            "IMM dt must be finite and positive."
        )

    # --------------------------------------------------------
    # 1. IMM Markov probability prediction
    # --------------------------------------------------------

    mixed = imm_mixing_probabilities(
        probabilities=probabilities,
        transition_matrix=TRANSITION_MATRIX,
    )

    predicted_probabilities = (
        mixed.predicted_probabilities
    )

    # --------------------------------------------------------
    # 2. Mode-specific one-step measurement prediction
    # --------------------------------------------------------

    cv_prediction = predict_cv_measurement(
        position=position,
        velocity=velocity,
        dt=dt,
    )

    ca_prediction = predict_ca_measurement(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        dt=dt,
    )

    speed_mps = math.hypot(
        velocity[0],
        velocity[1],
    )

    ctrv_prediction = predict_ctrv_measurement(
        position=position,
        speed_mps=speed_mps,
        heading_rad=heading_rad,
        yaw_rate_radps=yaw_rate_radps,
        vertical_velocity_mps=velocity[2],
        dt=dt,
    )

    # --------------------------------------------------------
    # 3. Same z_t and R_t for all modes
    # --------------------------------------------------------

    cv = evaluate_mode(
        mode=IMMModeMeasurement(
            mode_name="CV",
            predicted_measurement=cv_prediction,
        ),
        measured=measurement,
        covariance=covariance,
    )

    ca = evaluate_mode(
        mode=IMMModeMeasurement(
            mode_name="CA",
            predicted_measurement=ca_prediction,
        ),
        measured=measurement,
        covariance=covariance,
    )

    ctrv = evaluate_mode(
        mode=IMMModeMeasurement(
            mode_name="CTRV",
            predicted_measurement=ctrv_prediction,
        ),
        measured=measurement,
        covariance=covariance,
    )

    # --------------------------------------------------------
    # 4. Numerically stable likelihood conversion
    # --------------------------------------------------------

    likelihoods = _stable_relative_likelihoods(
        cv_log_likelihood=(
            cv.log_likelihood
        ),
        ca_log_likelihood=(
            ca.log_likelihood
        ),
        ctrv_log_likelihood=(
            ctrv.log_likelihood
        ),
    )

    # --------------------------------------------------------
    # 5. Bayesian IMM probability correction
    # --------------------------------------------------------

    updated = update_imm_probabilities(
        predicted_probabilities=(
            predicted_probabilities
        ),
        measurement_likelihoods=likelihoods,
    )

    return IMMRealStepResult(
        probabilities=updated,

        cv_log_likelihood=(
            cv.log_likelihood
        ),

        ca_log_likelihood=(
            ca.log_likelihood
        ),

        ctrv_log_likelihood=(
            ctrv.log_likelihood
        ),
    )
