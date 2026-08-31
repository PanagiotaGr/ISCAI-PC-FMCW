from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


PRIMARY_HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

CONFIDENCE_LEVELS = (
    0.50,
    0.80,
    0.90,
    0.95,
    0.99,
)


Vector3 = tuple[
    float,
    float,
    float,
]

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _det2(
    a: float,
    b: float,
    c: float,
    d: float,
) -> float:
    return a * d - b * c


def _det3(
    m: Matrix3,
) -> float:
    return (
        m[0][0]
        * (
            m[1][1] * m[2][2]
            -
            m[1][2] * m[2][1]
        )
        -
        m[0][1]
        * (
            m[1][0] * m[2][2]
            -
            m[1][2] * m[2][0]
        )
        +
        m[0][2]
        * (
            m[1][0] * m[2][1]
            -
            m[1][1] * m[2][0]
        )
    )


def _validate_spd3(
    covariance: Matrix3,
) -> None:
    flat = (
        covariance[0]
        +
        covariance[1]
        +
        covariance[2]
    )

    if not all(
        isfinite(float(x))
        for x in flat
    ):
        raise ValueError(
            "Covariance must be finite."
        )

    for i in range(3):
        for j in range(3):
            if abs(
                covariance[i][j]
                -
                covariance[j][i]
            ) > 1e-10:
                raise ValueError(
                    "Covariance must be symmetric."
                )

    # Sylvester criterion for symmetric 3x3 matrix.
    minor1 = covariance[0][0]

    minor2 = _det2(
        covariance[0][0],
        covariance[0][1],
        covariance[1][0],
        covariance[1][1],
    )

    minor3 = _det3(
        covariance
    )

    if not (
        minor1 > 0.0
        and
        minor2 > 0.0
        and
        minor3 > 0.0
    ):
        raise ValueError(
            "Covariance must be positive definite."
        )


@dataclass(frozen=True)
class GaussianHorizonPosterior:
    horizon_s: float
    mean_H0_m: Vector3
    covariance_H0_m2: Matrix3

    def __post_init__(self) -> None:
        if self.horizon_s not in (
            PRIMARY_HORIZONS_S
        ):
            raise ValueError(
                "Unsupported Stage4 horizon."
            )

        if not all(
            isfinite(float(x))
            for x in self.mean_H0_m
        ):
            raise ValueError(
                "Posterior mean must be finite."
            )

        _validate_spd3(
            self.covariance_H0_m2
        )


@dataclass(frozen=True)
class TrajectoryGaussianPosterior:
    prediction_id: str
    horizons: tuple[
        GaussianHorizonPosterior,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.prediction_id:
            raise ValueError(
                "prediction_id cannot be empty."
            )

        actual = tuple(
            item.horizon_s
            for item in self.horizons
        )

        if actual != PRIMARY_HORIZONS_S:
            raise ValueError(
                "Stage4 posterior must contain "
                "exactly the frozen primary horizons."
            )


@dataclass(frozen=True)
class Stage4ScopeContract:
    stage: int = 4

    main_observation_mode: str = (
        "full_frozen_stage2_degraded"
    )

    association_frontend: str = (
        "frozen_stage3_estimated_gnn"
    )

    measurement_covariance_is_input: bool = True
    measurement_uncertainty_distinct_from_predictive: bool = True

    future_states_are_model_input: bool = False
    future_validity_is_model_input: bool = False
    future_track_duration_is_model_input: bool = False
    future_lidar_is_model_input: bool = False

    future_trajectory_is_supervision_only: bool = True

    tracks_to_predict_is_model_input: bool = False
    objects_of_interest_is_model_input: bool = False

    annotated_velocity_is_realistic_input: bool = False
    perfect_track_id_is_numeric_feature: bool = False

    deterministic_predictor_required: bool = True
    gaussian_probabilistic_predictor_required: bool = True
    trajectory_calibration_required: bool = True

    multi_agent_context_required: bool = True
    map_context_ablation_required: bool = True

    gmm_implemented_and_evaluated: bool = True
    gmm_closure_critical_only_if_gaussian_inadequate: bool = True

    transformer_required: bool = False

    receiver_angular_posterior_in_stage4: bool = False
    beam_controller_in_stage4: bool = False
    optical_link_in_stage4: bool = False
    adb_in_stage4: bool = False
    deepsense_in_stage4: bool = False

    def __post_init__(self) -> None:
        if self.stage != 4:
            raise ValueError(
                "Stage4 contract has wrong stage."
            )

        if not self.measurement_covariance_is_input:
            raise ValueError(
                "R_t must be a predictor input."
            )

        if any((
            self.future_states_are_model_input,
            self.future_validity_is_model_input,
            self.future_track_duration_is_model_input,
            self.future_lidar_is_model_input,
            self.tracks_to_predict_is_model_input,
            self.objects_of_interest_is_model_input,
            self.annotated_velocity_is_realistic_input,
            self.perfect_track_id_is_numeric_feature,
        )):
            raise ValueError(
                "Forbidden realistic-model dependency."
            )

        if any((
            self.receiver_angular_posterior_in_stage4,
            self.beam_controller_in_stage4,
            self.optical_link_in_stage4,
            self.adb_in_stage4,
            self.deepsense_in_stage4,
        )):
            raise ValueError(
                "Downstream scope entered Stage4."
            )
