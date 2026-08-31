from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from iscai_stage3.geometry import (
    CartesianObservation,
)


Vec3 = tuple[
    float,
    float,
    float,
]

Matrix6 = tuple[
    tuple[float, ...],
    ...
]


@dataclass(frozen=True)
class CausalCVState:
    """
    Causal current Cartesian state in H0.

    position:
        latest noisy Stage2-derived position.

    velocity:
        strict backward difference of the last
        two noisy positions in the SAME H0 frame.

    No annotated WOMD velocity.
    No future states.
    """

    track_id: str

    timestamp_s: float

    position_H0_m: Vec3

    velocity_H0_mps: Vec3

    covariance_6x6: Matrix6

    source_frame_indices: tuple[
        int,
        int,
    ]

    annotated_velocity_used: bool = False
    future_information_used: bool = False

    def __post_init__(self) -> None:
        if self.annotated_velocity_used:
            raise ValueError(
                "Annotated velocity cannot be "
                "used in realistic CV state."
            )

        if self.future_information_used:
            raise ValueError(
                "Future information cannot enter "
                "the causal state."
            )

        values = (
            *self.position_H0_m,
            *self.velocity_H0_mps,
            self.timestamp_s,
        )

        if not all(
            isfinite(float(x))
            for x in values
        ):
            raise ValueError(
                "State contains non-finite values."
            )

        if len(self.covariance_6x6) != 6:
            raise ValueError(
                "State covariance must be 6x6."
            )

        if any(
            len(row) != 6
            for row in self.covariance_6x6
        ):
            raise ValueError(
                "State covariance must be 6x6."
            )

        for i in range(6):
            if (
                self.covariance_6x6[i][i]
                < -1e-12
            ):
                raise ValueError(
                    "State covariance has "
                    "negative diagonal."
                )

            for j in range(6):
                if abs(
                    self.covariance_6x6[i][j]
                    -
                    self.covariance_6x6[j][i]
                ) > 1e-8:
                    raise ValueError(
                        "State covariance "
                        "must be symmetric."
                    )


def _add3(a, b):
    return tuple(
        tuple(
            a[i][j] + b[i][j]
            for j in range(3)
        )
        for i in range(3)
    )


def _scale3(a, scale):
    return tuple(
        tuple(
            a[i][j] * scale
            for j in range(3)
        )
        for i in range(3)
    )


def estimate_causal_cv_state(
    observations: tuple[
        CartesianObservation,
        ...
    ],
) -> CausalCVState:
    if len(observations) < 2:
        raise ValueError(
            "CV state requires at least two "
            "causal observations."
        )

    previous = observations[-2]
    current = observations[-1]

    if previous.track_id != current.track_id:
        raise ValueError(
            "Causal state observations must "
            "belong to one Stage3 track."
        )

    dt = (
        current.timestamp_s
        -
        previous.timestamp_s
    )

    if dt <= 0.0:
        raise ValueError(
            "State timestamps must increase."
        )

    position = (
        current.position_H0_m
    )

    velocity = tuple(
        (
            current.position_H0_m[i]
            -
            previous.position_H0_m[i]
        )
        / dt
        for i in range(3)
    )

    P_current = (
        current.position_covariance_H0_m2
    )

    P_previous = (
        previous.position_covariance_H0_m2
    )

    # State:
    #
    # x = [p_t, (p_t - p_{t-1}) / dt]
    #
    # Assuming independent measurement errors:
    #
    # Cov(p,p) = P_t
    # Cov(p,v) = P_t / dt
    # Cov(v,v) = (P_t + P_{t-1}) / dt^2

    P_pp = P_current

    P_pv = _scale3(
        P_current,
        1.0 / dt,
    )

    P_vv = _scale3(
        _add3(
            P_current,
            P_previous,
        ),
        1.0 / (dt * dt),
    )

    covariance = tuple(
        tuple(
            (
                P_pp[i][j]
                if i < 3 and j < 3
                else
                P_pv[i][j - 3]
                if i < 3 and j >= 3
                else
                P_pv[j][i - 3]
                if i >= 3 and j < 3
                else
                P_vv[i - 3][j - 3]
            )
            for j in range(6)
        )
        for i in range(6)
    )

    return CausalCVState(
        track_id=current.track_id,
        timestamp_s=current.timestamp_s,
        position_H0_m=position,
        velocity_H0_mps=velocity,
        covariance_6x6=covariance,
        source_frame_indices=(
            previous.frame_index,
            current.frame_index,
        ),
        annotated_velocity_used=False,
        future_information_used=False,
    )
