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

Matrix9 = tuple[
    tuple[float, ...],
    ...
]


@dataclass(frozen=True)
class CausalCAState:
    """
    Causal Constant-Acceleration state in H0.

    State ordering:
        [px, py, pz,
         vx, vy, vz,
         ax, ay, az]

    Uses only the last three past/current
    Stage2-derived Cartesian observations.

    No annotated WOMD velocity.
    No annotated WOMD acceleration.
    No future information.
    """

    track_id: str
    timestamp_s: float

    position_H0_m: Vec3
    velocity_H0_mps: Vec3
    acceleration_H0_mps2: Vec3

    covariance_9x9: Matrix9

    source_frame_indices: tuple[
        int,
        int,
        int,
    ]

    annotated_velocity_used: bool = False
    annotated_acceleration_used: bool = False
    future_information_used: bool = False

    def __post_init__(self) -> None:
        if self.annotated_velocity_used:
            raise ValueError(
                "Annotated velocity cannot enter "
                "the realistic CA state."
            )

        if self.annotated_acceleration_used:
            raise ValueError(
                "Annotated acceleration cannot "
                "enter the realistic CA state."
            )

        if self.future_information_used:
            raise ValueError(
                "Future information cannot enter "
                "the causal CA state."
            )

        values = (
            *self.position_H0_m,
            *self.velocity_H0_mps,
            *self.acceleration_H0_mps2,
            self.timestamp_s,
        )

        if not all(
            isfinite(float(x))
            for x in values
        ):
            raise ValueError(
                "CA state contains non-finite "
                "values."
            )

        if len(self.covariance_9x9) != 9:
            raise ValueError(
                "CA covariance must be 9x9."
            )

        if any(
            len(row) != 9
            for row in self.covariance_9x9
        ):
            raise ValueError(
                "CA covariance must be 9x9."
            )

        for i in range(9):
            if (
                self.covariance_9x9[i][i]
                < -1e-12
            ):
                raise ValueError(
                    "CA covariance contains a "
                    "negative diagonal."
                )

            for j in range(9):
                if abs(
                    self.covariance_9x9[i][j]
                    -
                    self.covariance_9x9[j][i]
                ) > 1e-8:
                    raise ValueError(
                        "CA covariance must be "
                        "symmetric."
                    )


def _combine_position_covariances(
    position_covariances,
    coefficients_a,
    coefficients_b,
):
    """
    Cov(
        sum_k a_k p_k,
        sum_k b_k p_k
    )

    assuming independent Stage2 measurement
    errors between different timestamps.
    """

    return tuple(
        tuple(
            sum(
                coefficients_a[k]
                *
                coefficients_b[k]
                *
                position_covariances[k][i][j]
                for k in range(3)
            )
            for j in range(3)
        )
        for i in range(3)
    )


def _assemble_state_covariance(
    position_covariances,
    coefficient_sets,
) -> Matrix9:
    rows = []

    for quantity_i in range(3):
        for axis_i in range(3):
            row = []

            for quantity_j in range(3):
                block = (
                    _combine_position_covariances(
                        position_covariances,
                        coefficient_sets[
                            quantity_i
                        ],
                        coefficient_sets[
                            quantity_j
                        ],
                    )
                )

                for axis_j in range(3):
                    row.append(
                        block[
                            axis_i
                        ][
                            axis_j
                        ]
                    )

            rows.append(tuple(row))

    return tuple(rows)


def estimate_causal_ca_state(
    observations: tuple[
        CartesianObservation,
        ...
    ],
) -> CausalCAState:
    if len(observations) < 3:
        raise ValueError(
            "CA state requires at least three "
            "causal observations."
        )

    p0_obs, p1_obs, p2_obs = (
        observations[-3:]
    )

    track_ids = {
        p0_obs.track_id,
        p1_obs.track_id,
        p2_obs.track_id,
    }

    if len(track_ids) != 1:
        raise ValueError(
            "CA observations must belong to "
            "one Stage3 track."
        )

    t0 = p0_obs.timestamp_s
    t1 = p1_obs.timestamp_s
    t2 = p2_obs.timestamp_s

    dt1 = t1 - t0
    dt2 = t2 - t1

    if dt1 <= 0.0 or dt2 <= 0.0:
        raise ValueError(
            "CA timestamps must strictly "
            "increase."
        )

    # Interval velocities live at the temporal
    # midpoint of each observation interval.
    v01 = tuple(
        (
            p1_obs.position_H0_m[i]
            -
            p0_obs.position_H0_m[i]
        )
        / dt1
        for i in range(3)
    )

    v12 = tuple(
        (
            p2_obs.position_H0_m[i]
            -
            p1_obs.position_H0_m[i]
        )
        / dt2
        for i in range(3)
    )

    midpoint_dt = 0.5 * (
        dt1 + dt2
    )

    acceleration = tuple(
        (
            v12[i]
            -
            v01[i]
        )
        / midpoint_dt
        for i in range(3)
    )

    # v12 is the average velocity over
    # [t1, t2]. Under constant acceleration,
    # extrapolating half dt2 gives the
    # instantaneous velocity at t2.
    velocity_current = tuple(
        v12[i]
        +
        0.5
        * acceleration[i]
        * dt2
        for i in range(3)
    )

    position_current = (
        p2_obs.position_H0_m
    )

    # Linear coefficients of [p0,p1,p2]
    # for the state quantities p(t2), v(t2), a.

    acceleration_coefficients = (
        1.0 / (
            midpoint_dt * dt1
        ),

        -(
            1.0 / dt1
            +
            1.0 / dt2
        )
        / midpoint_dt,

        1.0 / (
            midpoint_dt * dt2
        ),
    )

    velocity_interval_coefficients = (
        0.0,
        -1.0 / dt2,
        1.0 / dt2,
    )

    velocity_coefficients = tuple(
        velocity_interval_coefficients[k]
        +
        0.5
        * dt2
        * acceleration_coefficients[k]
        for k in range(3)
    )

    position_coefficients = (
        0.0,
        0.0,
        1.0,
    )

    position_covariances = (
        p0_obs.position_covariance_H0_m2,
        p1_obs.position_covariance_H0_m2,
        p2_obs.position_covariance_H0_m2,
    )

    covariance = (
        _assemble_state_covariance(
            position_covariances,
            (
                position_coefficients,
                velocity_coefficients,
                acceleration_coefficients,
            ),
        )
    )

    return CausalCAState(
        track_id=p2_obs.track_id,
        timestamp_s=t2,
        position_H0_m=(
            position_current
        ),
        velocity_H0_mps=(
            velocity_current
        ),
        acceleration_H0_mps2=(
            acceleration
        ),
        covariance_9x9=covariance,
        source_frame_indices=(
            p0_obs.frame_index,
            p1_obs.frame_index,
            p2_obs.frame_index,
        ),
        annotated_velocity_used=False,
        annotated_acceleration_used=False,
        future_information_used=False,
    )
