from __future__ import annotations

from dataclasses import dataclass
from math import (
    atan2,
    hypot,
    isfinite,
    sin,
)

from iscai_stage3.geometry import (
    CartesianObservation,
)

from iscai_stage3.observations import (
    wrap_angle_rad,
)


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class CausalCTRVState:
    """
    Planar Constant-Turn-Rate-and-Velocity
    state in H0, plus constant vertical
    velocity.

    State:
        position_H0
        planar speed
        heading
        turn rate
        vertical velocity

    Heading and turn rate are estimated only
    from past/current H0 positions.

    No annotated heading/velocity.
    No future information.
    """

    track_id: str
    timestamp_s: float

    position_H0_m: Vec3

    planar_speed_mps: float
    heading_rad: float
    turn_rate_radps: float

    vertical_velocity_mps: float

    source_frame_indices: tuple[
        int,
        int,
        int,
    ]

    annotated_velocity_used: bool = False
    annotated_heading_used: bool = False
    future_information_used: bool = False

    def __post_init__(self) -> None:
        if self.annotated_velocity_used:
            raise ValueError(
                "Annotated velocity cannot "
                "enter CTRV."
            )

        if self.annotated_heading_used:
            raise ValueError(
                "Annotated heading cannot "
                "enter CTRV."
            )

        if self.future_information_used:
            raise ValueError(
                "Future information cannot "
                "enter CTRV."
            )

        values = (
            *self.position_H0_m,
            self.timestamp_s,
            self.planar_speed_mps,
            self.heading_rad,
            self.turn_rate_radps,
            self.vertical_velocity_mps,
        )

        if not all(
            isfinite(float(x))
            for x in values
        ):
            raise ValueError(
                "CTRV state contains "
                "non-finite values."
            )

        if self.planar_speed_mps < 0.0:
            raise ValueError(
                "CTRV speed cannot be negative."
            )


def estimate_causal_ctrv_state(
    observations: tuple[
        CartesianObservation,
        ...
    ],
    *,
    min_heading_speed_mps: float = 0.2,
) -> CausalCTRVState:
    if len(observations) < 3:
        raise ValueError(
            "CTRV state requires at least "
            "three causal observations."
        )

    if (
        not isfinite(
            min_heading_speed_mps
        )
        or min_heading_speed_mps < 0.0
    ):
        raise ValueError(
            "min_heading_speed_mps must be "
            "finite and non-negative."
        )

    o0, o1, o2 = observations[-3:]

    if len(
        {
            o0.track_id,
            o1.track_id,
            o2.track_id,
        }
    ) != 1:
        raise ValueError(
            "CTRV observations must belong "
            "to one Stage3 track."
        )

    dt1 = (
        o1.timestamp_s
        -
        o0.timestamp_s
    )

    dt2 = (
        o2.timestamp_s
        -
        o1.timestamp_s
    )

    if dt1 <= 0.0 or dt2 <= 0.0:
        raise ValueError(
            "CTRV timestamps must strictly "
            "increase."
        )

    v01 = tuple(
        (
            o1.position_H0_m[i]
            -
            o0.position_H0_m[i]
        )
        / dt1
        for i in range(3)
    )

    v12 = tuple(
        (
            o2.position_H0_m[i]
            -
            o1.position_H0_m[i]
        )
        / dt2
        for i in range(3)
    )

    speed01 = hypot(
        v01[0],
        v01[1],
    )

    chord_speed12 = hypot(
        v12[0],
        v12[1],
    )

    vertical_velocity = v12[2]

    # If the latest planar motion is too small,
    # heading and turn-rate are not identifiable.
    if (
        chord_speed12
        < min_heading_speed_mps
    ):
        if speed01 >= min_heading_speed_mps:
            heading = atan2(
                v01[1],
                v01[0],
            )
        else:
            heading = 0.0

        turn_rate = 0.0
        planar_speed = chord_speed12

    # A turn rate also cannot be estimated from
    # two reliable headings if the earlier
    # interval was nearly stationary.
    elif (
        speed01
        < min_heading_speed_mps
    ):
        heading = atan2(
            v12[1],
            v12[0],
        )

        turn_rate = 0.0
        planar_speed = chord_speed12

    else:
        heading01 = atan2(
            v01[1],
            v01[0],
        )

        heading12_midpoint = atan2(
            v12[1],
            v12[0],
        )

        # The interval velocities correspond
        # approximately to interval midpoints.
        midpoint_dt = 0.5 * (
            dt1 + dt2
        )

        turn_rate = (
            wrap_angle_rad(
                heading12_midpoint
                -
                heading01
            )
            /
            midpoint_dt
        )

        # Extrapolate the latest interval
        # heading from its midpoint to t2.
        heading = wrap_angle_rad(
            heading12_midpoint
            +
            0.5
            * turn_rate
            * dt2
        )

        # For a constant-turn circle, the finite
        # displacement is a chord. Correct the
        # chord-speed toward the corresponding
        # arc speed. Limit continuously to 1
        # as turn rate -> 0.
        half_angle = (
            0.5
            *
            turn_rate
            *
            dt2
        )

        denominator = sin(
            half_angle
        )

        if (
            abs(half_angle) > 1e-8
            and abs(denominator) > 1e-8
        ):
            correction = abs(
                half_angle
                /
                denominator
            )

            planar_speed = (
                chord_speed12
                *
                correction
            )
        else:
            planar_speed = (
                chord_speed12
            )

    return CausalCTRVState(
        track_id=o2.track_id,
        timestamp_s=o2.timestamp_s,
        position_H0_m=(
            o2.position_H0_m
        ),
        planar_speed_mps=(
            planar_speed
        ),
        heading_rad=wrap_angle_rad(
            heading
        ),
        turn_rate_radps=(
            turn_rate
        ),
        vertical_velocity_mps=(
            vertical_velocity
        ),
        source_frame_indices=(
            o0.frame_index,
            o1.frame_index,
            o2.frame_index,
        ),
        annotated_velocity_used=False,
        annotated_heading_used=False,
        future_information_used=False,
    )
