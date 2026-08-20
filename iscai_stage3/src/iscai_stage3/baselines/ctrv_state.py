from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)


from iscai_stage3.baselines.velocity_estimation import (
    estimate_causal_velocity,
)



@dataclass(frozen=True)
class CTRVState:
    """
    Constant Turn Rate and Velocity state.

    Built only from causal observations.
    """

    position_m: tuple[
        float,
        float,
        float,
    ]

    speed_mps: float

    heading_rad: float

    yaw_rate_radps: float



def estimate_ctrv_state(
    observations: tuple[
        IdealCausalObservable,
        ...
    ],
    index: int,
) -> CTRVState:
    """
    Build CTRV state from causal history.

    Uses:

        position(t)
        velocity(t)
        velocity(t-1)

    No future access.
    """


    if index <= 1:
        raise ValueError(
            "Need at least two causal velocity estimates."
        )


    observation = observations[index]


    if (
        not observation.geometry_valid
        or observation.actor_position_Ht_m is None
    ):
        raise ValueError(
            "Invalid causal geometry."
        )


    velocity = estimate_causal_velocity(
        observations,
        index,
    )


    velocity_previous = estimate_causal_velocity(
        observations,
        index - 1,
    )


    if (
        velocity is None
        or velocity_previous is None
    ):
        raise ValueError(
            "Velocity unavailable."
        )


    speed = math.sqrt(
        velocity[0] ** 2
        +
        velocity[1] ** 2
    )


    if speed <= 1e-9:
        raise ValueError(
            "Undefined heading for zero speed."
        )


    heading = math.atan2(
        velocity[1],
        velocity[0],
    )


    previous_heading = math.atan2(
        velocity_previous[1],
        velocity_previous[0],
    )


    dt = (
        observations[index].timestamp_s
        -
        observations[index - 1].timestamp_s
    )


    if dt <= 0:
        raise ValueError(
            "Invalid timestep."
        )


    yaw_rate = (
        heading
        -
        previous_heading
    ) / dt


    return CTRVState(
        position_m=(
            float(
                observation.actor_position_Ht_m[0]
            ),
            float(
                observation.actor_position_Ht_m[1]
            ),
            float(
                observation.actor_position_Ht_m[2]
            ),
        ),

        speed_mps=speed,

        heading_rad=heading,

        yaw_rate_radps=yaw_rate,
    )
