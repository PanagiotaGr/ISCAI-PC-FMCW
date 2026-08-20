from __future__ import annotations


from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)


from iscai_stage3.baselines.state import (
    TrajectoryState,
)


from iscai_stage3.baselines.velocity_estimation import (
    estimate_causal_velocity,
)


from iscai_stage3.baselines.acceleration_estimation import (
    estimate_causal_acceleration,
)



def ideal_observation_to_state(
    observations: tuple[
        IdealCausalObservable,
        ...
    ],
    index: int,
) -> TrajectoryState:
    """
    Convert Stage2 causal observation series
    into Stage3 trajectory state.

    State components:

        position
        velocity
        acceleration

    Velocity and acceleration are computed
    only from previous causal samples.

    No future access.
    """


    if index <= 0:
        raise ValueError(
            "Need previous causal sample."
        )


    observation = observations[index]


    if not observation.geometry_valid:
        raise ValueError(
            "Invalid geometry."
        )


    if observation.actor_position_Ht_m is None:
        raise ValueError(
            "Missing position."
        )


    velocity = estimate_causal_velocity(
        observations,
        index,
    )


    if velocity is None:
        raise ValueError(
            "Velocity unavailable."
        )


    acceleration = estimate_causal_acceleration(
        observations,
        index,
    )


    return TrajectoryState(

        timestamp_s=(
            observation.timestamp_s
        ),


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


        velocity_mps=velocity,


        acceleration_mps2=acceleration,
    )
