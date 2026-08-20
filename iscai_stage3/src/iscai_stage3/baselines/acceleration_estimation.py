from __future__ import annotations


from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)


from iscai_stage3.baselines.velocity_estimation import (
    estimate_causal_velocity,
)


def estimate_causal_acceleration(
    observations: tuple[
        IdealCausalObservable,
        ...
    ],
    index: int,
) -> tuple[float, float, float] | None:
    """
    Strict causal acceleration estimate.

    Uses only:

        v(t)
        v(t-1)

    Both velocities are estimated
    from backward causal differences.

    No future access.
    """

    if index <= 1:
        return None


    velocity_now = estimate_causal_velocity(
        observations,
        index,
    )


    velocity_previous = estimate_causal_velocity(
        observations,
        index - 1,
    )


    if (
        velocity_now is None
        or velocity_previous is None
    ):
        return None


    dt = (
        observations[index].timestamp_s
        -
        observations[index - 1].timestamp_s
    )


    if dt <= 0:
        return None


    return (
        (
            velocity_now[0]
            -
            velocity_previous[0]
        ) / dt,

        (
            velocity_now[1]
            -
            velocity_previous[1]
        ) / dt,

        (
            velocity_now[2]
            -
            velocity_previous[2]
        ) / dt,
    )
