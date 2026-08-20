from __future__ import annotations

from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)


def estimate_causal_velocity(
    observations: tuple[
        IdealCausalObservable,
        ...
    ],
    index: int,
) -> tuple[float, float, float] | None:
    """
    Backward causal velocity estimate.

    Uses only current and previous
    causal observations.

    No future trajectory access.
    """

    if index <= 0:
        return None

    current = observations[index]
    previous = observations[index - 1]

    if (
        not current.geometry_valid
        or not previous.geometry_valid
    ):
        return None

    if (
        current.actor_position_Ht_m is None
        or previous.actor_position_Ht_m is None
    ):
        return None

    dt = (
        current.timestamp_s
        -
        previous.timestamp_s
    )

    if dt <= 0:
        return None

    return (
        (
            current.actor_position_Ht_m[0]
            -
            previous.actor_position_Ht_m[0]
        ) / dt,

        (
            current.actor_position_Ht_m[1]
            -
            previous.actor_position_Ht_m[1]
        ) / dt,

        (
            current.actor_position_Ht_m[2]
            -
            previous.actor_position_Ht_m[2]
        ) / dt,
    )
