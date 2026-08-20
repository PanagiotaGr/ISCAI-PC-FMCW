from __future__ import annotations

from math import sqrt


def trajectory_uncertainty(
    trajectory,
) -> float:

    if len(trajectory) == 0:
        raise ValueError(
            "Empty trajectory."
        )

    spread = 0.0

    origin = trajectory[0]

    for point in trajectory:

        dx = point[0] - origin[0]
        dy = point[1] - origin[1]
        dz = point[2] - origin[2]

        spread += sqrt(
            dx*dx +
            dy*dy +
            dz*dz
        )


    return (
        spread /
        len(trajectory)
    )



def target_utility(
    *,
    trajectory,
    confidence: float = 1.0,
) -> float:

    if not (
        0.0 <= confidence <= 1.0
    ):
        raise ValueError(
            "Invalid confidence."
        )


    distance = trajectory_uncertainty(
        trajectory
    )


    utility = (
        confidence *
        (1.0 / (1.0 + distance))
    )


    return utility
