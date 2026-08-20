from __future__ import annotations

import math


Vec3 = tuple[
    float,
    float,
    float,
]


def euclidean_error(
    predicted: Vec3,
    actual: Vec3,
) -> float:
    """
    Euclidean position error.
    """

    dx = predicted[0] - actual[0]
    dy = predicted[1] - actual[1]
    dz = predicted[2] - actual[2]

    return math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )



def normalized_error(
    *,
    error: float,
    variance_trace: float,
) -> float:
    """
    Error scaled by uncertainty.

    Smaller values indicate
    better calibrated uncertainty.
    """

    if variance_trace <= 0:
        raise ValueError(
            "Invalid uncertainty."
        )

    return (
        error /
        math.sqrt(
            variance_trace
        )
    )
