from __future__ import annotations

from dataclasses import dataclass


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class TrajectoryHypothesis:
    trajectory: tuple[Vec3, ...]
    score: float


def trajectory_smoothness_cost(
    trajectory: tuple[Vec3, ...],
) -> float:

    if len(trajectory) < 3:
        return 0.0

    cost = 0.0

    for i in range(2, len(trajectory)):

        x0, y0, z0 = trajectory[i - 2]
        x1, y1, z1 = trajectory[i - 1]
        x2, y2, z2 = trajectory[i]

        cost += (
            (x2 - 2*x1 + x0)**2
            +
            (y2 - 2*y1 + y0)**2
            +
            (z2 - 2*z1 + z0)**2
        )

    return cost


def select_best_hypothesis(
    hypotheses: tuple[
        TrajectoryHypothesis,
        ...
    ],
) -> TrajectoryHypothesis:

    if len(hypotheses) == 0:
        raise ValueError(
            "No trajectory hypotheses."
        )

    return min(
        hypotheses,
        key=lambda h:
            h.score
            +
            trajectory_smoothness_cost(
                h.trajectory
            ),
    )
