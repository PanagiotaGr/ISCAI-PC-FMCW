"""Strict causal adjacent backward-difference velocity.

Frozen Stage-1A convention:

    v[t] = (p[t] - p[t-1]) / (timestamp[t] - timestamp[t-1])

only if:
    state[t].valid
    state[t-1].valid
    dt > 0

No search for an older valid state.
No interpolation.
No central difference.
No annotated-WOMD velocity fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from iscai_stage1.contracts.stage1a import Vec3, ZERO_VEC3


@dataclass(frozen=True)
class AdjacentVelocityResult:
    velocity_W_mps: tuple[Vec3, ...]
    velocity_valid: tuple[bool, ...]


def derive_adjacent_backward_velocity(
    *,
    timestamps_s: Sequence[float],
    positions_W_m: Sequence[Vec3],
    state_valid: Sequence[bool],
) -> AdjacentVelocityResult:
    """Derive strict adjacent causal backward-difference velocity."""

    count = len(timestamps_s)

    if len(positions_W_m) != count or len(state_valid) != count:
        raise ValueError(
            "timestamps_s, positions_W_m and state_valid must have equal length."
        )

    velocities: list[Vec3] = []
    validities: list[bool] = []

    for index in range(count):
        # First causal sample has no adjacent predecessor.
        if index == 0:
            velocities.append(ZERO_VEC3)
            validities.append(False)
            continue

        if not state_valid[index] or not state_valid[index - 1]:
            velocities.append(ZERO_VEC3)
            validities.append(False)
            continue

        t_now = timestamps_s[index]
        t_prev = timestamps_s[index - 1]

        if not isfinite(t_now) or not isfinite(t_prev):
            raise ValueError("Valid causal timestamps must be finite.")

        dt = t_now - t_prev

        # Frozen convention: non-positive dt does NOT produce velocity.
        if dt <= 0.0:
            velocities.append(ZERO_VEC3)
            validities.append(False)
            continue

        p_now = positions_W_m[index]
        p_prev = positions_W_m[index - 1]

        if not all(isfinite(value) for value in (*p_now, *p_prev)):
            raise ValueError(
                "Positions used for a valid velocity must be finite."
            )

        velocity: Vec3 = (
            (p_now[0] - p_prev[0]) / dt,
            (p_now[1] - p_prev[1]) / dt,
            (p_now[2] - p_prev[2]) / dt,
        )

        velocities.append(velocity)
        validities.append(True)

    return AdjacentVelocityResult(
        velocity_W_mps=tuple(velocities),
        velocity_valid=tuple(validities),
    )