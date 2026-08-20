from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlState:

    timestamp_s: float
    target_id: int
    uncertainty: float
    utility: float



@dataclass(frozen=True)
class ClosedLoopResult:

    states: tuple[ControlState, ...]

    switches: int



def propagate_uncertainty(
    sigma: float,
    dt: float,
) -> float:

    if sigma < 0:
        raise ValueError(
            "Negative uncertainty"
        )

    return sigma + 0.05 * dt



def run_closed_loop(
    target_id: int,
    initial_uncertainty: float,
    horizon_s: float,
    dt: float,
) -> ClosedLoopResult:


    if horizon_s <= 0:
        raise ValueError(
            "Invalid horizon"
        )

    if dt <= 0:
        raise ValueError(
            "Invalid dt"
        )


    states = []

    t = 0.0
    sigma = initial_uncertainty


    while t <= horizon_s:

        utility = (
            1.0 /
            (1.0 + sigma)
        )


        states.append(
            ControlState(
                timestamp_s=t,
                target_id=target_id,
                uncertainty=sigma,
                utility=utility,
            )
        )


        sigma = propagate_uncertainty(
            sigma,
            dt,
        )

        t += dt


    return ClosedLoopResult(
        states=tuple(states),
        switches=0,
    )
