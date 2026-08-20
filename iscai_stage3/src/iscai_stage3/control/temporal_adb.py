from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class ADBControlStep:

    timestamp_s: float

    target_id: int

    utility: float

    uncertainty: float



@dataclass(frozen=True)
class ADBControlSequence:

    steps: tuple[ADBControlStep, ...]



def uncertainty_growth(
    initial_sigma: float,
    dt: float,
    growth_rate: float = 0.1,
) -> float:

    if initial_sigma < 0:
        raise ValueError(
            "Negative uncertainty."
        )

    return (
        initial_sigma
        +
        growth_rate * dt
    )



def build_temporal_schedule(
    targets,
    horizon_s: float,
    dt: float,
):

    if horizon_s <= 0:
        raise ValueError(
            "Invalid horizon."
        )

    if dt <= 0:
        raise ValueError(
            "Invalid timestep."
        )


    steps = []

    t = 0.0

    while t <= horizon_s:

        best = max(
            targets,
            key=lambda x: x.utility,
        )

        steps.append(
            ADBControlStep(
                timestamp_s=t,
                target_id=best.target_id,
                utility=best.utility,
                uncertainty=best.uncertainty_penalty,
            )
        )

        t += dt


    return ADBControlSequence(
        tuple(steps)
    )
