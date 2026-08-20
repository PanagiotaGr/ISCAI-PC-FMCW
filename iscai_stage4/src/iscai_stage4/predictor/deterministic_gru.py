from __future__ import annotations

from dataclasses import dataclass

from iscai_stage4.predictor.stage4_input import (
    Stage4ActorInput,
)


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class DeterministicTrajectory:

    positions: tuple[Vec3, ...]

    @property
    def horizon(self) -> int:

        return len(
            self.positions
        )



class DeterministicGRUPredictor:
    """
    Stage 4 deterministic predictor.

    MVP causal implementation.

    Later replaced by trained GRU.
    """


    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.1,
    ):

        if horizon <= 0:

            raise ValueError(
                "Invalid horizon."
            )

        if dt <= 0:

            raise ValueError(
                "Invalid dt."
            )

        self.horizon = horizon
        self.dt = dt



    def predict(
        self,
        actor: Stage4ActorInput,
    ) -> DeterministicTrajectory:
        """
        Constant velocity deterministic head.

        Provides the same interface
        that the GRU will later replace.
        """


        px, py, pz = actor.position

        vx, vy, vz = actor.velocity


        future = []


        for step in range(
            1,
            self.horizon + 1,
        ):

            t = (
                step
                *
                self.dt
            )


            future.append(

                (
                    px + vx*t,

                    py + vy*t,

                    pz + vz*t,
                )

            )


        return DeterministicTrajectory(
            positions=tuple(
                future
            )
        )
