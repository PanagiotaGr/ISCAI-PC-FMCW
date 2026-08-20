from __future__ import annotations

from dataclasses import dataclass


Vec3 = tuple[
    float,
    float,
    float,
]


GMM_MODES = (
    "straight",
    "turn",
    "lane_change",
    "stop_decelerate",
    "cross_wait",
)


@dataclass(frozen=True)
class GMMComponent:

    mode: str

    probability: float

    trajectory: tuple[Vec3, ...]



@dataclass(frozen=True)
class GMMTrajectory:

    components: tuple[GMMComponent, ...]



    @property
    def mode_count(self):

        return len(
            self.components
        )



    def probabilities_sum(
        self,
    ):

        return sum(
            x.probability
            for x in self.components
        )



class GMMPredictor:
    """
    Stage 4 multimodal predictor.

    MVP mixture model.

    Later replaced by learned GMM head.
    """


    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.1,
    ):

        self.horizon = horizon
        self.dt = dt



    def predict(
        self,
        actor,
    ) -> GMMTrajectory:


        px, py, pz = actor.position

        vx, vy, vz = actor.velocity


        components = []


        for mode in GMM_MODES:


            trajectory = []


            for step in range(
                1,
                self.horizon + 1,
            ):

                t = (
                    step
                    *
                    self.dt
                )


                if mode == "straight":

                    point = (
                        px + vx*t,
                        py + vy*t,
                        pz + vz*t,
                    )


                elif mode == "turn":

                    point = (
                        px + vx*t,
                        py + 0.5*t*t,
                        pz,
                    )


                elif mode == "lane_change":

                    point = (
                        px + vx*t,
                        py + 2.0*t,
                        pz,
                    )


                elif mode == "stop_decelerate":

                    point = (
                        px + vx*t*0.5,
                        py,
                        pz,
                    )


                else:

                    point = (
                        px,
                        py + vy*t,
                        pz,
                    )


                trajectory.append(
                    point
                )


            components.append(
                GMMComponent(

                    mode=mode,

                    probability=(
                        1.0
                        /
                        len(GMM_MODES)
                    ),

                    trajectory=tuple(
                        trajectory
                    ),
                )
            )


        result = GMMTrajectory(
            components=tuple(
                components
            )
        )


        if abs(
            result.probabilities_sum()
            -
            1.0
        ) > 1e-9:

            raise ValueError(
                "Invalid GMM probabilities."
            )


        return result
