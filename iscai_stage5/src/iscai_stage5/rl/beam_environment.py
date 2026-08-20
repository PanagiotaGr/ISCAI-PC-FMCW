from dataclasses import dataclass
import numpy as np


@dataclass
class BeamState:

    angular_posterior: np.ndarray

    uncertainty: float

    previous_beams: list

    link_quality: float



class BeamEnvironment:


    def __init__(
        self,
        num_beams=64,
        max_active_beams=4,
    ):

        self.num_beams = num_beams
        self.max_active_beams = max_active_beams

        self.state = None



    def reset(
        self,
        angular_posterior,
        uncertainty=0.0,
    ):

        self.state = BeamState(

            angular_posterior=np.asarray(
                angular_posterior,
                dtype=np.float32
            ),

            uncertainty=float(
                uncertainty
            ),

            previous_beams=[],

            link_quality=0.0
        )


        return self._get_state()



    def _get_state(self):

        return {

            "posterior":
                self.state.angular_posterior,

            "uncertainty":
                self.state.uncertainty,

            "previous_beams":
                self.state.previous_beams,

            "link_quality":
                self.state.link_quality,
        }



    def step(
        self,
        selected_beams,
    ):


        selected_beams = list(
            selected_beams
        )


        posterior = (
            self.state.angular_posterior
        )


        beam_probability = sum(
            posterior[b]
            for b in selected_beams
        )


        # simplified optical link model

        link_quality = (
            beam_probability
        )


        energy_cost = (
            len(selected_beams)
            /
            self.num_beams
        )


        switching_cost = (

            len(
                set(selected_beams)
                -
                set(
                    self.state.previous_beams
                )
            )
            /
            self.max_active_beams

        )


        reward = (

            link_quality

            -
            0.3 * energy_cost

            -
            0.2 * switching_cost

        )


        self.state.previous_beams = (
            selected_beams
        )

        self.state.link_quality = (
            link_quality
        )


        done = True


        info = {

            "link_quality":
                link_quality,

            "energy_cost":
                energy_cost,

            "switching_cost":
                switching_cost,

            "reward":
                reward,
        }


        return (
            self._get_state(),
            reward,
            done,
            info,
        )
