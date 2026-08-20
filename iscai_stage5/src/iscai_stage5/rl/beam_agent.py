import numpy as np


class BeamAgent:


    def __init__(
        self,
        num_beams=64,
        max_active_beams=4,
    ):

        self.num_beams = num_beams
        self.max_active_beams = max_active_beams



    def select_action(
        self,
        state,
    ):

        """
        Select beams based on posterior probability.

        This is the initial policy.
        Later replaced by RL policy.
        """


        posterior = (
            state["posterior"]
        )


        selected = np.argsort(
            posterior
        )[
            -self.max_active_beams:
        ]


        selected = selected.tolist()


        return selected



    def update(
        self,
        reward,
        info,
    ):

        """
        Placeholder for learning update.

        Future:
        Q-learning / PPO / Actor-Critic
        """

        return {
            "reward": reward,
            "updated": False,
        }
