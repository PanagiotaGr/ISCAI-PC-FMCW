import numpy as np


class RealBeamEnvironment:

    def __init__(
        self,
        posteriors,
        trajectory_ids,
        future_steps,
        max_k=5,
        target_mass=0.95,
        beam_cost_weight=0.15,
        switching_cost_weight=0.10,
        outage_penalty=1.0,
    ):

        self.posteriors = np.asarray(
            posteriors,
            dtype=np.float32,
        )

        self.trajectory_ids = np.asarray(
            trajectory_ids,
        )

        self.future_steps = np.asarray(
            future_steps,
        )

        self.max_k = int(max_k)
        self.target_mass = float(target_mass)

        self.beam_cost_weight = float(
            beam_cost_weight
        )

        self.switching_cost_weight = float(
            switching_cost_weight
        )

        self.outage_penalty = float(
            outage_penalty
        )

        self.actions = np.arange(
            1,
            self.max_k + 1,
            dtype=np.int64,
        )

        self.index = None
        self.previous_beams = []

    def reset(self, index):

        self.index = int(index)
        self.previous_beams = []

        return self._state()

    def _state(self):

        p = self.posteriors[
            self.index
        ]

        sorted_p = np.sort(
            p
        )[::-1]

        top1 = float(
            sorted_p[0]
        )

        entropy = float(
            -np.sum(
                p
                *
                np.log(
                    p + 1e-12
                )
            )
        )

        norm_entropy = (
            entropy
            /
            np.log(
                len(p)
            )
        )

        return {
            "posterior":
                p,

            "top1_probability":
                top1,

            "normalized_entropy":
                float(
                    norm_entropy
                ),

            "previous_beam_count":
                len(
                    self.previous_beams
                ),
        }

    def step(self, action_k):

        k = int(action_k)

        if not (
            1 <= k <= self.max_k
        ):
            raise ValueError(
                "action_k out of range"
            )

        posterior = self.posteriors[
            self.index
        ]

        ranked = np.argsort(
            posterior
        )[::-1]

        selected = [
            int(x)
            for x in ranked[:k]
        ]

        covered_mass = float(
            posterior[
                selected
            ].sum()
        )

        beam_cost = (
            k
            /
            self.max_k
        )

        if self.previous_beams:

            new_beams = (
                set(selected)
                -
                set(
                    self.previous_beams
                )
            )

            switching_cost = (
                len(new_beams)
                /
                self.max_k
            )

        else:
            switching_cost = 0.0

        outage = (
            covered_mass
            <
            self.target_mass
        )

        reward = (
            covered_mass
            -
            self.beam_cost_weight
            *
            beam_cost
            -
            self.switching_cost_weight
            *
            switching_cost
            -
            self.outage_penalty
            *
            float(outage)
        )

        self.previous_beams = (
            selected
        )

        info = {
            "selected_beams":
                selected,

            "k":
                k,

            "covered_mass":
                covered_mass,

            "beam_cost":
                float(
                    beam_cost
                ),

            "switching_cost":
                float(
                    switching_cost
                ),

            "outage":
                bool(
                    outage
                ),

            "reward":
                float(
                    reward
                ),
        }

        return (
            self._state(),
            float(reward),
            info,
        )
