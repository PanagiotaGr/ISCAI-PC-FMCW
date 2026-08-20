import numpy as np


class HorizonBeamEnvironment:

    ACTION_K = np.asarray(
        [1, 2, 3, 4, 5, 64],
        dtype=np.int64,
    )

    def __init__(
        self,
        posteriors,
        trajectory_ids,
        future_steps,
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
            dtype=np.int64,
        )

        self.future_steps = np.asarray(
            future_steps,
            dtype=np.int64,
        )

        self.num_beams = self.posteriors.shape[1]

        if self.num_beams != 64:
            raise ValueError(
                "Expected 64-beam posterior."
            )

        self.target_mass = float(
            target_mass
        )

        self.beam_cost_weight = float(
            beam_cost_weight
        )

        self.switching_cost_weight = float(
            switching_cost_weight
        )

        self.outage_penalty = float(
            outage_penalty
        )

        self.trajectory_to_indices = {}

        for tid in np.unique(
            self.trajectory_ids
        ):
            idx = np.where(
                self.trajectory_ids == tid
            )[0]

            idx = idx[
                np.argsort(
                    self.future_steps[idx]
                )
            ]

            self.trajectory_to_indices[
                int(tid)
            ] = idx

        self.indices = None
        self.position = None

        self.previous_mask = np.zeros(
            self.num_beams,
            dtype=np.float32,
        )

    @property
    def state_dim(self):
        return (
            self.num_beams
            +
            self.num_beams
            +
            1
        )

    @property
    def num_actions(self):
        return len(
            self.ACTION_K
        )

    def reset(
        self,
        trajectory_id,
    ):
        self.indices = (
            self.trajectory_to_indices[
                int(trajectory_id)
            ]
        )

        self.position = 0

        self.previous_mask.fill(
            0.0
        )

        return self._state()

    def _current_index(self):
        return int(
            self.indices[
                self.position
            ]
        )

    def _state(self):
        idx = self._current_index()

        posterior = self.posteriors[
            idx
        ]

        horizon = float(
            self.future_steps[idx]
        )

        # Assuming horizons 0..9.
        horizon_normalized = (
            horizon / 9.0
        )

        state = np.concatenate(
            [
                posterior,
                self.previous_mask,
                np.asarray(
                    [horizon_normalized],
                    dtype=np.float32,
                ),
            ]
        ).astype(
            np.float32
        )

        return state

    def select_beams(
        self,
        posterior,
        k,
    ):
        ranked = np.argsort(
            posterior
        )[::-1]

        return ranked[:k]

    def step(
        self,
        action_index,
    ):
        action_index = int(
            action_index
        )

        if not (
            0
            <= action_index
            < self.num_actions
        ):
            raise ValueError(
                "Invalid action index."
            )

        k = int(
            self.ACTION_K[
                action_index
            ]
        )

        idx = self._current_index()

        posterior = self.posteriors[
            idx
        ]

        selected = self.select_beams(
            posterior,
            k,
        )

        current_mask = np.zeros(
            self.num_beams,
            dtype=np.float32,
        )

        current_mask[
            selected
        ] = 1.0

        covered_mass = float(
            posterior[
                selected
            ].sum()
        )

        # Beam resource cost.
        beam_cost = (
            k
            /
            self.num_beams
        )

        # Symmetric beam-set change:
        # penalizes both activation and removal.
        if self.position == 0:
            switching_cost = 0.0
        else:
            switching_cost = float(
                np.count_nonzero(
                    current_mask
                    !=
                    self.previous_mask
                )
                /
                self.num_beams
            )

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

        self.previous_mask = (
            current_mask
        )

        done = (
            self.position
            ==
            len(self.indices) - 1
        )

        info = {
            "k":
                k,

            "selected_beams":
                [
                    int(x)
                    for x in selected
                ],

            "covered_mass":
                covered_mass,

            "beam_cost":
                float(
                    beam_cost
                ),

            "switching_cost":
                switching_cost,

            "outage":
                bool(
                    outage
                ),

            "reward":
                float(
                    reward
                ),
        }

        if done:
            next_state = np.zeros(
                self.state_dim,
                dtype=np.float32,
            )
        else:
            self.position += 1
            next_state = self._state()

        return (
            next_state,
            float(reward),
            bool(done),
            info,
        )
