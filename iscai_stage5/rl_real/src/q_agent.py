import numpy as np


class TabularQBeamAgent:

    def __init__(
        self,
        max_k=5,
        alpha=0.10,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        seed=42,
    ):

        self.max_k = int(max_k)

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon)
        self.epsilon_min = float(
            epsilon_min
        )
        self.epsilon_decay = float(
            epsilon_decay
        )

        self.rng = np.random.default_rng(
            seed
        )

        # state:
        # top1 bin        : 0..9
        # entropy bin     : 0..9
        # previous K bin  : 0..5
        #
        # action:
        # K = 1..5

        self.q = np.zeros(
            (
                10,
                10,
                6,
                self.max_k,
            ),
            dtype=np.float32,
        )

    def encode_state(
        self,
        state,
    ):

        top1 = np.clip(
            state[
                "top1_probability"
            ],
            0.0,
            0.999999,
        )

        entropy = np.clip(
            state[
                "normalized_entropy"
            ],
            0.0,
            0.999999,
        )

        prev_k = int(
            np.clip(
                state[
                    "previous_beam_count"
                ],
                0,
                self.max_k,
            )
        )

        top1_bin = int(
            top1 * 10
        )

        entropy_bin = int(
            entropy * 10
        )

        return (
            top1_bin,
            entropy_bin,
            prev_k,
        )

    def select_action(
        self,
        state,
        training=True,
    ):

        s = self.encode_state(
            state
        )

        if (
            training
            and
            self.rng.random()
            <
            self.epsilon
        ):

            action_index = int(
                self.rng.integers(
                    0,
                    self.max_k,
                )
            )

        else:

            action_index = int(
                np.argmax(
                    self.q[s]
                )
            )

        return action_index + 1

    def update(
        self,
        state,
        action_k,
        reward,
        next_state,
    ):

        s = self.encode_state(
            state
        )

        ns = self.encode_state(
            next_state
        )

        a = int(
            action_k - 1
        )

        current = self.q[
            s + (a,)
        ]

        target = (
            reward
            +
            self.gamma
            *
            np.max(
                self.q[ns]
            )
        )

        self.q[
            s + (a,)
        ] = (
            current
            +
            self.alpha
            *
            (
                target
                -
                current
            )
        )

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            *
            self.epsilon_decay,
        )

    def save(
        self,
        path,
    ):

        np.savez_compressed(
            path,
            q=self.q,
            epsilon=np.asarray(
                self.epsilon
            ),
        )

    def load(
        self,
        path,
    ):

        data = np.load(
            path
        )

        self.q = data[
            "q"
        ]

        self.epsilon = float(
            data[
                "epsilon"
        ]
    )
