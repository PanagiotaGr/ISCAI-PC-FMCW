from pathlib import Path
import json
import time

import numpy as np

from rl_real.src.environment import RealBeamEnvironment
from rl_real.src.q_agent import TabularQBeamAgent


DATA_FILE = Path(
    "rl_real/data/"
    "real_rl_states.npz"
)

MODEL_FILE = Path(
    "rl_real/models/"
    "q_beam_policy.npz"
)

REPORT_FILE = Path(
    "rl_real/reports/"
    "q_beam_training_report.json"
)

SEED = 42

MAX_K = 5
TARGET_MASS = 0.95

EPOCHS = 25

TRAIN_FRACTION = 0.80


def main():

    print("=" * 72)
    print("REAL STAGE 5 Q-LEARNING BEAM POLICY")
    print("=" * 72)

    rng = np.random.default_rng(
        SEED
    )

    data = np.load(
        DATA_FILE
    )

    posterior = data[
        "posterior"
    ]

    trajectory_id = data[
        "trajectory_id"
    ]

    future_step = data[
        "future_step"
    ]


    # --------------------------------------------------------
    # Split by trajectory ID, not individual future states.
    # This prevents temporal leakage.
    # --------------------------------------------------------

    unique_trajectories = np.unique(
        trajectory_id
    )

    rng.shuffle(
        unique_trajectories
    )

    n_train = int(
        TRAIN_FRACTION
        *
        len(
            unique_trajectories
        )
    )

    train_trajectories = set(
        unique_trajectories[
            :n_train
        ].tolist()
    )

    val_trajectories = set(
        unique_trajectories[
            n_train:
        ].tolist()
    )

    train_mask = np.asarray(
        [
            tid in train_trajectories
            for tid in trajectory_id
        ]
    )

    val_mask = ~train_mask

    train_indices = np.where(
        train_mask
    )[0]

    val_indices = np.where(
        val_mask
    )[0]

    print(
        "Train trajectories:",
        len(
            train_trajectories
        )
    )

    print(
        "Validation trajectories:",
        len(
            val_trajectories
        )
    )

    print(
        "Train states:",
        len(
            train_indices
        )
    )

    print(
        "Validation states:",
        len(
            val_indices
        )
    )


    env = RealBeamEnvironment(
        posteriors=posterior,
        trajectory_ids=trajectory_id,
        future_steps=future_step,
        max_k=MAX_K,
        target_mass=TARGET_MASS,
        beam_cost_weight=0.15,
        switching_cost_weight=0.10,
        outage_penalty=1.0,
    )

    agent = TabularQBeamAgent(
        max_k=MAX_K,
        alpha=0.10,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        seed=SEED,
    )


    epoch_rows = []

    t0 = time.perf_counter()


    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        shuffled = train_indices.copy()

        rng.shuffle(
            shuffled
        )

        rewards = []
        ks = []
        masses = []
        outages = []

        for idx in shuffled:

            state = env.reset(
                idx
            )

            action_k = agent.select_action(
                state,
                training=True,
            )

            next_state, reward, info = (
                env.step(
                    action_k
                )
            )

            agent.update(
                state,
                action_k,
                reward,
                next_state,
            )

            rewards.append(
                reward
            )

            ks.append(
                info[
                    "k"
                ]
            )

            masses.append(
                info[
                    "covered_mass"
                ]
            )

            outages.append(
                info[
                    "outage"
                ]
            )

        row = {
            "epoch":
                epoch,

            "epsilon":
                float(
                    agent.epsilon
                ),

            "mean_reward":
                float(
                    np.mean(
                        rewards
                    )
                ),

            "mean_K":
                float(
                    np.mean(
                        ks
                    )
                ),

            "mean_covered_mass":
                float(
                    np.mean(
                        masses
                    )
                ),

            "outage_rate":
                float(
                    np.mean(
                        outages
                    )
                ),
        }

        epoch_rows.append(
            row
        )

        print(
            f"epoch={epoch:02d} "
            f"eps={row['epsilon']:.4f} "
            f"reward={row['mean_reward']:.6f} "
            f"meanK={row['mean_K']:.3f} "
            f"coverage={row['mean_covered_mass']:.6f} "
            f"outage={100*row['outage_rate']:.3f}%"
        )


    elapsed = (
        time.perf_counter()
        -
        t0
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    val_rewards = []
    val_ks = []
    val_masses = []
    val_outages = []

    for idx in val_indices:

        state = env.reset(
            idx
        )

        action_k = agent.select_action(
            state,
            training=False,
        )

        _, reward, info = env.step(
            action_k
        )

        val_rewards.append(
            reward
        )

        val_ks.append(
            info[
                "k"
            ]
        )

        val_masses.append(
            info[
                "covered_mass"
            ]
        )

        val_outages.append(
            info[
                "outage"
            ]
        )


    validation = {
        "states":
            int(
                len(
                    val_indices
                )
            ),

        "mean_reward":
            float(
                np.mean(
                    val_rewards
                )
            ),

        "mean_K":
            float(
                np.mean(
                    val_ks
                )
            ),

        "median_K":
            float(
                np.median(
                    val_ks
                )
            ),

        "p95_K":
            float(
                np.percentile(
                    val_ks,
                    95,
                )
            ),

        "mean_covered_mass":
            float(
                np.mean(
                    val_masses
                )
            ),

        "outage_rate":
            float(
                np.mean(
                    val_outages
                )
            ),

        "fraction_K1":
            float(
                np.mean(
                    np.asarray(
                        val_ks
                    )
                    ==
                    1
                )
            ),
    }


    # ========================================================
    # SAVE
    # ========================================================

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    agent.save(
        MODEL_FILE
    )

    report = {
        "status":
            "PASS",

        "data_source":
            "real_WOMD_Gaussian_posteriors",

        "future_used_as_input":
            False,

        "train_split":
            {
                "trajectory_level_split":
                    True,

                "train_fraction":
                    TRAIN_FRACTION,

                "train_trajectories":
                    len(
                        train_trajectories
                    ),

                "validation_trajectories":
                    len(
                        val_trajectories
                    ),

                "train_states":
                    int(
                        len(
                            train_indices
                        )
                    ),

                "validation_states":
                    int(
                        len(
                            val_indices
                        )
                    ),
            },

        "rl": {
            "algorithm":
                "tabular_Q_learning",

            "actions":
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],

            "target_mass":
                TARGET_MASS,

            "epochs":
                EPOCHS,

            "alpha":
                agent.alpha,

            "gamma":
                agent.gamma,

            "epsilon_final":
                float(
                    agent.epsilon
                ),
        },

        "reward_definition": {
            "coverage_term":
                "+covered_mass",

            "beam_cost":
                "-0.15*(K/5)",

            "switching_cost":
                "-0.10*switching_cost",

            "outage_penalty":
                "-1.0 if covered_mass < 0.95",
        },

        "training":
            epoch_rows,

        "validation":
            validation,

        "training_time_seconds":
            float(
                elapsed
            ),

        "model_file":
            str(
                MODEL_FILE
            ),
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    print()
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)

    print(
        "mean reward =",
        validation[
            "mean_reward"
        ]
    )

    print(
        "mean K =",
        validation[
            "mean_K"
        ]
    )

    print(
        "median K =",
        validation[
            "median_K"
        ]
    )

    print(
        "p95 K =",
        validation[
            "p95_K"
        ]
    )

    print(
        "mean covered mass =",
        validation[
            "mean_covered_mass"
        ]
    )

    print(
        "outage rate =",
        validation[
            "outage_rate"
        ]
    )

    print(
        "K=1 fraction =",
        validation[
            "fraction_K1"
        ]
    )

    print()
    print(
        "Training time =",
        elapsed,
        "sec",
    )

    print(
        "Saved model:",
        MODEL_FILE,
    )

    print(
        "Saved report:",
        REPORT_FILE,
    )


if __name__ == "__main__":
    main()
