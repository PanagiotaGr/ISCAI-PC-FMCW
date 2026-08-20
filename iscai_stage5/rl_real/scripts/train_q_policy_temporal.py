from pathlib import Path
import json
import time

import numpy as np

from rl_real.src.environment import RealBeamEnvironment
from rl_real.src.q_agent import TabularQBeamAgent


DATA_FILE = Path(
    "rl_real/data/real_rl_states.npz"
)

MODEL_FILE = Path(
    "rl_real/models/q_beam_policy_temporal.npz"
)

REPORT_FILE = Path(
    "rl_real/reports/q_beam_training_temporal_report.json"
)

SEED = 42

MAX_K = 5
TARGET_MASS = 0.95

EPOCHS = 30
TRAIN_FRACTION = 0.80


def main():

    print("=" * 74)
    print("REAL STAGE 5 TEMPORAL Q-LEARNING BEAM POLICY")
    print("=" * 74)

    rng = np.random.default_rng(SEED)

    data = np.load(DATA_FILE)

    posterior = data["posterior"]
    trajectory_id = data["trajectory_id"]
    future_step = data["future_step"]

    unique_trajectories = np.unique(
        trajectory_id
    )

    rng.shuffle(
        unique_trajectories
    )

    n_train = int(
        TRAIN_FRACTION
        * len(unique_trajectories)
    )

    train_ids = unique_trajectories[:n_train]
    val_ids = unique_trajectories[n_train:]

    print(
        "Train trajectories:",
        len(train_ids)
    )

    print(
        "Validation trajectories:",
        len(val_ids)
    )

    env = RealBeamEnvironment(
        posteriors=posterior,
        trajectory_ids=trajectory_id,
        future_steps=future_step,
        max_k=MAX_K,
        target_mass=TARGET_MASS,
        beam_cost_weight=0.20,
        switching_cost_weight=0.15,
        outage_penalty=1.25,
    )

    agent = TabularQBeamAgent(
        max_k=MAX_K,
        alpha=0.08,
        gamma=0.90,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99995,
        seed=SEED,
    )

    # --------------------------------------------
    # trajectory -> ordered state indices
    # --------------------------------------------

    trajectory_to_indices = {}

    for tid in unique_trajectories:

        idx = np.where(
            trajectory_id == tid
        )[0]

        idx = idx[
            np.argsort(
                future_step[idx]
            )
        ]

        trajectory_to_indices[
            int(tid)
        ] = idx


    epoch_rows = []

    t0 = time.perf_counter()

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        shuffled_ids = train_ids.copy()
        rng.shuffle(shuffled_ids)

        rewards = []
        ks = []
        masses = []
        outages = []
        switching = []

        for tid in shuffled_ids:

            indices = trajectory_to_indices[
                int(tid)
            ]

            # initialize episode at first temporal state
            env.index = int(indices[0])
            env.previous_beams = []

            state = env._state()

            for pos, idx in enumerate(indices):

                env.index = int(idx)

                # state at current temporal position
                state = env._state()

                action_k = agent.select_action(
                    state,
                    training=True,
                )

                _, reward, info = env.step(
                    action_k
                )

                # next temporal state
                if pos + 1 < len(indices):

                    env.index = int(
                        indices[pos + 1]
                    )

                    next_state = env._state()

                    agent.update(
                        state,
                        action_k,
                        reward,
                        next_state,
                    )

                else:
                    # terminal target: no future value
                    s = agent.encode_state(
                        state
                    )

                    a = action_k - 1

                    current = agent.q[
                        s + (a,)
                    ]

                    agent.q[
                        s + (a,)
                    ] = (
                        current
                        +
                        agent.alpha
                        * (
                            reward
                            -
                            current
                        )
                    )

                    agent.epsilon = max(
                        agent.epsilon_min,
                        agent.epsilon
                        * agent.epsilon_decay,
                    )

                rewards.append(
                    reward
                )

                ks.append(
                    info["k"]
                )

                masses.append(
                    info["covered_mass"]
                )

                outages.append(
                    info["outage"]
                )

                switching.append(
                    info["switching_cost"]
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
                    np.mean(rewards)
                ),

            "mean_K":
                float(
                    np.mean(ks)
                ),

            "mean_covered_mass":
                float(
                    np.mean(masses)
                ),

            "outage_rate":
                float(
                    np.mean(outages)
                ),

            "mean_switching_cost":
                float(
                    np.mean(switching)
                ),
        }

        epoch_rows.append(row)

        print(
            f"epoch={epoch:02d} "
            f"eps={row['epsilon']:.4f} "
            f"reward={row['mean_reward']:.6f} "
            f"meanK={row['mean_K']:.3f} "
            f"coverage={row['mean_covered_mass']:.6f} "
            f"outage={100*row['outage_rate']:.3f}% "
            f"switch={row['mean_switching_cost']:.4f}"
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
    val_switching = []

    for tid in val_ids:

        indices = trajectory_to_indices[
            int(tid)
        ]

        env.previous_beams = []

        for idx in indices:

            env.index = int(idx)

            state = env._state()

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
                info["k"]
            )

            val_masses.append(
                info["covered_mass"]
            )

            val_outages.append(
                info["outage"]
            )

            val_switching.append(
                info["switching_cost"]
            )


    val_ks_arr = np.asarray(
        val_ks
    )

    validation = {
        "states":
            int(
                len(val_ks)
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

        "mean_switching_cost":
            float(
                np.mean(
                    val_switching
                )
            ),

        "fraction_K1":
            float(
                np.mean(
                    val_ks_arr == 1
                )
            ),

        "fraction_K2":
            float(
                np.mean(
                    val_ks_arr == 2
                )
            ),

        "fraction_K3":
            float(
                np.mean(
                    val_ks_arr == 3
                )
            ),

        "fraction_K4":
            float(
                np.mean(
                    val_ks_arr == 4
                )
            ),

        "fraction_K5":
            float(
                np.mean(
                    val_ks_arr == 5
                )
            ),
    }


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

        "algorithm":
            "temporal_tabular_Q_learning",

        "data_source":
            "real_WOMD_Gaussian_posteriors",

        "future_used_as_input":
            False,

        "episode_definition":
            "one real trajectory = 10 ordered future-posterior states",

        "train_trajectories":
            int(len(train_ids)),

        "validation_trajectories":
            int(len(val_ids)),

        "actions":
            [1, 2, 3, 4, 5],

        "target_mass":
            TARGET_MASS,

        "epochs":
            EPOCHS,

        "reward_definition": {
            "coverage":
                "+covered_mass",

            "beam_cost":
                "-0.20*(K/5)",

            "switching":
                "-0.15*switching_cost",

            "outage":
                "-1.25 if coverage < 0.95",
        },

        "training":
            epoch_rows,

        "validation":
            validation,

        "training_time_seconds":
            float(elapsed),

        "model_file":
            str(MODEL_FILE),
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
    print("=" * 74)
    print("TEMPORAL VALIDATION")
    print("=" * 74)

    for k, v in validation.items():
        print(
            f"{k} = {v}"
        )

    print()
    print(
        "Training time =",
        elapsed,
        "sec"
    )

    print(
        "Saved model:",
        MODEL_FILE
    )

    print(
        "Saved report:",
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
