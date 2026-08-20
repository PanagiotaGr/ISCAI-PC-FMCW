from pathlib import Path
import json
import time

import numpy as np
import torch

from rl_real.src.horizon_env import HorizonBeamEnvironment
from rl_real.src.dqn_agent import (
    DQNBeamAgent,
    ReplayBuffer,
)


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path(
    "rl_real/data/real_rl_states.npz"
)

MODEL_FILE = Path(
    "rl_real/models/dqn_horizon_best.pt"
)

FINAL_MODEL_FILE = Path(
    "rl_real/models/dqn_horizon_final.pt"
)

REPORT_FILE = Path(
    "rl_real/reports/dqn_horizon_training_report.json"
)

SEED = 42

TRAIN_FRACTION = 0.80

EPOCHS = 20

REPLAY_CAPACITY = 100_000
MIN_REPLAY_SIZE = 2_000
BATCH_SIZE = 256

TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 1_000

EPS_START = 1.00
EPS_END = 0.05
EPS_DECAY_STEPS = 200_000

LR = 1e-3
GAMMA = 0.95

TARGET_MASS = 0.95

BEAM_COST_WEIGHT = 0.15
SWITCHING_COST_WEIGHT = 0.10
OUTAGE_PENALTY = 1.0


def epsilon_by_step(step):

    fraction = min(
        1.0,
        step / EPS_DECAY_STEPS,
    )

    return (
        EPS_START
        +
        fraction
        *
        (
            EPS_END
            -
            EPS_START
        )
    )


def evaluate(
    env,
    agent,
    trajectory_ids,
):

    rewards = []
    ks = []
    masses = []
    outages = []
    switching = []

    action_counts = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        64: 0,
    }

    for tid in trajectory_ids:

        state = env.reset(
            int(tid)
        )

        done = False

        while not done:

            action_index = (
                agent.select_action(
                    state,
                    epsilon=0.0,
                )
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(
                action_index
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

            action_counts[
                info["k"]
            ] += 1

            state = next_state


    ks = np.asarray(
        ks,
        dtype=np.float64,
    )

    masses = np.asarray(
        masses,
        dtype=np.float64,
    )

    outages = np.asarray(
        outages,
        dtype=np.float64,
    )

    switching = np.asarray(
        switching,
        dtype=np.float64,
    )

    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    )

    return {
        "states":
            int(
                len(ks)
            ),

        "mean_reward":
            float(
                rewards.mean()
            ),

        "mean_K":
            float(
                ks.mean()
            ),

        "median_K":
            float(
                np.median(ks)
            ),

        "p95_K":
            float(
                np.percentile(
                    ks,
                    95,
                )
            ),

        "mean_covered_mass":
            float(
                masses.mean()
            ),

        "outage_rate":
            float(
                outages.mean()
            ),

        "mean_switching_cost":
            float(
                switching.mean()
            ),

        "fallback_rate":
            float(
                np.mean(
                    ks == 64
                )
            ),

        "fraction_K1":
            float(
                np.mean(
                    ks == 1
                )
            ),

        "fraction_K2":
            float(
                np.mean(
                    ks == 2
                )
            ),

        "fraction_K3":
            float(
                np.mean(
                    ks == 3
                )
            ),

        "fraction_K4":
            float(
                np.mean(
                    ks == 4
                )
            ),

        "fraction_K5":
            float(
                np.mean(
                    ks == 5
                )
            ),

        "action_counts":
            {
                str(k): int(v)
                for k, v
                in action_counts.items()
            },
    }


def main():

    print("=" * 76)
    print(
        "REAL STAGE 5 HORIZON-SEQUENTIAL DQN"
    )
    print("=" * 76)

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    rng = np.random.default_rng(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "DEVICE:",
        device
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ========================================================
    # DATA
    # ========================================================

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

    unique_ids = np.unique(
        trajectory_id
    )

    rng.shuffle(
        unique_ids
    )

    n_train = int(
        TRAIN_FRACTION
        *
        len(unique_ids)
    )

    train_ids = unique_ids[
        :n_train
    ]

    val_ids = unique_ids[
        n_train:
    ]

    print()
    print(
        "Train trajectories:",
        len(train_ids)
    )

    print(
        "Validation trajectories:",
        len(val_ids)
    )


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    env = HorizonBeamEnvironment(
        posteriors=posterior,
        trajectory_ids=trajectory_id,
        future_steps=future_step,
        target_mass=TARGET_MASS,
        beam_cost_weight=
            BEAM_COST_WEIGHT,
        switching_cost_weight=
            SWITCHING_COST_WEIGHT,
        outage_penalty=
            OUTAGE_PENALTY,
    )

    print(
        "State dimension:",
        env.state_dim
    )

    print(
        "Actions:",
        env.ACTION_K.tolist()
    )


    # ========================================================
    # AGENT / REPLAY
    # ========================================================

    agent = DQNBeamAgent(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        device=device,
        lr=LR,
        gamma=GAMMA,
        seed=SEED,
    )

    replay = ReplayBuffer(
        capacity=REPLAY_CAPACITY,
        state_dim=env.state_dim,
        seed=SEED,
    )


    # ========================================================
    # TRAIN
    # ========================================================

    global_step = 0
    gradient_steps = 0

    best_val_reward = -np.inf
    best_epoch = None

    epoch_rows = []

    start_time = time.perf_counter()


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        shuffled_ids = train_ids.copy()

        rng.shuffle(
            shuffled_ids
        )

        epoch_rewards = []
        epoch_ks = []
        epoch_masses = []
        epoch_outages = []
        epoch_switch = []
        epoch_losses = []

        for tid in shuffled_ids:

            state = env.reset(
                int(tid)
            )

            done = False

            while not done:

                epsilon = (
                    epsilon_by_step(
                        global_step
                    )
                )

                action = (
                    agent.select_action(
                        state,
                        epsilon=epsilon,
                    )
                )

                (
                    next_state,
                    reward,
                    done,
                    info,
                ) = env.step(
                    action
                )

                replay.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                if (
                    replay.size
                    >= MIN_REPLAY_SIZE
                    and
                    global_step
                    % TRAIN_EVERY
                    == 0
                ):

                    loss = (
                        agent.train_step(
                            replay,
                            batch_size=
                                BATCH_SIZE,
                        )
                    )

                    epoch_losses.append(
                        loss
                    )

                    gradient_steps += 1

                    if (
                        gradient_steps
                        %
                        TARGET_UPDATE_EVERY
                        == 0
                    ):
                        agent.update_target()


                epoch_rewards.append(
                    reward
                )

                epoch_ks.append(
                    info["k"]
                )

                epoch_masses.append(
                    info[
                        "covered_mass"
                    ]
                )

                epoch_outages.append(
                    info["outage"]
                )

                epoch_switch.append(
                    info[
                        "switching_cost"
                    ]
                )

                state = next_state

                global_step += 1


        # ----------------------------------------------------
        # Validation after each epoch
        # ----------------------------------------------------

        validation = evaluate(
            env,
            agent,
            val_ids,
        )

        epoch_row = {
            "epoch":
                epoch,

            "global_step":
                global_step,

            "gradient_steps":
                gradient_steps,

            "epsilon":
                float(
                    epsilon_by_step(
                        global_step
                    )
                ),

            "train_mean_reward":
                float(
                    np.mean(
                        epoch_rewards
                    )
                ),

            "train_mean_K":
                float(
                    np.mean(
                        epoch_ks
                    )
                ),

            "train_mean_covered_mass":
                float(
                    np.mean(
                        epoch_masses
                    )
                ),

            "train_outage_rate":
                float(
                    np.mean(
                        epoch_outages
                    )
                ),

            "train_mean_switching_cost":
                float(
                    np.mean(
                        epoch_switch
                    )
                ),

            "mean_loss":
                (
                    float(
                        np.mean(
                            epoch_losses
                        )
                    )
                    if epoch_losses
                    else None
                ),

            "validation":
                validation,
        }

        epoch_rows.append(
            epoch_row
        )


        print(
            f"epoch={epoch:02d} "
            f"eps={epoch_row['epsilon']:.4f} "
            f"loss="
            f"{epoch_row['mean_loss'] if epoch_row['mean_loss'] is not None else float('nan'):.6f} "
            f"trainR="
            f"{epoch_row['train_mean_reward']:.5f} "
            f"valR="
            f"{validation['mean_reward']:.5f} "
            f"valK="
            f"{validation['mean_K']:.3f} "
            f"mass="
            f"{validation['mean_covered_mass']:.5f} "
            f"outage="
            f"{100*validation['outage_rate']:.3f}% "
            f"fallback="
            f"{100*validation['fallback_rate']:.3f}%"
        )


        # ----------------------------------------------------
        # Best model = highest validation reward
        # ----------------------------------------------------

        if (
            validation[
                "mean_reward"
            ]
            >
            best_val_reward
        ):

            best_val_reward = (
                validation[
                    "mean_reward"
                ]
            )

            best_epoch = epoch

            agent.save(
                MODEL_FILE
            )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    # ========================================================
    # FINAL MODEL
    # ========================================================

    agent.save(
        FINAL_MODEL_FILE
    )


    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "status":
            "PASS",

        "algorithm":
            "DQN",

        "experiment_scope":
            (
                "Horizon-sequential RL over the "
                "10 forecast horizons from one "
                "causal Gaussian trajectory prediction. "
                "This is not claimed as closed-loop "
                "multi-frame sensor RL."
            ),

        "data_source":
            "real_WOMD_Gaussian_posteriors",

        "future_used_as_input":
            False,

        "state": {
            "dimension":
                env.state_dim,

            "components": [
                "current_64_beam_posterior",
                "previous_64_beam_mask",
                "normalized_forecast_horizon",
            ],
        },

        "actions":
            env.ACTION_K.tolist(),

        "fallback_action":
            64,

        "reward": {
            "covered_mass_weight":
                1.0,

            "beam_cost_weight":
                BEAM_COST_WEIGHT,

            "switching_cost_weight":
                SWITCHING_COST_WEIGHT,

            "outage_penalty":
                OUTAGE_PENALTY,

            "target_mass":
                TARGET_MASS,
        },

        "split": {
            "trajectory_level":
                True,

            "train_fraction":
                TRAIN_FRACTION,

            "train_trajectories":
                int(
                    len(train_ids)
                ),

            "validation_trajectories":
                int(
                    len(val_ids)
                ),
        },

        "training": {
            "epochs":
                EPOCHS,

            "replay_capacity":
                REPLAY_CAPACITY,

            "min_replay_size":
                MIN_REPLAY_SIZE,

            "batch_size":
                BATCH_SIZE,

            "train_every":
                TRAIN_EVERY,

            "target_update_every":
                TARGET_UPDATE_EVERY,

            "gamma":
                GAMMA,

            "learning_rate":
                LR,

            "epsilon_start":
                EPS_START,

            "epsilon_end":
                EPS_END,

            "epsilon_decay_steps":
                EPS_DECAY_STEPS,

            "global_steps":
                global_step,

            "gradient_steps":
                gradient_steps,

            "training_time_seconds":
                float(
                    elapsed
                ),

            "epochs_log":
                epoch_rows,
        },

        "selection": {
            "best_epoch":
                best_epoch,

            "best_validation_reward":
                float(
                    best_val_reward
                ),

            "best_model":
                str(
                    MODEL_FILE
                ),

            "final_model":
                str(
                    FINAL_MODEL_FILE
                ),
        },
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
    print("=" * 76)
    print("DQN TRAINING COMPLETE")
    print("=" * 76)

    print(
        "best epoch =",
        best_epoch
    )

    print(
        "best validation reward =",
        best_val_reward
    )

    print(
        "training time =",
        elapsed,
        "sec"
    )

    print(
        "best model =",
        MODEL_FILE
    )

    print(
        "report =",
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
