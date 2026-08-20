from pathlib import Path
import json

import numpy as np
import torch

from rl_real.src.horizon_env import HorizonBeamEnvironment
from rl_real.src.dqn_agent import DQNBeamAgent


DATA_FILE = Path(
    "rl_real/data/real_rl_states.npz"
)

MODEL_FILE = Path(
    "rl_real/models/dqn_horizon_best.pt"
)

OUTPUT_FILE = Path(
    "rl_real/reports/dqn_vs_baselines.json"
)

SEED = 42
TRAIN_FRACTION = 0.80

TARGET_MASS = 0.95

BEAM_COST_WEIGHT = 0.15
SWITCHING_COST_WEIGHT = 0.10
OUTAGE_PENALTY = 1.0


def select_topk(posterior, k):

    return np.argsort(
        posterior
    )[::-1][:k]


def adaptive_action(posterior):

    ranked = np.argsort(
        posterior
    )[::-1]

    cumulative = np.cumsum(
        posterior[ranked]
    )

    k = int(
        np.searchsorted(
            cumulative,
            TARGET_MASS,
            side="left",
        )
        + 1
    )

    # Same action space as DQN:
    # 1..5, otherwise exhaustive fallback.
    if k > 5:
        return 64

    return k


def fixed_action(k):

    def policy(_posterior):
        return k

    return policy


def adaptive_policy(posterior):
    return adaptive_action(
        posterior
    )


def calculate_step(
    posterior,
    k,
    previous_mask,
    first_step,
):

    selected = select_topk(
        posterior,
        k,
    )

    current_mask = np.zeros(
        64,
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

    beam_cost = (
        k / 64.0
    )

    if first_step:
        switching_cost = 0.0
    else:
        switching_cost = float(
            np.count_nonzero(
                current_mask
                !=
                previous_mask
            )
            /
            64.0
        )

    outage = (
        covered_mass
        <
        TARGET_MASS
    )

    reward = (
        covered_mass
        -
        BEAM_COST_WEIGHT
        *
        beam_cost
        -
        SWITCHING_COST_WEIGHT
        *
        switching_cost
        -
        OUTAGE_PENALTY
        *
        float(outage)
    )

    return (
        current_mask,
        {
            "k": int(k),
            "covered_mass":
                covered_mass,
            "beam_cost":
                float(beam_cost),
            "switching_cost":
                switching_cost,
            "outage":
                bool(outage),
            "reward":
                float(reward),
        },
    )


def summarize(rows):

    ks = np.asarray(
        [x["k"] for x in rows],
        dtype=np.float64,
    )

    masses = np.asarray(
        [
            x["covered_mass"]
            for x in rows
        ],
        dtype=np.float64,
    )

    outages = np.asarray(
        [x["outage"] for x in rows],
        dtype=np.float64,
    )

    rewards = np.asarray(
        [x["reward"] for x in rows],
        dtype=np.float64,
    )

    switches = np.asarray(
        [
            x["switching_cost"]
            for x in rows
        ],
        dtype=np.float64,
    )

    beam_costs = np.asarray(
        [
            x["beam_cost"]
            for x in rows
        ],
        dtype=np.float64,
    )

    return {
        "states":
            int(len(rows)),

        "mean_reward":
            float(rewards.mean()),

        "mean_K":
            float(ks.mean()),

        "median_K":
            float(np.median(ks)),

        "p95_K":
            float(
                np.percentile(
                    ks,
                    95,
                )
            ),

        "mean_covered_mass":
            float(masses.mean()),

        "outage_rate":
            float(outages.mean()),

        "mean_beam_cost":
            float(
                beam_costs.mean()
            ),

        "mean_switching_cost":
            float(
                switches.mean()
            ),

        "fallback_rate":
            float(
                np.mean(
                    ks == 64
                )
            ),

        "K1_fraction":
            float(
                np.mean(
                    ks == 1
                )
            ),

        "K2_fraction":
            float(
                np.mean(
                    ks == 2
                )
            ),

        "K3_fraction":
            float(
                np.mean(
                    ks == 3
                )
            ),

        "K4_fraction":
            float(
                np.mean(
                    ks == 4
                )
            ),

        "K5_fraction":
            float(
                np.mean(
                    ks == 5
                )
            ),
    }


def evaluate_baseline(
    posterior,
    trajectory_id,
    future_step,
    validation_ids,
    policy,
):

    rows = []

    for tid in validation_ids:

        idx = np.where(
            trajectory_id == tid
        )[0]

        idx = idx[
            np.argsort(
                future_step[idx]
            )
        ]

        previous_mask = np.zeros(
            64,
            dtype=np.float32,
        )

        for position, i in enumerate(
            idx
        ):

            p = posterior[i]

            k = int(
                policy(p)
            )

            (
                previous_mask,
                info,
            ) = calculate_step(
                posterior=p,
                k=k,
                previous_mask=
                    previous_mask,
                first_step=(
                    position == 0
                ),
            )

            rows.append(
                info
            )

    return summarize(
        rows
    )


def evaluate_dqn(
    posterior,
    trajectory_id,
    future_step,
    validation_ids,
    device,
):

    env = HorizonBeamEnvironment(
        posteriors=posterior,
        trajectory_ids=
            trajectory_id,
        future_steps=future_step,
        target_mass=TARGET_MASS,
        beam_cost_weight=
            BEAM_COST_WEIGHT,
        switching_cost_weight=
            SWITCHING_COST_WEIGHT,
        outage_penalty=
            OUTAGE_PENALTY,
    )

    agent = DQNBeamAgent(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        device=device,
        lr=1e-3,
        gamma=0.95,
        seed=SEED,
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    agent.online.load_state_dict(
        checkpoint["online"]
    )

    agent.target.load_state_dict(
        checkpoint["target"]
    )

    agent.online.eval()
    agent.target.eval()

    rows = []

    with torch.no_grad():

        for tid in validation_ids:

            state = env.reset(
                int(tid)
            )

            done = False

            while not done:

                action = (
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
                    action
                )

                rows.append(
                    info
                )

                state = next_state

    return summarize(
        rows
    )


def main():

    print("=" * 78)
    print(
        "REAL STAGE 5 DQN VS BASELINES"
    )
    print("=" * 78)

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
    # Reconstruct EXACT training/validation split.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

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

    validation_ids = (
        unique_ids[n_train:]
    )

    print(
        "Validation trajectories:",
        len(validation_ids)
    )

    print(
        "Validation states:",
        len(validation_ids) * 10
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
            torch.cuda.get_device_name(
                0
            )
        )

    print()
    print(
        "Evaluating DQN..."
    )

    dqn = evaluate_dqn(
        posterior,
        trajectory_id,
        future_step,
        validation_ids,
        device,
    )

    print(
        "Evaluating Adaptive Top-K..."
    )

    adaptive = evaluate_baseline(
        posterior,
        trajectory_id,
        future_step,
        validation_ids,
        adaptive_policy,
    )

    print(
        "Evaluating Fixed K=1..."
    )

    fixed1 = evaluate_baseline(
        posterior,
        trajectory_id,
        future_step,
        validation_ids,
        fixed_action(1),
    )

    print(
        "Evaluating Fixed K=3..."
    )

    fixed3 = evaluate_baseline(
        posterior,
        trajectory_id,
        future_step,
        validation_ids,
        fixed_action(3),
    )

    print(
        "Evaluating Fixed K=5..."
    )

    fixed5 = evaluate_baseline(
        posterior,
        trajectory_id,
        future_step,
        validation_ids,
        fixed_action(5),
    )

    results = {
        "DQN":
            dqn,

        "AdaptiveTopK95":
            adaptive,

        "FixedK1":
            fixed1,

        "FixedK3":
            fixed3,

        "FixedK5":
            fixed5,
    }

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    for name, r in results.items():

        print(
            f"{name:16s} "
            f"reward={r['mean_reward']:.6f} "
            f"meanK={r['mean_K']:.3f} "
            f"mass={r['mean_covered_mass']:.6f} "
            f"outage={100*r['outage_rate']:.3f}% "
            f"switch={r['mean_switching_cost']:.5f} "
            f"fallback={100*r['fallback_rate']:.3f}%"
        )

    report = {
        "status":
            "PASS",

        "comparison":
            (
                "Same 1000 held-out trajectories, "
                "same posterior states, same reward "
                "definition and same switching metric."
            ),

        "target_mass":
            TARGET_MASS,

        "validation_trajectories":
            int(
                len(validation_ids)
            ),

        "validation_states":
            int(
                len(validation_ids)
                * 10
            ),

        "DQN_checkpoint":
            str(
                MODEL_FILE
            ),

        "adaptive_policy":
            (
                "Minimum K reaching 95% posterior "
                "mass for K<=5; otherwise K=64 "
                "exhaustive fallback."
            ),

        "reward_definition": {
            "covered_mass":
                1.0,

            "beam_cost_weight":
                BEAM_COST_WEIGHT,

            "switching_cost_weight":
                SWITCHING_COST_WEIGHT,

            "outage_penalty":
                OUTAGE_PENALTY,
        },

        "results":
            results,

        "scientific_scope":
            (
                "Horizon-sequential decisions over "
                "forecast horizons from one causal "
                "prediction; not claimed as multi-frame "
                "closed-loop sensor interaction."
            ),
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
