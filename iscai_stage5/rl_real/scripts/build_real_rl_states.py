from pathlib import Path
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import GaussianGRUPredictor


# ============================================================
# CONFIG
# ============================================================

VAL_DATA = (
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)

CHECKPOINT = Path(
    "/home/agni/waymo/iscai_stage4/"
    "checkpoints/stage4_auto_cuda/"
    "ALL_gaussian_h128_l1_d0.0_lr0.001_b32.pt"
)

OUTPUT_NPZ = Path(
    "rl_real/data/"
    "real_rl_states.npz"
)

OUTPUT_JSON = Path(
    "rl_real/reports/"
    "real_rl_states_summary.json"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_FILES = 5

# Same scale as the previous real Stage-5 evaluation.
MAX_TRAJECTORIES = 5000

BATCH_SIZE = 64

NUM_BEAMS = 64

MC_SAMPLES = 500

FUTURE_STEPS = 10

SEED = 42


# ============================================================
# GAUSSIAN SAMPLING
# ============================================================

def sample_gaussian(
    mu,
    std,
    rho,
    n,
    rng,
):
    """
    Parameters
    ----------
    mu:
        [B,T,2]

    std:
        [B,T,2]

    rho:
        [B,T]

    Returns
    -------
    xs, ys:
        [B,T,n]
    """

    B, T, _ = mu.shape

    z1 = rng.standard_normal(
        (
            B,
            T,
            n,
        )
    )

    z2 = rng.standard_normal(
        (
            B,
            T,
            n,
        )
    )

    mux = mu[:, :, 0, None]
    muy = mu[:, :, 1, None]

    sx = std[:, :, 0, None]
    sy = std[:, :, 1, None]

    r = np.clip(
        rho[:, :, None],
        -0.999,
        0.999,
    )

    xs = (
        mux
        +
        sx * z1
    )

    ys = (
        muy
        +
        sy
        *
        (
            r * z1
            +
            np.sqrt(
                1.0 - r*r
            )
            * z2
        )
    )

    return xs, ys


# ============================================================
# 64-BEAM POSTERIOR
# ============================================================

def beam_posterior(
    angles,
    num_beams,
):
    """
    angles:
        [MC]

    output:
        [num_beams]
    """

    normalized = (
        angles + np.pi
    ) / (
        2.0 * np.pi
    )

    indices = np.floor(
        normalized
        *
        num_beams
    ).astype(
        np.int64
    )

    indices = np.clip(
        indices,
        0,
        num_beams - 1,
    )

    counts = np.bincount(
        indices,
        minlength=num_beams,
    ).astype(
        np.float64
    )

    total = counts.sum()

    if total <= 0:
        raise RuntimeError(
            "Empty beam posterior."
        )

    return counts / total


# ============================================================
# POSTERIOR FEATURES
# ============================================================

def posterior_features(
    posterior,
):

    eps = 1e-12

    p = np.asarray(
        posterior,
        dtype=np.float64,
    )

    top1_beam = int(
        np.argmax(p)
    )

    sorted_p = np.sort(
        p
    )[::-1]

    top1_prob = float(
        sorted_p[0]
    )

    top2_mass = float(
        sorted_p[:2].sum()
    )

    top3_mass = float(
        sorted_p[:3].sum()
    )

    top5_mass = float(
        sorted_p[:5].sum()
    )

    entropy = float(
        -np.sum(
            p
            *
            np.log(
                p + eps
            )
        )
    )

    max_entropy = np.log(
        len(p)
    )

    normalized_entropy = float(
        entropy
        /
        max_entropy
    )

    effective_beams = float(
        np.exp(
            entropy
        )
    )

    return {
        "top1_beam":
            top1_beam,

        "top1_prob":
            top1_prob,

        "top2_mass":
            top2_mass,

        "top3_mass":
            top3_mass,

        "top5_mass":
            top5_mass,

        "entropy":
            entropy,

        "normalized_entropy":
            normalized_entropy,

        "effective_beams":
            effective_beams,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print(
        "STAGE 5 REAL RL STATE DATASET"
    )
    print("=" * 76)

    print(
        "DEVICE:",
        DEVICE,
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Max trajectories:",
        MAX_TRAJECTORIES,
    )

    print(
        "MC samples:",
        MC_SAMPLES,
    )

    print(
        "Beams:",
        NUM_BEAMS,
    )

    print()


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    rng = np.random.default_rng(
        SEED
    )


    # --------------------------------------------------------
    # Real WOMD validation dataset
    # --------------------------------------------------------

    print(
        "Loading real WOMD validation dataset..."
    )

    dataset = WOMDGRUDataset(
        VAL_DATA,
        max_files=MAX_FILES,
        actor_type=[
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
        ],
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


    # --------------------------------------------------------
    # Stage-4 trained Gaussian predictor
    # --------------------------------------------------------

    model = GaussianGRUPredictor(
        input_size=8,
        hidden=128,
        layers=1,
        dropout=0.0,
        future=FUTURE_STEPS,
    ).to(
        DEVICE
    )

    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()


    # --------------------------------------------------------
    # Output buffers
    # --------------------------------------------------------

    posteriors = []

    trajectory_ids = []

    future_steps = []

    top1_beams = []

    top1_probs = []

    top2_masses = []

    top3_masses = []

    top5_masses = []

    entropies = []

    normalized_entropies = []

    effective_beams_list = []

    mean_std_x = []

    mean_std_y = []

    abs_rhos = []


    processed_trajectories = 0

    global_trajectory_id = 0

    start_time = time.perf_counter()


    # ========================================================
    # REAL INFERENCE
    # ========================================================

    with torch.no_grad():

        for x, _future_target in loader:

            remaining = (
                MAX_TRAJECTORIES
                -
                processed_trajectories
            )

            if remaining <= 0:
                break

            if x.shape[0] > remaining:

                x = x[
                    :remaining
                ]


            B = x.shape[0]

            x = x.to(
                DEVICE
            )


            prediction = model(
                x
            )


            mu = (
                prediction["mu"]
                .detach()
                .cpu()
                .numpy()
            )

            std = (
                prediction["std"]
                .detach()
                .cpu()
                .numpy()
            )

            rho = (
                prediction["rho"][
                    :,
                    :,
                    0
                ]
                .detach()
                .cpu()
                .numpy()
            )


            # ------------------------------------------------
            # Monte Carlo predictive samples
            # ------------------------------------------------

            xs, ys = sample_gaussian(
                mu,
                std,
                rho,
                MC_SAMPLES,
                rng,
            )


            # ------------------------------------------------
            # One RL state per future step
            # ------------------------------------------------

            for i in range(B):

                trajectory_id = (
                    global_trajectory_id
                    +
                    i
                )

                for t in range(
                    FUTURE_STEPS
                ):

                    angles = np.arctan2(
                        ys[i, t],
                        xs[i, t],
                    )

                    posterior = (
                        beam_posterior(
                            angles,
                            NUM_BEAMS,
                        )
                    )

                    f = posterior_features(
                        posterior
                    )


                    posteriors.append(
                        posterior.astype(
                            np.float32
                        )
                    )

                    trajectory_ids.append(
                        trajectory_id
                    )

                    future_steps.append(
                        t
                    )

                    top1_beams.append(
                        f["top1_beam"]
                    )

                    top1_probs.append(
                        f["top1_prob"]
                    )

                    top2_masses.append(
                        f["top2_mass"]
                    )

                    top3_masses.append(
                        f["top3_mass"]
                    )

                    top5_masses.append(
                        f["top5_mass"]
                    )

                    entropies.append(
                        f["entropy"]
                    )

                    normalized_entropies.append(
                        f[
                            "normalized_entropy"
                        ]
                    )

                    effective_beams_list.append(
                        f[
                            "effective_beams"
                        ]
                    )

                    mean_std_x.append(
                        float(
                            std[i, t, 0]
                        )
                    )

                    mean_std_y.append(
                        float(
                            std[i, t, 1]
                        )
                    )

                    abs_rhos.append(
                        float(
                            abs(
                                rho[i, t]
                            )
                        )
                    )


            processed_trajectories += B

            global_trajectory_id += B


            print(
                f"trajectories="
                f"{processed_trajectories}/"
                f"{MAX_TRAJECTORIES} "
                f"states="
                f"{len(posteriors)}",
                flush=True,
            )


            if (
                processed_trajectories
                >=
                MAX_TRAJECTORIES
            ):
                break


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    # ========================================================
    # Convert arrays
    # ========================================================

    posteriors = np.asarray(
        posteriors,
        dtype=np.float32,
    )

    trajectory_ids = np.asarray(
        trajectory_ids,
        dtype=np.int32,
    )

    future_steps = np.asarray(
        future_steps,
        dtype=np.int16,
    )

    top1_beams = np.asarray(
        top1_beams,
        dtype=np.int16,
    )

    top1_probs = np.asarray(
        top1_probs,
        dtype=np.float32,
    )

    top2_masses = np.asarray(
        top2_masses,
        dtype=np.float32,
    )

    top3_masses = np.asarray(
        top3_masses,
        dtype=np.float32,
    )

    top5_masses = np.asarray(
        top5_masses,
        dtype=np.float32,
    )

    entropies = np.asarray(
        entropies,
        dtype=np.float32,
    )

    normalized_entropies = np.asarray(
        normalized_entropies,
        dtype=np.float32,
    )

    effective_beams_list = np.asarray(
        effective_beams_list,
        dtype=np.float32,
    )

    mean_std_x = np.asarray(
        mean_std_x,
        dtype=np.float32,
    )

    mean_std_y = np.asarray(
        mean_std_y,
        dtype=np.float32,
    )

    abs_rhos = np.asarray(
        abs_rhos,
        dtype=np.float32,
    )


    # ========================================================
    # Safety checks
    # ========================================================

    if posteriors.ndim != 2:

        raise RuntimeError(
            "Posterior array must be 2D."
        )

    if posteriors.shape[1] != NUM_BEAMS:

        raise RuntimeError(
            "Unexpected beam dimension."
        )

    posterior_sums = (
        posteriors.sum(
            axis=1
        )
    )

    if not np.allclose(
        posterior_sums,
        1.0,
        atol=1e-5,
    ):

        raise RuntimeError(
            "Beam posterior rows "
            "do not sum to one."
        )

    if not np.isfinite(
        posteriors
    ).all():

        raise RuntimeError(
            "Non-finite posterior values."
        )


    # ========================================================
    # Save compressed RL dataset
    # ========================================================

    OUTPUT_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,

        posterior=
            posteriors,

        trajectory_id=
            trajectory_ids,

        future_step=
            future_steps,

        top1_beam=
            top1_beams,

        top1_probability=
            top1_probs,

        top2_mass=
            top2_masses,

        top3_mass=
            top3_masses,

        top5_mass=
            top5_masses,

        entropy=
            entropies,

        normalized_entropy=
            normalized_entropies,

        effective_beams=
            effective_beams_list,

        gaussian_std_x=
            mean_std_x,

        gaussian_std_y=
            mean_std_y,

        gaussian_abs_rho=
            abs_rhos,
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "status":
            "PASS",

        "data_source":
            "real_WOMD_validation",

        "checkpoint":
            str(CHECKPOINT),

        "future_used_as_input":
            False,

        "processed_trajectories":
            int(
                processed_trajectories
            ),

        "states":
            int(
                posteriors.shape[0]
            ),

        "future_steps_per_trajectory":
            FUTURE_STEPS,

        "num_beams":
            NUM_BEAMS,

        "monte_carlo_samples_per_state":
            MC_SAMPLES,

        "posterior_shape":
            list(
                posteriors.shape
            ),

        "posterior_sum_mean":
            float(
                posterior_sums.mean()
            ),

        "top1_probability": {
            "mean":
                float(
                    top1_probs.mean()
                ),

            "median":
                float(
                    np.median(
                        top1_probs
                    )
                ),

            "p05":
                float(
                    np.percentile(
                        top1_probs,
                        5,
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        top1_probs,
                        95,
                    )
                ),
        },

        "normalized_entropy": {
            "mean":
                float(
                    normalized_entropies.mean()
                ),

            "median":
                float(
                    np.median(
                        normalized_entropies
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        normalized_entropies,
                        95,
                    )
                ),
        },

        "effective_beams": {
            "mean":
                float(
                    effective_beams_list.mean()
                ),

            "median":
                float(
                    np.median(
                        effective_beams_list
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        effective_beams_list,
                        95,
                    )
                ),
        },

        "generation_time_seconds":
            float(
                elapsed
            ),

        "output_npz":
            str(
                OUTPUT_NPZ
            ),
    }


    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )


    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 76)
    print(
        "REAL RL STATE DATASET COMPLETE"
    )
    print("=" * 76)

    print(
        "trajectories =",
        processed_trajectories,
    )

    print(
        "states =",
        posteriors.shape[0],
    )

    print(
        "posterior shape =",
        posteriors.shape,
    )

    print(
        "mean top1 probability =",
        float(
            top1_probs.mean()
        ),
    )

    print(
        "mean normalized entropy =",
        float(
            normalized_entropies.mean()
        ),
    )

    print(
        "mean effective beams =",
        float(
            effective_beams_list.mean()
        ),
    )

    print(
        "generation time =",
        elapsed,
        "sec",
    )

    print()
    print(
        "Saved dataset:",
        OUTPUT_NPZ,
    )

    print(
        "Saved summary:",
        OUTPUT_JSON,
    )


if __name__ == "__main__":
    main()
