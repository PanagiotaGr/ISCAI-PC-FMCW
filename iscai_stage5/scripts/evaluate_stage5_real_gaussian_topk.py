from pathlib import Path
import json
import math

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

OUTPUT = Path(
    "reports/stage5_real/"
    "real_gaussian_adaptive_topk_evaluation.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Evaluation only — no training.
MAX_FILES = 5
MAX_SAMPLES = 2000
BATCH_SIZE = 64

# Monte Carlo samples per trajectory / future timestep.
MC_SAMPLES = 500

CODEBOOKS = [16, 32, 64]
TARGETS = [0.90, 0.95, 0.975, 0.99]

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# GAUSSIAN SAMPLING
# ============================================================

def sample_gaussian(mu, std, rho, n):
    """
    mu:  [B,T,2]
    std: [B,T,2]
    rho: [B,T]

    Returns:
        xs, ys with shape [B,T,n]
    """

    B, T, _ = mu.shape

    z1 = np.random.randn(B, T, n)
    z2 = np.random.randn(B, T, n)

    mux = mu[:, :, 0, None]
    muy = mu[:, :, 1, None]

    sx = std[:, :, 0, None]
    sy = std[:, :, 1, None]

    r = np.clip(
        rho[:, :, None],
        -0.999,
        0.999,
    )

    xs = mux + sx * z1

    ys = (
        muy
        + sy
        * (
            r * z1
            + np.sqrt(1.0 - r * r) * z2
        )
    )

    return xs, ys


# ============================================================
# BEAM POSTERIOR
# ============================================================

def beam_probabilities(angles, num_beams):
    """
    angles: [N] radians in [-pi, pi]

    Uniform full-azimuth codebook used for this evaluation.
    """

    normalized = (
        angles + np.pi
    ) / (
        2.0 * np.pi
    )

    idx = np.floor(
        normalized * num_beams
    ).astype(np.int64)

    idx = np.clip(
        idx,
        0,
        num_beams - 1,
    )

    counts = np.bincount(
        idx,
        minlength=num_beams,
    ).astype(np.float64)

    total = counts.sum()

    if total > 0:
        counts /= total

    return counts


def adaptive_topk(probs, target):
    ranked = np.argsort(probs)[::-1]

    cumulative = 0.0
    k = 0

    for b in ranked:
        cumulative += float(probs[b])
        k += 1

        if cumulative >= target:
            break

    return k, cumulative


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("STAGE 5 REAL WOMD GAUSSIAN ADAPTIVE TOP-K EVALUATION")
    print("=" * 72)

    print("DEVICE:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Checkpoint:", CHECKPOINT)
    print("Max samples:", MAX_SAMPLES)
    print("MC samples:", MC_SAMPLES)
    print()


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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
    # Model
    # --------------------------------------------------------

    model = GaussianGRUPredictor(
        input_size=8,
        hidden=128,
        layers=1,
        dropout=0.0,
        future=10,
    ).to(DEVICE)

    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()


    # results[(num_beams, target)] = list of K / covered mass
    stats = {}

    for nb in CODEBOOKS:
        for target in TARGETS:
            stats[(nb, target)] = {
                "k": [],
                "mass": [],
            }


    processed = 0
    evaluated_future_points = 0


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_index, batch in enumerate(loader):

            x, y = batch

            remaining = (
                MAX_SAMPLES - processed
            )

            if remaining <= 0:
                break

            if x.shape[0] > remaining:
                x = x[:remaining]

            x = x.to(DEVICE)

            pred = model(x)

            mu = (
                pred["mu"]
                .detach()
                .cpu()
                .numpy()
            )

            std = (
                pred["std"]
                .detach()
                .cpu()
                .numpy()
            )

            rho = (
                pred["rho"][:, :, 0]
                .detach()
                .cpu()
                .numpy()
            )

            xs, ys = sample_gaussian(
                mu,
                std,
                rho,
                MC_SAMPLES,
            )

            B, T, _ = xs.shape


            # ------------------------------------------------
            # IMPORTANT:
            # posterior evaluated independently per
            # trajectory and future timestep.
            # ------------------------------------------------

            for i in range(B):

                for t in range(T):

                    angles = np.arctan2(
                        ys[i, t],
                        xs[i, t],
                    )

                    for nb in CODEBOOKS:

                        probs = beam_probabilities(
                            angles,
                            nb,
                        )

                        for target in TARGETS:

                            k, mass = adaptive_topk(
                                probs,
                                target,
                            )

                            stats[(nb, target)]["k"].append(k)
                            stats[(nb, target)]["mass"].append(
                                mass
                            )

                    evaluated_future_points += 1


            processed += B

            print(
                f"samples={processed}/{MAX_SAMPLES} "
                f"future_points={evaluated_future_points}",
                flush=True,
            )

            if processed >= MAX_SAMPLES:
                break


    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    rows = []

    for nb in CODEBOOKS:

        for target in TARGETS:

            ks = np.asarray(
                stats[(nb, target)]["k"],
                dtype=np.float64,
            )

            masses = np.asarray(
                stats[(nb, target)]["mass"],
                dtype=np.float64,
            )

            row = {
                "num_beams": nb,
                "beam_width_deg": 360.0 / nb,
                "target_mass": target,

                "evaluations": int(len(ks)),

                "mean_K": float(np.mean(ks)),
                "median_K": float(np.median(ks)),
                "p95_K": float(np.percentile(ks, 95)),
                "max_K": int(np.max(ks)),

                "mean_covered_mass":
                    float(np.mean(masses)),

                "fraction_K1":
                    float(np.mean(ks == 1)),

                "fraction_K_le_3":
                    float(np.mean(ks <= 3)),

                "fraction_K_le_5":
                    float(np.mean(ks <= 5)),
            }

            rows.append(row)


    report = {
        "status": "PASS",

        "evaluation":
            "real_WOMD_validation_trained_Gaussian_GRU",

        "checkpoint":
            str(CHECKPOINT),

        "future_used_as_input":
            False,

        "processed_trajectories":
            processed,

        "evaluated_future_points":
            evaluated_future_points,

        "monte_carlo_samples_per_future_point":
            MC_SAMPLES,

        "codebooks":
            CODEBOOKS,

        "targets":
            TARGETS,

        "codebook_assumption":
            "uniform_full_azimuth_-pi_to_pi",

        "results":
            rows,
    }


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    for r in rows:

        print(
            f"beams={r['num_beams']:2d} "
            f"target={100*r['target_mass']:5.1f}% "
            f"meanK={r['mean_K']:.3f} "
            f"medianK={r['median_K']:.1f} "
            f"p95K={r['p95_K']:.1f} "
            f"maxK={r['max_K']:2d} "
            f"K1={100*r['fraction_K1']:.1f}% "
            f"mass={r['mean_covered_mass']:.4f}"
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
