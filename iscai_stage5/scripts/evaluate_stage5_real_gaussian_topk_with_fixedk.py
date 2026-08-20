from pathlib import Path
import json

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
    "real_gaussian_adaptive_topk_with_fixedk.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_FILES = 5
MAX_SAMPLES = 2000
BATCH_SIZE = 64

MC_SAMPLES = 500

CODEBOOKS = [
    16,
    32,
    64,
]

TARGETS = [
    0.90,
    0.95,
    0.975,
    0.99,
]

FIXED_K = [
    1,
    3,
    5,
]

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# GAUSSIAN SAMPLING
# ============================================================

def sample_gaussian(
    mu,
    std,
    rho,
    n,
):
    """
    mu:
        [B,T,2]

    std:
        [B,T,2]

    rho:
        [B,T]

    Returns:
        xs, ys:
        [B,T,n]
    """

    B, T, _ = mu.shape

    z1 = np.random.randn(
        B,
        T,
        n,
    )

    z2 = np.random.randn(
        B,
        T,
        n,
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
# BEAM PROBABILITIES
# ============================================================

def beam_probabilities(
    angles,
    num_beams,
):

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

    if total > 0:
        counts /= total

    return counts


# ============================================================
# ADAPTIVE TOP-K
# ============================================================

def adaptive_topk(
    probabilities,
    target_mass,
):

    ranked = np.argsort(
        probabilities
    )[::-1]

    selected = []

    accumulated = 0.0

    for beam in ranked:

        selected.append(
            int(beam)
        )

        accumulated += float(
            probabilities[beam]
        )

        if accumulated >= target_mass:
            break

    return (
        selected,
        accumulated,
    )


# ============================================================
# FIXED TOP-K
# ============================================================

def fixed_topk(
    probabilities,
    k,
):

    ranked = np.argsort(
        probabilities
    )[::-1]

    selected = ranked[:k]

    covered_mass = float(
        probabilities[
            selected
        ].sum()
    )

    return (
        [
            int(x)
            for x in selected
        ],
        covered_mass,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print(
        "STAGE 5 REAL GAUSSIAN "
        "ADAPTIVE TOP-K + FIXED-K BASELINES"
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
        "Max samples:",
        MAX_SAMPLES,
    )

    print(
        "MC samples:",
        MC_SAMPLES,
    )

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

    model.load_state_dict(
        state
    )

    model.eval()


    # --------------------------------------------------------
    # Statistics containers
    # --------------------------------------------------------

    adaptive_stats = {}

    for nb in CODEBOOKS:

        for target in TARGETS:

            adaptive_stats[
                (nb, target)
            ] = {
                "k": [],
                "mass": [],
            }


    fixed_stats = {}

    for nb in CODEBOOKS:

        for k in FIXED_K:

            fixed_stats[
                (nb, k)
            ] = {
                "mass": [],
            }


    processed = 0

    future_points = 0


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for x, y in loader:

            remaining = (
                MAX_SAMPLES
                -
                processed
            )

            if remaining <= 0:
                break

            if x.shape[0] > remaining:
                x = x[:remaining]

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


            xs, ys = (
                sample_gaussian(
                    mu,
                    std,
                    rho,
                    MC_SAMPLES,
                )
            )


            B, T, _ = xs.shape


            for i in range(B):

                for t in range(T):

                    angles = np.arctan2(
                        ys[i, t],
                        xs[i, t],
                    )


                    for nb in CODEBOOKS:

                        probabilities = (
                            beam_probabilities(
                                angles,
                                nb,
                            )
                        )


                        # ------------------------------------
                        # Adaptive Top-K
                        # ------------------------------------

                        for target in TARGETS:

                            selected, mass = (
                                adaptive_topk(
                                    probabilities,
                                    target,
                                )
                            )

                            adaptive_stats[
                                (nb, target)
                            ]["k"].append(
                                len(selected)
                            )

                            adaptive_stats[
                                (nb, target)
                            ]["mass"].append(
                                mass
                            )


                        # ------------------------------------
                        # Fixed-K
                        # ------------------------------------

                        for k in FIXED_K:

                            _, mass = (
                                fixed_topk(
                                    probabilities,
                                    k,
                                )
                            )

                            fixed_stats[
                                (nb, k)
                            ]["mass"].append(
                                mass
                            )


                    future_points += 1


            processed += B

            print(
                f"samples="
                f"{processed}/{MAX_SAMPLES} "
                f"future_points="
                f"{future_points}",
                flush=True,
            )

            if processed >= MAX_SAMPLES:
                break


    # ========================================================
    # Adaptive aggregation
    # ========================================================

    adaptive_rows = []

    for nb in CODEBOOKS:

        for target in TARGETS:

            ks = np.asarray(
                adaptive_stats[
                    (nb, target)
                ]["k"],
                dtype=np.float64,
            )

            masses = np.asarray(
                adaptive_stats[
                    (nb, target)
                ]["mass"],
                dtype=np.float64,
            )

            adaptive_rows.append(
                {
                    "num_beams":
                        nb,

                    "beam_width_deg":
                        360.0 / nb,

                    "target_mass":
                        target,

                    "evaluations":
                        int(len(ks)),

                    "mean_K":
                        float(
                            np.mean(ks)
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

                    "max_K":
                        int(
                            np.max(ks)
                        ),

                    "fraction_K1":
                        float(
                            np.mean(
                                ks == 1
                            )
                        ),

                    "fraction_K_le_3":
                        float(
                            np.mean(
                                ks <= 3
                            )
                        ),

                    "fraction_K_le_5":
                        float(
                            np.mean(
                                ks <= 5
                            )
                        ),

                    "mean_covered_mass":
                        float(
                            np.mean(
                                masses
                            )
                        ),
                }
            )


    # ========================================================
    # Fixed-K aggregation
    # ========================================================

    fixed_rows = []

    for nb in CODEBOOKS:

        for k in FIXED_K:

            masses = np.asarray(
                fixed_stats[
                    (nb, k)
                ]["mass"],
                dtype=np.float64,
            )

            fixed_rows.append(
                {
                    "num_beams":
                        nb,

                    "fixed_K":
                        k,

                    "evaluations":
                        int(
                            len(masses)
                        ),

                    "mean_covered_mass":
                        float(
                            np.mean(
                                masses
                            )
                        ),

                    "fraction_mass_ge_90":
                        float(
                            np.mean(
                                masses >= 0.90
                            )
                        ),

                    "fraction_mass_ge_95":
                        float(
                            np.mean(
                                masses >= 0.95
                            )
                        ),

                    "fraction_mass_ge_975":
                        float(
                            np.mean(
                                masses >= 0.975
                            )
                        ),

                    "fraction_mass_ge_99":
                        float(
                            np.mean(
                                masses >= 0.99
                            )
                        ),
                }
            )


    # ========================================================
    # Save
    # ========================================================

    report = {
        "status":
            "PASS",

        "evaluation":
            (
                "real_WOMD_validation_"
                "trained_Gaussian_GRU"
            ),

        "checkpoint":
            str(CHECKPOINT),

        "future_used_as_input":
            False,

        "processed_trajectories":
            processed,

        "evaluated_future_points":
            future_points,

        "monte_carlo_samples_per_future_point":
            MC_SAMPLES,

        "codebooks":
            CODEBOOKS,

        "targets":
            TARGETS,

        "fixed_k":
            FIXED_K,

        "codebook_assumption":
            "uniform_full_azimuth_-pi_to_pi",

        "adaptive_topk":
            adaptive_rows,

        "fixed_k_baselines":
            fixed_rows,
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


    # ========================================================
    # Print adaptive
    # ========================================================

    print()
    print("=" * 76)
    print("ADAPTIVE TOP-K")
    print("=" * 76)

    for r in adaptive_rows:

        print(
            f"beams={r['num_beams']:2d} "
            f"target="
            f"{100*r['target_mass']:5.1f}% "
            f"meanK="
            f"{r['mean_K']:.3f} "
            f"medianK="
            f"{r['median_K']:.1f} "
            f"p95K="
            f"{r['p95_K']:.1f} "
            f"maxK="
            f"{r['max_K']:2d} "
            f"K1="
            f"{100*r['fraction_K1']:.1f}% "
            f"mass="
            f"{r['mean_covered_mass']:.4f}"
        )


    # ========================================================
    # Print fixed
    # ========================================================

    print()
    print("=" * 76)
    print("FIXED-K BASELINES")
    print("=" * 76)

    for r in fixed_rows:

        print(
            f"beams="
            f"{r['num_beams']:2d} "
            f"K="
            f"{r['fixed_K']} "
            f"meanMass="
            f"{r['mean_covered_mass']:.4f} "
            f">=90%="
            f"{100*r['fraction_mass_ge_90']:.1f}% "
            f">=95%="
            f"{100*r['fraction_mass_ge_95']:.1f}% "
            f">=97.5%="
            f"{100*r['fraction_mass_ge_975']:.1f}% "
            f">=99%="
            f"{100*r['fraction_mass_ge_99']:.1f}%"
        )


    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
