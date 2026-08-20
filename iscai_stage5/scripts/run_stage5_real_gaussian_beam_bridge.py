from pathlib import Path
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import GaussianGRUPredictor


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
    "real_gaussian_beam_bridge.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_BEAMS = 64
NUM_SAMPLES = 2000
TARGET_MASS = 0.95


def sample_correlated_gaussian(mu, std, rho, n):
    """
    mu:   [T,2]
    std:  [T,2]
    rho:  [T]
    returns:
        [n,T,2]
    """

    T = mu.shape[0]

    samples = []

    for t in range(T):

        mux, muy = mu[t]
        sx, sy = std[t]
        r = float(rho[t])

        cov = np.array(
            [
                [sx*sx, r*sx*sy],
                [r*sx*sy, sy*sy],
            ],
            dtype=np.float64,
        )

        pts = np.random.multivariate_normal(
            mean=[mux, muy],
            cov=cov,
            size=n,
        )

        samples.append(pts)

    return np.stack(
        samples,
        axis=1,
    )


def angles_to_beam_probs(angles, num_beams):
    """
    angles:
        flat vector of azimuth angles
    """

    probs = np.zeros(
        num_beams,
        dtype=np.float64,
    )

    normalized = (
        angles + np.pi
    ) / (
        2.0 * np.pi
    )

    indices = np.floor(
        normalized * num_beams
    ).astype(int)

    indices = np.clip(
        indices,
        0,
        num_beams - 1,
    )

    for b in indices:
        probs[b] += 1.0

    if probs.sum() > 0:
        probs /= probs.sum()

    return probs


def adaptive_topk(probs, target_mass):
    ranked = np.argsort(
        probs
    )[::-1]

    selected = []
    mass = 0.0

    for b in ranked:
        selected.append(
            int(b)
        )

        mass += float(
            probs[b]
        )

        if mass >= target_mass:
            break

    return selected, mass


def main():

    print("=" * 70)
    print("STAGE 5 REAL GAUSSIAN → BEAM BRIDGE")
    print("=" * 70)

    print("DEVICE:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    dataset = WOMDGRUDataset(
        VAL_DATA,
        max_files=1,
        actor_type=[
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
        ],
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    x, y = next(
        iter(loader)
    )

    x = x.to(DEVICE)

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

    with torch.no_grad():
        pred = model(x)

    mu = (
        pred["mu"][0]
        .cpu()
        .numpy()
    )

    std = (
        pred["std"][0]
        .cpu()
        .numpy()
    )

    rho = (
        pred["rho"][0, :, 0]
        .cpu()
        .numpy()
    )

    samples = sample_correlated_gaussian(
        mu,
        std,
        rho,
        NUM_SAMPLES,
    )

    x_s = samples[..., 0]
    y_s = samples[..., 1]

    angles = np.arctan2(
        y_s,
        x_s,
    ).reshape(-1)

    probs = angles_to_beam_probs(
        angles,
        NUM_BEAMS,
    )

    selected, covered_mass = adaptive_topk(
        probs,
        TARGET_MASS,
    )

    result = {
        "status": "PASS",
        "source": "real_WOMD_validation",
        "checkpoint": str(CHECKPOINT),
        "num_beams": NUM_BEAMS,
        "monte_carlo_samples": NUM_SAMPLES,
        "target_mass": TARGET_MASS,
        "selected_beams": selected,
        "selected_k": len(selected),
        "covered_mass": covered_mass,
        "beam_probabilities": [
            float(x)
            for x in probs
        ],
        "trajectory_mean": mu.tolist(),
        "trajectory_std": std.tolist(),
        "trajectory_rho": rho.tolist(),
        "future_used_as_input": False,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print("Selected beams:", selected)
    print("K =", len(selected))
    print("Covered mass =", covered_mass)

    print()
    print("Top 10 beams:")

    ranked = np.argsort(
        probs
    )[::-1][:10]

    for b in ranked:
        print(
            int(b),
            float(probs[b]),
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
