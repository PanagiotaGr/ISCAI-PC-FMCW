from pathlib import Path
import json
import math

import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import GaussianGRUPredictor
from iscai_stage4.training.gaussian_loss import gaussian_nll


VAL_DATA = (
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)

CHECKPOINT_DIR = Path("checkpoints/stage4_auto_cuda")
REPORT_DIR = Path("reports/stage4_auto_cuda")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MAX_VAL_FILES = 150
NUM_WORKERS = 4

TARGET_CLASSES = [
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


CANDIDATES = [
    {
        "label": "best_nll",
        "name": "ALL_gaussian_h128_l1_d0.0_lr0.001_b32",
        "hidden": 128,
        "layers": 1,
        "dropout": 0.0,
        "batch": 32,
    },
    {
        "label": "best_ade",
        "name": "ALL_gaussian_h128_l2_d0.0_lr0.001_b32",
        "hidden": 128,
        "layers": 2,
        "dropout": 0.0,
        "batch": 32,
    },
    {
        "label": "best_fde",
        "name": "ALL_gaussian_h128_l2_d0.0_lr0.001_b64",
        "hidden": 128,
        "layers": 2,
        "dropout": 0.0,
        "batch": 64,
    },
]


# Chi-square quantiles for df=2.
# For df=2: q(p) = -2 ln(1-p)
LEVELS = [
    0.50,
    0.80,
    0.90,
    0.95,
    0.99,
]


def chi2_df2_quantile(p):
    return -2.0 * math.log(1.0 - p)


def ade_fde(mu, target):
    distance = torch.linalg.vector_norm(
        mu - target,
        dim=-1,
    )

    ade = distance.mean()
    fde = distance[:, -1].mean()

    return ade, fde


def mahalanobis_squared(pred, target, eps=1e-6):
    """
    Squared Mahalanobis distance for the same correlated
    bivariate Gaussian parameterization used by gaussian_nll.
    """

    mu = pred["mu"]

    std = pred["std"].clamp_min(eps)

    rho = (
        pred["rho"]
        .squeeze(-1)
        .clamp(
            min=-1.0 + eps,
            max=1.0 - eps,
        )
    )

    dx = target[..., 0] - mu[..., 0]
    dy = target[..., 1] - mu[..., 1]

    sx = std[..., 0]
    sy = std[..., 1]

    nx = dx / sx
    ny = dy / sy

    one_minus_rho2 = (
        1.0 - rho * rho
    ).clamp_min(eps)

    d2 = (
        nx * nx
        + ny * ny
        - 2.0 * rho * nx * ny
    ) / one_minus_rho2

    return d2


def evaluate_candidate(cfg, dataset):

    checkpoint = (
        CHECKPOINT_DIR
        / f"{cfg['name']}.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    print()
    print("=" * 70)
    print("MODEL:", cfg["name"])
    print("CHECKPOINT:", checkpoint)
    print("=" * 70)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch"],
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = GaussianGRUPredictor(
        input_size=8,
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        dropout=cfg["dropout"],
        future=10,
    ).to(DEVICE)

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    nll_sum = 0.0
    ade_sum = 0.0
    fde_sum = 0.0

    batch_samples = 0

    coverage_counts = {
        p: 0
        for p in LEVELS
    }

    total_points = 0

    mean_std_x_sum = 0.0
    mean_std_y_sum = 0.0
    mean_abs_rho_sum = 0.0
    uncertainty_points = 0

    with torch.no_grad():

        for batch_index, (x, y) in enumerate(loader, start=1):

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            pred = model(x)

            nll = gaussian_nll(
                pred,
                y,
            )

            ade, fde = ade_fde(
                pred["mu"],
                y,
            )

            n = x.shape[0]

            nll_sum += nll.item() * n
            ade_sum += ade.item() * n
            fde_sum += fde.item() * n
            batch_samples += n

            d2 = mahalanobis_squared(
                pred,
                y,
            )

            total_points += d2.numel()

            for p in LEVELS:

                threshold = (
                    chi2_df2_quantile(p)
                )

                coverage_counts[p] += (
                    (d2 <= threshold)
                    .sum()
                    .item()
                )

            std = pred["std"]
            rho = pred["rho"].squeeze(-1)

            points = std[..., 0].numel()

            mean_std_x_sum += (
                std[..., 0].sum().item()
            )

            mean_std_y_sum += (
                std[..., 1].sum().item()
            )

            mean_abs_rho_sum += (
                rho.abs().sum().item()
            )

            uncertainty_points += points

            if batch_index % 100 == 0:
                print(
                    f"batch={batch_index}/{len(loader)}",
                    flush=True,
                )

    coverage = {
        f"{int(p*100)}": (
            coverage_counts[p]
            / total_points
        )
        for p in LEVELS
    }

    calibration_errors = {
        f"{int(p*100)}": (
            coverage[f"{int(p*100)}"]
            - p
        )
        for p in LEVELS
    }

    mean_abs_calibration_error = (
        sum(
            abs(v)
            for v in calibration_errors.values()
        )
        / len(calibration_errors)
    )

    result = {
        "label": cfg["label"],
        "name": cfg["name"],
        "checkpoint": str(checkpoint),
        "samples": batch_samples,
        "trajectory_points": total_points,

        "NLL": nll_sum / batch_samples,
        "ADE": ade_sum / batch_samples,
        "FDE": fde_sum / batch_samples,

        "coverage": coverage,
        "coverage_error": calibration_errors,

        "mean_absolute_calibration_error":
            mean_abs_calibration_error,

        "mean_std_x":
            mean_std_x_sum / uncertainty_points,

        "mean_std_y":
            mean_std_y_sum / uncertainty_points,

        "mean_abs_rho":
            mean_abs_rho_sum / uncertainty_points,
    }

    return result


def main():

    print("=" * 70)
    print("STAGE 4 GAUSSIAN CALIBRATION COMPARISON")
    print("=" * 70)

    print("DEVICE:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()
    print("Loading WOMD validation dataset...")

    dataset = WOMDGRUDataset(
        VAL_DATA,
        max_files=MAX_VAL_FILES,
        actor_type=TARGET_CLASSES,
    )

    print(
        "Validation samples:",
        len(dataset),
    )

    results = []

    for cfg in CANDIDATES:

        result = evaluate_candidate(
            cfg,
            dataset,
        )

        results.append(result)

        print()
        print("NLL =", result["NLL"])
        print("ADE =", result["ADE"])
        print("FDE =", result["FDE"])

        print("Coverage:")
        for level, value in result["coverage"].items():
            print(
                f"  {level}% -> {100*value:.2f}%"
            )

        print(
            "Mean abs calibration error =",
            result[
                "mean_absolute_calibration_error"
            ],
        )

    # ---------------------------------------------------------
    # Rankings
    # ---------------------------------------------------------

    best_nll = min(
        results,
        key=lambda r: r["NLL"],
    )

    best_ade = min(
        results,
        key=lambda r: r["ADE"],
    )

    best_fde = min(
        results,
        key=lambda r: r["FDE"],
    )

    best_calibrated = min(
        results,
        key=lambda r:
            r["mean_absolute_calibration_error"],
    )

    report = {
        "evaluation": (
            "real_WOMD_validation_"
            "trained_gaussian_GRU"
        ),
        "max_validation_files":
            MAX_VAL_FILES,
        "validation_samples":
            len(dataset),
        "confidence_levels":
            LEVELS,
        "results":
            results,
        "selection": {
            "best_NLL":
                best_nll["name"],
            "best_ADE":
                best_ade["name"],
            "best_FDE":
                best_fde["name"],
            "best_calibrated":
                best_calibrated["name"],
        },
    }

    output = (
        REPORT_DIR
        / "gaussian_calibration_comparison.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("FINAL RANKING")
    print("=" * 70)

    print(
        "BEST NLL:",
        best_nll["name"],
    )

    print(
        "BEST ADE:",
        best_ade["name"],
    )

    print(
        "BEST FDE:",
        best_fde["name"],
    )

    print(
        "BEST CALIBRATED:",
        best_calibrated["name"],
    )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
