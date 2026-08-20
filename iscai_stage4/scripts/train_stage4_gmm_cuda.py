from pathlib import Path
import json
import random
import time

import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gmm_gru_model import GMMGRUPredictor
from iscai_stage4.training.gmm_loss import gmm_nll


SEED = 42

TRAIN_DATA = (
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "training/motion"
)

VAL_DATA = (
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)

MAX_TRAIN_FILES = 200
MAX_VAL_FILES = 150

BATCH_SIZE = 32
NUM_WORKERS = 4

HIDDEN = 128
LAYERS = 1
DROPOUT = 0.0
FUTURE = 10
MODES = 5

LR = 1e-3

MAX_EPOCHS = 1000
PATIENCE = 50

CHECKPOINT = Path(
    "checkpoints/stage4_auto_cuda/"
    "ALL_gmm_h128_l1_d0.0_lr0.001_b32_k5.pt"
)

REPORT = Path(
    "reports/stage4_auto_cuda/"
    "gmm_real_training_result.json"
)

CHECKPOINT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def minade_minfde(mu, target):
    """
    mu:
        [B,K,T,2]

    target:
        [B,T,2]
    """

    target = target.unsqueeze(1)

    distance = torch.linalg.vector_norm(
        mu - target,
        dim=-1,
    )

    ade_per_mode = distance.mean(
        dim=-1,
    )

    fde_per_mode = distance[..., -1]

    minade = (
        ade_per_mode
        .min(dim=1)
        .values
        .mean()
        .item()
    )

    minfde = (
        fde_per_mode
        .min(dim=1)
        .values
        .mean()
        .item()
    )

    return minade, minfde


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
):

    model.train()

    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        with torch.cuda.amp.autocast(
            enabled=(DEVICE.type == "cuda"),
        ):

            pred = model(x)

            loss = gmm_nll(
                pred,
                y,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(
                f"batch "
                f"{batch_idx}/{len(loader)} "
                f"loss={loss.item():.6f}",
                flush=True,
            )

    return (
        total_loss
        /
        len(loader)
    )


def validate(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0
    total_minade = 0.0
    total_minfde = 0.0
    batches = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            pred = model(x)

            loss = gmm_nll(
                pred,
                y,
            )

            minade, minfde = (
                minade_minfde(
                    pred["mu"],
                    y,
                )
            )

            total_loss += loss.item()
            total_minade += minade
            total_minfde += minfde
            batches += 1

    return {
        "loss":
            total_loss / batches,

        "minADE":
            total_minade / batches,

        "minFDE":
            total_minfde / batches,
    }


def main():

    print("=" * 70)
    print("STAGE 4 REAL WOMD LEARNED GMM")
    print("=" * 70)

    print("DEVICE:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()
    print("Loading training dataset...")

    train_dataset = WOMDGRUDataset(
        TRAIN_DATA,
        max_files=MAX_TRAIN_FILES,
        actor_type=[
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
        ],
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print()
    print("Loading validation dataset...")

    val_dataset = WOMDGRUDataset(
        VAL_DATA,
        max_files=MAX_VAL_FILES,
        actor_type=[
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
        ],
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = GMMGRUPredictor(
        input_size=8,
        hidden=HIDDEN,
        layers=LAYERS,
        dropout=DROPOUT,
        future=FUTURE,
        modes=MODES,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(DEVICE.type == "cuda"),
    )

    best_loss = float("inf")
    best_epoch = None
    best_metrics = None

    patience_counter = 0

    start_time = time.time()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
        )

        metrics = validate(
            model,
            val_loader,
        )

        val_loss = metrics["loss"]

        improved = (
            val_loss < best_loss
        )

        if improved:

            best_loss = val_loss
            best_epoch = epoch
            best_metrics = dict(metrics)

            patience_counter = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT,
            )

        else:

            patience_counter += 1

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"minADE={metrics['minADE']:.6f} "
            f"minFDE={metrics['minFDE']:.6f} "
            f"best={best_loss:.6f} "
            f"patience={patience_counter}/{PATIENCE}",
            flush=True,
        )

        if patience_counter >= PATIENCE:
            break

    elapsed = (
        time.time()
        -
        start_time
    )

    result = {
        "name":
            "ALL_gmm_h128_l1_d0.0_lr0.001_b32_k5",

        "data_source":
            "real_WOMD",

        "modes":
            MODES,

        "hidden":
            HIDDEN,

        "layers":
            LAYERS,

        "dropout":
            DROPOUT,

        "lr":
            LR,

        "batch":
            BATCH_SIZE,

        "best_epoch":
            best_epoch,

        "epochs_run":
            epoch,

        "metrics":
            best_metrics,

        "time_seconds":
            elapsed,

        "checkpoint":
            str(CHECKPOINT),

        "future_used_as_input":
            False,
    }

    REPORT.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("GMM TRAINING COMPLETE")
    print("=" * 70)

    print(
        "BEST EPOCH:",
        best_epoch,
    )

    print(
        "BEST METRICS:",
        best_metrics,
    )

    print(
        "CHECKPOINT:",
        CHECKPOINT,
    )

    print(
        "REPORT:",
        REPORT,
    )


if __name__ == "__main__":
    main()
