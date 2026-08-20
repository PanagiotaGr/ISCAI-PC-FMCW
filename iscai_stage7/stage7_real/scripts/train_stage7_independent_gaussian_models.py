from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import (
    WOMDGRUDataset,
)
from iscai_stage4.training.gru_model import (
    GaussianGRUPredictor,
)
from iscai_stage4.training.gaussian_loss import (
    gaussian_nll,
)


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

OUT_DIR = Path(
    "stage7_real/checkpoints/section47"
)

REPORT = Path(
    "stage7_real/reports/section47/"
    "independent_gaussian_training.json"
)

TARGET_CLASSES = [
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST",
]

MAX_TRAIN_FILES = 200
MAX_VAL_FILES = 150

HISTORY = 10
FUTURE = 10

INPUT_SIZE = 8
HIDDEN = 128
LAYERS = 1
DROPOUT = 0.0

BATCH = 32
LR = 1e-3

MAX_EPOCHS = 30
PATIENCE = 5

NUM_WORKERS = 8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)


MODELS = [
    {
        "name":
            "independent_comm_gaussian",

        "seed":
            142,
    },
    {
        "name":
            "independent_adb_gaussian",

        "seed":
            242,
    },
]


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model,
    loader,
):

    model.eval()

    loss_sum = 0.0
    count = 0

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

            loss = gaussian_nll(
                pred,
                y,
            )

            n = x.shape[0]

            loss_sum += (
                loss.item()
                *
                n
            )

            count += n

    return (
        loss_sum
        /
        max(
            count,
            1,
        )
    )


def train_one(
    cfg,
    train_loader,
    val_loader,
):

    print()
    print("=" * 84)
    print("MODEL:", cfg["name"])
    print("SEED :", cfg["seed"])
    print("=" * 84)

    set_seed(
        cfg["seed"]
    )

    model = GaussianGRUPredictor(
        input_size=INPUT_SIZE,
        hidden=HIDDEN,
        layers=LAYERS,
        dropout=DROPOUT,
        future=FUTURE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    checkpoint = (
        OUT_DIR
        /
        f"{cfg['name']}.pt"
    )

    best_val = float("inf")
    best_epoch = None

    patience_left = PATIENCE

    history = []

    start = time.perf_counter()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        train_sum = 0.0
        train_count = 0

        for x, y in train_loader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(x)

            loss = gaussian_nll(
                pred,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            n = x.shape[0]

            train_sum += (
                loss.item()
                *
                n
            )

            train_count += n

        train_nll = (
            train_sum
            /
            max(
                train_count,
                1,
            )
        )

        val_nll = evaluate(
            model,
            val_loader,
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_nll":
                    train_nll,

                "val_nll":
                    val_nll,
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"train_nll={train_nll:.6f} "
            f"val_nll={val_nll:.6f}"
        )

        if val_nll < best_val:

            best_val = val_nll
            best_epoch = epoch

            patience_left = (
                PATIENCE
            )

            torch.save(
                model.state_dict(),
                checkpoint,
            )

        else:

            patience_left -= 1

            if patience_left <= 0:

                print(
                    "early stopping"
                )

                break

    runtime = (
        time.perf_counter()
        -
        start
    )

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        state
    )

    final_val = evaluate(
        model,
        val_loader,
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    result = {
        "name":
            cfg["name"],

        "seed":
            cfg["seed"],

        "checkpoint":
            str(checkpoint),

        "architecture": {
            "input_size":
                INPUT_SIZE,

            "history":
                HISTORY,

            "future":
                FUTURE,

            "hidden":
                HIDDEN,

            "layers":
                LAYERS,

            "dropout":
                DROPOUT,

            "parameters":
                params,
        },

        "optimization": {
            "learning_rate":
                LR,

            "batch_size":
                BATCH,

            "max_epochs":
                MAX_EPOCHS,

            "early_stopping_patience":
                PATIENCE,
        },

        "best_epoch":
            best_epoch,

        "best_val_nll":
            best_val,

        "reloaded_val_nll":
            final_val,

        "runtime_seconds":
            runtime,

        "history":
            history,
    }

    print()
    print(
        "best epoch =",
        best_epoch
    )

    print(
        "best val NLL =",
        best_val
    )

    print(
        "parameters =",
        params
    )

    print(
        "checkpoint =",
        checkpoint
    )

    return result


def main():

    print("=" * 84)
    print("STAGE 7 SECTION 47 — INDEPENDENT GAUSSIAN MODELS")
    print("=" * 84)

    print("device =", DEVICE)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Exact Stage-4 train/validation protocol
    # --------------------------------------------------------

    print()
    print("loading train dataset")

    train_ds = WOMDGRUDataset(
        TRAIN_DATA,
        max_files=MAX_TRAIN_FILES,
        history=HISTORY,
        future=FUTURE,
        actor_type=TARGET_CLASSES,
    )

    print(
        "train samples =",
        len(train_ds)
    )

    print()
    print("loading validation dataset")

    val_ds = WOMDGRUDataset(
        VAL_DATA,
        max_files=MAX_VAL_FILES,
        history=HISTORY,
        future=FUTURE,
        actor_type=TARGET_CLASSES,
    )

    print(
        "validation samples =",
        len(val_ds)
    )

    # Generator makes batch order reproducible.
    loader_generator = (
        torch.Generator()
    )

    loader_generator.manual_seed(
        42
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        pin_memory=(
            DEVICE.type
            ==
            "cuda"
        ),
        num_workers=NUM_WORKERS,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        generator=
            loader_generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        pin_memory=(
            DEVICE.type
            ==
            "cuda"
        ),
        num_workers=NUM_WORKERS,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    results = []

    for cfg in MODELS:

        results.append(
            train_one(
                cfg,
                train_loader,
                val_loader,
            )
        )

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "experiment":
            "section47_independent_models",

        "comparison_role":
            (
                "Independent trajectory predictors "
                "for communication and ADB branches"
            ),

        "reference_shared_architecture":
            (
                "ALL_gaussian_h128_l1_"
                "d0.0_lr0.001_b32"
            ),

        "data": {
            "train_root":
                TRAIN_DATA,

            "validation_root":
                VAL_DATA,

            "max_train_files":
                MAX_TRAIN_FILES,

            "max_validation_files":
                MAX_VAL_FILES,

            "train_samples":
                len(train_ds),

            "validation_samples":
                len(val_ds),

            "classes":
                TARGET_CLASSES,

            "future_used_as_input":
                False,
        },

        "models":
            results,

        "scientific_scope": (
            "Two independently initialized and "
            "trained Gaussian trajectory models "
            "use the same architecture, optimizer, "
            "WOMD train/validation roots and target "
            "definition as the shared Stage-4 "
            "reference. This isolates the model-"
            "sharing factor rather than changing "
            "model capacity or data."
        ),
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("=" * 84)
    print("SECTION 47 INDEPENDENT TRAINING COMPLETE")
    print("=" * 84)

    for x in results:

        print(
            x["name"],
            "best_val_nll=",
            x[
                "best_val_nll"
            ],
            "params=",
            x[
                "architecture"
            ][
                "parameters"
            ],
        )

    print()
    print("Saved:", REPORT)

    print()
    print(
        "STAGE 7 INDEPENDENT MODELS = PASS"
    )


if __name__ == "__main__":
    main()
