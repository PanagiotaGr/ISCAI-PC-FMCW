import json
import itertools
import random
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import (
    GRUPredictor,
    GaussianGRUPredictor,
)
from iscai_stage4.training.gaussian_loss import gaussian_nll


# ============================================================
# CONFIGURATION
# ============================================================

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

REPORT = Path(
    "reports/stage4_auto_cuda"
)

CHECKPOINT = Path(
    "checkpoints/stage4_auto_cuda"
)

RESULTS_FILE = (
    REPORT / "all_results.json"
)

BEST_FILE = (
    REPORT / "best_models.json"
)

MAX_EPOCHS = 1000

PATIENCE = 50

MAX_TRAIN_FILES = 200

MAX_VAL_FILES = 150

NUM_WORKERS = 8


TARGET_CLASSES = [
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST",
]

CLASSES = [
    "ALL",
]


# ============================================================
# REDUCED SCIENTIFIC SEARCH GRID
# ============================================================
#
# We keep:
#
#   model:
#       deterministic / gaussian
#
#   hidden:
#       64 / 128
#
#   layers:
#       1 / 2
#
#   learning rate:
#       1e-3 / 1e-4
#
#   batch:
#       32 / 64
#
# Dropout is handled separately:
#
#   layers = 1 -> dropout = 0.0 only
#   layers = 2 -> dropout = 0.0 or 0.2
#
# Reason:
# PyTorch recurrent dropout acts between recurrent layers.
# Therefore dropout experiments for a one-layer GRU are redundant.
#
# Number of configurations:
#
# 2 models
# x 2 hidden sizes
# x 2 learning rates
# x 2 batches
# x 3 valid (layers, dropout) combinations
#
# = 48 experiments
# ============================================================


MODELS = [
    "deterministic",
    "gaussian",
]

HIDDEN_SIZES = [
    64,
    128,
]

LEARNING_RATES = [
    1e-3,
    1e-4,
]

BATCH_SIZES = [
    32,
    64,
]

LAYER_DROPOUT = [
    (1, 0.0),
    (2, 0.0),
    (2, 0.2),
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "DEVICE:",
    DEVICE,
    flush=True,
)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DIRECTORIES
# ============================================================

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# METRICS
# ============================================================

def ade_fde(
    pred,
    target,
):

    distance = torch.norm(
        pred - target,
        dim=-1,
    )

    ade = (
        distance
        .mean()
        .item()
    )

    fde = (
        distance[:, -1]
        .mean()
        .item()
    )

    return ade, fde


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    gaussian,
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
            set_to_none=True
        )


        with torch.cuda.amp.autocast(
            enabled=DEVICE.type == "cuda"
        ):

            output = model(x)


            if gaussian:

                loss = gaussian_nll(
                    output,
                    y,
                )

            else:

                loss = (
                    torch.nn.functional
                    .mse_loss(
                        output,
                        y,
                    )
                )


        scaler.scale(
            loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )


        scaler.step(
            optimizer
        )


        scaler.update()


        total_loss += (
            loss.item()
        )


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


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    gaussian,
):

    model.eval()

    losses = []

    ades = []

    fdes = []


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


            output = model(x)


            if gaussian:

                loss = gaussian_nll(
                    output,
                    y,
                )

                prediction = (
                    output["mu"]
                )

            else:

                loss = (
                    torch.nn.functional
                    .mse_loss(
                        output,
                        y,
                    )
                )

                prediction = output


            ade, fde = ade_fde(
                prediction,
                y,
            )


            losses.append(
                loss.item()
            )

            ades.append(
                ade
            )

            fdes.append(
                fde
            )


    return {

        "loss": (
            sum(losses)
            /
            len(losses)
        ),

        "ADE": (
            sum(ades)
            /
            len(ades)
        ),

        "FDE": (
            sum(fdes)
            /
            len(fdes)
        ),
    }


# ============================================================
# EXPERIMENT NAME
# ============================================================

def experiment_name(
    actor,
    cfg,
):

    return (
        f"{actor}_"
        f"{cfg['model']}_"
        f"h{cfg['hidden']}_"
        f"l{cfg['layers']}_"
        f"d{cfg['dropout']}_"
        f"lr{cfg['lr']}_"
        f"b{cfg['batch']}"
    )


# ============================================================
# BUILD REDUCED CONFIGURATIONS
# ============================================================

def build_configs():

    configs = []


    for (
        model,
        hidden,
        lr,
        batch,
        layer_dropout,
    ) in itertools.product(

        MODELS,
        HIDDEN_SIZES,
        LEARNING_RATES,
        BATCH_SIZES,
        LAYER_DROPOUT,

    ):

        layers, dropout = (
            layer_dropout
        )


        configs.append(
            {
                "model": model,
                "hidden": hidden,
                "layers": layers,
                "dropout": dropout,
                "lr": lr,
                "batch": batch,
            }
        )


    return configs


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results():

    if not RESULTS_FILE.exists():

        return []


    with open(
        RESULTS_FILE,
        "r",
    ) as f:

        results = json.load(f)


    if not isinstance(
        results,
        list,
    ):

        raise RuntimeError(
            "Existing all_results.json "
            "is not a JSON list."
        )


    return results


# ============================================================
# ATOMIC JSON SAVE
# ============================================================

def atomic_save_json(
    data,
    destination,
):

    temporary = Path(
        str(destination)
        + ".tmp"
    )


    with open(
        temporary,
        "w",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


    temporary.replace(
        destination
    )


# ============================================================
# BEST MODELS
# ============================================================

def calculate_best_models(
    results,
):

    best = {}


    for result in results:

        if (
            not isinstance(result, dict)
            or "class" not in result
            or "metrics" not in result
        ):

            continue


        actor = result["class"]


        if (
            actor not in best
            or
            result["metrics"]["loss"]
            <
            best[actor]["metrics"]["loss"]
        ):

            best[actor] = result


    return best


# ============================================================
# MAIN
# ============================================================

def main():

    configs = build_configs()


    print(
        "\n========================================"
    )

    print(
        "STAGE 4 REDUCED CUDA SEARCH"
    )

    print(
        "TOTAL REDUCED CONFIGS:",
        len(configs),
    )

    print(
        "========================================\n"
    )


    # --------------------------------------------------------
    # Load all previous results.
    #
    # This preserves experiments from the original 216-grid,
    # including dropout=0.5 experiments.
    # --------------------------------------------------------

    all_results = (
        load_existing_results()
    )


    completed_names = {

        result["name"]

        for result in all_results

        if (
            isinstance(
                result,
                dict,
            )
            and
            "name" in result
        )
    }


    planned_names = {

        experiment_name(
            actor,
            cfg,
        )

        for actor in CLASSES

        for cfg in configs
    }


    already_completed = (
        completed_names
        &
        planned_names
    )


    remaining_names = (
        planned_names
        -
        completed_names
    )


    print(
        "OLD RESULTS FOUND:",
        len(all_results),
    )

    print(
        "REDUCED EXPERIMENTS "
        "ALREADY COMPLETED:",
        len(already_completed),
    )

    print(
        "REDUCED EXPERIMENTS "
        "REMAINING:",
        len(remaining_names),
    )

    print()


    # ========================================================
    # CLASS LOOP
    # ========================================================

    for actor in CLASSES:


        print(
            "\n========================================"
        )

        print(
            "CLASS:",
            actor,
        )

        print(
            "========================================"
        )


        train_dataset = (
            WOMDGRUDataset(
                TRAIN_DATA,
                max_files=MAX_TRAIN_FILES,
                actor_type=TARGET_CLASSES,
            )
        )


        validation_dataset = (
            WOMDGRUDataset(
                VAL_DATA,
                max_files=MAX_VAL_FILES,
                actor_type=TARGET_CLASSES,
            )
        )


        # ====================================================
        # CONFIG LOOP
        # ====================================================

        for config_index, cfg in enumerate(
            configs,
            start=1,
        ):


            name = experiment_name(
                actor,
                cfg,
            )


            # ------------------------------------------------
            # RESUME:
            # Do not retrain experiments that are already
            # recorded as successfully completed.
            # ------------------------------------------------

            if name in completed_names:

                print(
                    f"\nSKIP "
                    f"[{config_index}/{len(configs)}] "
                    f"{name} "
                    f"(already completed)",
                    flush=True,
                )

                continue


            completed_reduced_count = len(
                completed_names
                &
                planned_names
            )


            remaining_count = (
                len(planned_names)
                -
                completed_reduced_count
            )


            print(
                "\n========================================",
                flush=True,
            )

            print(
                f"RUN "
                f"[{config_index}/{len(configs)}]",
                flush=True,
            )

            print(
                "NAME:",
                name,
                flush=True,
            )

            print(
                "CONFIG:",
                cfg,
                flush=True,
            )

            print(
                "COMPLETED:",
                completed_reduced_count,
                "/",
                len(planned_names),
                flush=True,
            )

            print(
                "REMAINING:",
                remaining_count,
                flush=True,
            )

            print(
                "========================================",
                flush=True,
            )


            # =================================================
            # DATA LOADERS
            # =================================================

            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg["batch"],
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
            )


            validation_loader = DataLoader(
                validation_dataset,
                batch_size=cfg["batch"],
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


            gaussian = (
                cfg["model"]
                ==
                "gaussian"
            )


            # =================================================
            # MODEL
            # =================================================

            if gaussian:

                model = GaussianGRUPredictor(
                    hidden=cfg["hidden"],
                    layers=cfg["layers"],
                    dropout=cfg["dropout"],
                )

            else:

                model = GRUPredictor(
                    input_size=8,
                    hidden=cfg["hidden"],
                    layers=cfg["layers"],
                    dropout=cfg["dropout"],
                )


            model = model.to(
                DEVICE
            )


            optimizer = (
                torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg["lr"],
                )
            )


            scaler = (
                torch.cuda.amp.GradScaler(
                    enabled=(
                        DEVICE.type
                        ==
                        "cuda"
                    )
                )
            )


            # =================================================
            # TRAINING STATE
            # =================================================

            best_loss = float(
                "inf"
            )

            best_metrics = None

            best_epoch = None

            patience_counter = 0

            start_time = (
                time.time()
            )


            # =================================================
            # EPOCH LOOP
            # =================================================

            for epoch in range(
                1,
                MAX_EPOCHS + 1,
            ):


                train_loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    gaussian,
                )


                validation_metrics = validate(
                    model,
                    validation_loader,
                    gaussian,
                )


                current_validation_loss = (
                    validation_metrics[
                        "loss"
                    ]
                )


                improved = (
                    current_validation_loss
                    <
                    best_loss
                )


                if improved:

                    best_loss = (
                        current_validation_loss
                    )

                    best_metrics = deepcopy(
                        validation_metrics
                    )

                    best_epoch = epoch

                    patience_counter = 0


                    torch.save(
                        model.state_dict(),
                        CHECKPOINT
                        /
                        f"{name}.pt",
                    )


                else:

                    patience_counter += 1


                print(
                    f"epoch={epoch} "
                    f"train_loss={train_loss:.6f} "
                    f"val_loss="
                    f"{current_validation_loss:.6f} "
                    f"best="
                    f"{best_loss:.6f} "
                    f"patience="
                    f"{patience_counter}/{PATIENCE}",
                    flush=True,
                )


                if (
                    patience_counter
                    >=
                    PATIENCE
                ):

                    print(
                        f"EARLY STOP: "
                        f"{name} "
                        f"at epoch {epoch}",
                        flush=True,
                    )

                    break


            # =================================================
            # EXPERIMENT FINISHED
            # =================================================

            elapsed = (
                time.time()
                -
                start_time
            )


            if best_metrics is None:

                raise RuntimeError(
                    f"No valid best metrics "
                    f"were produced for {name}"
                )


            result = {

                "name": name,

                "class": actor,

                "config": cfg,

                # IMPORTANT:
                # Metrics correspond to the
                # BEST checkpoint, not the
                # final training epoch.
                "metrics": best_metrics,

                "best_epoch": best_epoch,

                "epochs_run": epoch,

                "time": elapsed,
            }


            all_results.append(
                result
            )


            completed_names.add(
                name
            )


            # ------------------------------------------------
            # Save immediately.
            #
            # Therefore if the process is interrupted later,
            # every fully completed experiment is retained.
            # ------------------------------------------------

            atomic_save_json(
                all_results,
                RESULTS_FILE,
            )


            best_models = (
                calculate_best_models(
                    all_results
                )
            )


            atomic_save_json(
                best_models,
                BEST_FILE,
            )


            completed_reduced_count = len(
                completed_names
                &
                planned_names
            )


            remaining_count = (
                len(planned_names)
                -
                completed_reduced_count
            )


            print(
                "\n----------------------------------------",
                flush=True,
            )

            print(
                "EXPERIMENT COMPLETED:",
                name,
                flush=True,
            )

            print(
                "BEST EPOCH:",
                best_epoch,
                flush=True,
            )

            print(
                "BEST LOSS:",
                best_metrics["loss"],
                flush=True,
            )

            print(
                "BEST ADE:",
                best_metrics["ADE"],
                flush=True,
            )

            print(
                "BEST FDE:",
                best_metrics["FDE"],
                flush=True,
            )

            print(
                "TIME HOURS:",
                elapsed / 3600.0,
                flush=True,
            )

            print(
                "REDUCED PROGRESS:",
                completed_reduced_count,
                "/",
                len(planned_names),
                flush=True,
            )

            print(
                "REMAINING:",
                remaining_count,
                flush=True,
            )

            print(
                "----------------------------------------\n",
                flush=True,
            )


            # Free references before the
            # next architecture.
            del model

            del optimizer

            del scaler

            del train_loader

            del validation_loader


            if DEVICE.type == "cuda":

                torch.cuda.empty_cache()


    # ========================================================
    # FINAL SAVE
    # ========================================================

    best_models = (
        calculate_best_models(
            all_results
        )
    )


    atomic_save_json(
        all_results,
        RESULTS_FILE,
    )


    atomic_save_json(
        best_models,
        BEST_FILE,
    )


    completed_reduced_count = len(
        completed_names
        &
        planned_names
    )


    print(
        "\n========================================",
        flush=True,
    )

    print(
        "REDUCED SEARCH FINISHED",
        flush=True,
    )

    print(
        "REDUCED EXPERIMENTS:",
        len(planned_names),
        flush=True,
    )

    print(
        "COMPLETED:",
        completed_reduced_count,
        flush=True,
    )

    print(
        "REMAINING:",
        (
            len(planned_names)
            -
            completed_reduced_count
        ),
        flush=True,
    )

    print(
        "TOTAL RESULTS RETAINED:",
        len(all_results),
        flush=True,
    )

    print(
        "========================================",
        flush=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
