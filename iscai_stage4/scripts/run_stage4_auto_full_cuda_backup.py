import json
import itertools
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import (
    GRUPredictor,
    GaussianGRUPredictor
)

from iscai_stage4.training.gaussian_loss import gaussian_nll


# ==========================
# DEVICE
# ==========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==========================
# SEED
# ==========================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



# ==========================
# PATHS
# ==========================

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


REPORT.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT.mkdir(
    parents=True,
    exist_ok=True
)



# ==========================
# EXPERIMENT GRID
# ==========================

GRID = {

    "model":[
        "deterministic",
        "gaussian"
    ],

    "hidden":[
        64,
        128,
        256
    ],

    "layers":[
        1,
        2,
        3
    ],

    "dropout":[
        0.0,
        0.2,
        0.5
    ],

    "lr":[
        1e-3,
        1e-4
    ],

    "batch":[
        32,
        64
    ]
}


CLASSES=[
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST"
]


MAX_EPOCHS = 1000

PATIENCE = 50


# None = all
SEARCH_LIMIT = 64



# ==========================
# METRICS
# ==========================


def ade_fde(
    pred,
    target
):

    d=torch.norm(
        pred-target,
        dim=-1
    )

    return (
        d.mean().item(),
        d[:,-1].mean().item()
    )



# ==========================
# TRAIN
# ==========================


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    gaussian
):

    model.train()

    total=0


    for x,y in loader:

        x=x.to(
            DEVICE,
            non_blocking=True
        )

        y=y.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        with torch.cuda.amp.autocast(
            enabled=DEVICE.type=="cuda"
        ):

            out=model(x)


            if gaussian:

                loss=gaussian_nll(
                    out,
                    y
                )

            else:

                loss=torch.nn.functional.mse_loss(
                    out,
                    y
                )


        scaler.scale(
            loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )


        scaler.step(
            optimizer
        )


        scaler.update()


        total += loss.item()


    return total/len(loader)



# ==========================
# VALIDATION
# ==========================


def validate(
    model,
    loader,
    gaussian
):

    model.eval()

    losses=[]
    ades=[]
    fdes=[]


    with torch.no_grad():

        for x,y in loader:

            x=x.to(DEVICE)
            y=y.to(DEVICE)


            out=model(x)


            if gaussian:

                loss=gaussian_nll(
                    out,
                    y
                )

                pred=out["mu"]

            else:

                loss=torch.nn.functional.mse_loss(
                    out,
                    y
                )

                pred=out


            ade,fde=ade_fde(
                pred,
                y
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

        "loss":
            sum(losses)/len(losses),

        "ADE":
            sum(ades)/len(ades),

        "FDE":
            sum(fdes)/len(fdes)
    }



# ==========================
# MAIN
# ==========================


configs=[]


keys=list(GRID.keys())


for values in itertools.product(
    *[
        GRID[k]
        for k in keys
    ]
):

    configs.append(
        dict(zip(keys,values))
    )


if SEARCH_LIMIT:

    configs=configs[:SEARCH_LIMIT]


print(
    "EXPERIMENTS:",
    len(configs)
)



all_results=[]



for actor in CLASSES:


    print(
        "\nCLASS:",
        actor
    )


    train_ds=WOMDGRUDataset(
        TRAIN_DATA,
        max_files=200,
        actor_type=actor
    )


    val_ds=WOMDGRUDataset(
        VAL_DATA,
        max_files=150,
        actor_type=actor
    )


    for idx,cfg in enumerate(configs):


        name=(
            actor+"_"+
            cfg["model"]+
            "_h"+str(cfg["hidden"])+
            "_l"+str(cfg["layers"])
        )


        print(
            "\nRUN",
            idx+1,
            "/",
            len(configs),
            name
        )


        train_loader=DataLoader(
            train_ds,
            batch_size=cfg["batch"],
            shuffle=True,
            pin_memory=True
        )


        val_loader=DataLoader(
            val_ds,
            batch_size=cfg["batch"],
            shuffle=False,
            pin_memory=True
        )


        gaussian=(
            cfg["model"]=="gaussian"
        )


        if gaussian:

            model=GaussianGRUPredictor(
                hidden=cfg["hidden"],
                layers=cfg["layers"],
                dropout=cfg["dropout"]
            )

        else:

            model=GRUPredictor(
                input_size=8,
                hidden=cfg["hidden"],
                layers=cfg["layers"],
                dropout=cfg["dropout"]
            )


        model.to(DEVICE)


        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"]
        )


        scaler=torch.cuda.amp.GradScaler(
            enabled=DEVICE.type=="cuda"
        )


        best=float("inf")
        wait=0


        start=time.time()


        for epoch in range(MAX_EPOCHS):


            train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                gaussian
            )


            metrics=validate(
                model,
                val_loader,
                gaussian
            )


            if metrics["loss"] < best:

                best=metrics["loss"]
                wait=0


                torch.save(
                    model.state_dict(),
                    CHECKPOINT /
                    f"{name}.pt"
                )


            else:

                wait+=1


            if wait >= PATIENCE:
                break



        all_results.append(
            {
                "name":name,
                "class":actor,
                "config":cfg,
                "metrics":metrics,
                "time":time.time()-start
            }
        )


        with open(
            REPORT/"all_results.json",
            "w"
        ) as f:

            json.dump(
                all_results,
                f,
                indent=2
            )



best={}


for r in all_results:

    c=r["class"]

    if (
        c not in best
        or r["metrics"]["loss"]
        <
        best[c]["metrics"]["loss"]
    ):

        best[c]=r



with open(
    REPORT/"best_models.json",
    "w"
) as f:

    json.dump(
        best,
        f,
        indent=2
    )


print(
    "\nAUTO CUDA SEARCH FINISHED"
)
