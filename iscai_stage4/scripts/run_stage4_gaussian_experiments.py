import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import GaussianGRUPredictor
from iscai_stage4.training.gaussian_loss import gaussian_nll


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE:", DEVICE)


SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


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


OUT = Path(
    "reports/stage4_gaussian"
)

CKPT = Path(
    "checkpoints/stage4_gaussian"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CKPT.mkdir(
    parents=True,
    exist_ok=True
)


def ade_fde(pred, target):

    dist = torch.norm(
        pred-target,
        dim=-1
    )

    ade = dist.mean()

    fde = dist[:,-1].mean()

    return ade.item(), fde.item()



def train_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    total=0

    for x,y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        pred=model(x)

        loss=gaussian_nll(
            pred,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total += loss.item()


    return total/len(loader)



def validate(
    model,
    loader
):

    model.eval()

    total=0
    ades=[]
    fdes=[]

    with torch.no_grad():

        for x,y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

            pred=model(x)

            loss=gaussian_nll(
                pred,
                y
            )

            total += loss.item()


            ade,fde = ade_fde(
                pred["mu"],
                y
            )

            ades.append(ade)
            fdes.append(fde)


    return (
        total/len(loader),
        sum(ades)/len(ades),
        sum(fdes)/len(fdes)
    )



for actor in [
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST"
]:

    print("\n================")
    print(actor)
    print("================")


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


    print(
        "TRAIN",
        len(train_ds),
        "VAL",
        len(val_ds)
    )


    train_loader=DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True
    )


    val_loader=DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False
    )


    model=GaussianGRUPredictor(
        hidden=128,
        layers=2,
        dropout=0.2
    )

    model.to(DEVICE)


    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )


    best=float("inf")


    history=[]


    for epoch in range(2):

        tr=train_epoch(
            model,
            train_loader,
            optimizer
        )


        val,ade,fde=validate(
            model,
            val_loader
        )


        history.append(val)


        print(
            actor,
            epoch+1,
            "train",
            tr,
            "val",
            val,
            "ADE",
            ade,
            "FDE",
            fde
        )


        if val < best:

            best=val

            torch.save(
                model.state_dict(),
                CKPT /
                f"{actor}_gaussian.pt"
            )


    with open(
        OUT /
        f"{actor}_results.json",
        "w"
    ) as f:

        json.dump(
            {
                "actor":actor,
                "best_nll":best,
                "history":history
            },
            f,
            indent=2
        )


print(
    "GAUSSIAN EXPERIMENTS FINISHED"
)
