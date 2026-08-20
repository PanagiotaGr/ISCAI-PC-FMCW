import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import GRUPredictor


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


CKPT = Path(
    "checkpoints/stage4_deterministic_final"
)

REPORT = Path(
    "reports/stage4_deterministic_final"
)

CKPT.mkdir(
    parents=True,
    exist_ok=True
)

REPORT.mkdir(
    parents=True,
    exist_ok=True
)


CLASSES = [
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST"
]


def metrics(pred,target):

    d = torch.norm(
        pred-target,
        dim=-1
    )

    return (
        d.mean().item(),
        d[:,-1].mean().item()
    )



def train_epoch(
    model,
    loader,
    optimizer,
    loss_fn
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

        loss=loss_fn(
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
    loader,
    loss_fn
):

    model.eval()

    losses=[]
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

            loss=loss_fn(
                pred,
                y
            )

            ade,fde=metrics(
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


    return (
        sum(losses)/len(losses),
        sum(ades)/len(ades),
        sum(fdes)/len(fdes)
    )



results=[]


for actor in CLASSES:

    print()
    print("================")
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


    model=GRUPredictor(
        input_size=8,
        hidden=256,
        layers=2,
        dropout=0.2
    )

    model.to(DEVICE)


    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )


    loss_fn=torch.nn.MSELoss()


    best=float("inf")

    history=[]


    for epoch in range(2):

        tr=train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn
        )


        val,ade,fde=validate(
            model,
            val_loader,
            loss_fn
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
                f"{actor}_deterministic.pt"
            )


    results.append(
        {
            "class":actor,
            "best_loss":best,
            "history":history
        }
    )


with open(
    REPORT/"results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print(
    "DETERMINISTIC FINAL DONE"
)
