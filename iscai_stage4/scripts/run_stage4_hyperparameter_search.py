import json
import itertools
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import (
    GRUPredictor,
    GaussianGRUPredictor
)
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


CLASSES = [
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST"
]


GRID = {

    "model":[
        "deterministic",
        "Gaussian"
    ],

    "hidden":[
        128,
        256
    ],

    "layers":[
        1,
        2
    ],

    "dropout":[
        0.0,
        0.2
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


OUT = Path(
    "reports/stage4_hypersearch"
)

CKPT = Path(
    "checkpoints/stage4_hypersearch"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CKPT.mkdir(
    parents=True,
    exist_ok=True
)



def metrics(pred,target):

    d=torch.norm(
        pred-target,
        dim=-1
    )

    return (
        d.mean().item(),
        d[:,-1].mean().item()
    )



def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    gaussian=False
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

        out=model(x)


        if gaussian:
            loss=loss_fn(out,y)
        else:
            loss=loss_fn(out,y)


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total += loss.item()


    return total/len(loader)



def validate(
    model,
    loader,
    gaussian=False
):

    model.eval()

    vals=[]
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


            ade,fde=metrics(
                pred,
                y
            )

            vals.append(
                loss.item()
            )

            ades.append(
                ade
            )

            fdes.append(
                fde
            )


    return (
        sum(vals)/len(vals),
        sum(ades)/len(ades),
        sum(fdes)/len(fdes)
    )



experiments=[]

keys=list(GRID.keys())

for values in itertools.product(
    *[GRID[k] for k in keys]
):

    cfg=dict(
        zip(keys,values)
    )

    experiments.append(cfg)



print(
    "TOTAL EXPERIMENTS:",
    len(experiments)
)


all_results=[]


for actor in CLASSES:


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


    for idx,cfg in enumerate(experiments):


        name=(
            actor+"_"+
            cfg["model"]+
            "_h"+str(cfg["hidden"])+
            "_l"+str(cfg["layers"])
        )


        print()
        print("================")
        print(name)
        print(
            idx+1,
            "/",
            len(experiments)
        )


        train_loader=DataLoader(
            train_ds,
            batch_size=cfg["batch"],
            shuffle=True
        )

        val_loader=DataLoader(
            val_ds,
            batch_size=cfg["batch"],
            shuffle=False
        )


        gaussian = (
            cfg["model"]=="Gaussian"
        )


        if gaussian:

            model=GaussianGRUPredictor(
                hidden=cfg["hidden"],
                layers=cfg["layers"],
                dropout=cfg["dropout"]
            )

        model.to(DEVICE)

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


        best=float("inf")


        for epoch in range(20):

            train_one_epoch(
                model,
                train_loader,
                optimizer,
                gaussian_nll if gaussian else torch.nn.MSELoss(),
                gaussian
            )


            val,ade,fde=validate(
                model,
                val_loader,
                gaussian
            )


            if val < best:

                best=val

                torch.save(
                    model.state_dict(),
                    CKPT /
                    f"{name}.pt"
                )


        all_results.append(
            {
                "name":name,
                "class":actor,
                "config":cfg,
                "val_loss":best,
                "checkpoint":str(
                    CKPT/f"{name}.pt"
                )
            }
        )


with open(
    OUT/"results.json",
    "w"
) as f:

    json.dump(
        all_results,
        f,
        indent=2
    )


best_models={}


for r in all_results:

    c=r["class"]

    if (
        c not in best_models
        or r["val_loss"] <
        best_models[c]["val_loss"]
    ):
        best_models[c]=r



with open(
    OUT/"best_models.json",
    "w"
) as f:

    json.dump(
        best_models,
        f,
        indent=2
    )


print(
    "HYPERSEARCH FINISHED"
)

