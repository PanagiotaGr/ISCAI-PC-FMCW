import json
import time
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset


SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


DATA = (
    "/home/agni/waymo/"
    "data/paired_womd_lidar_v1_3_0/"
    "training/motion"
)


CHECKPOINT_DIR = Path(
    "checkpoints/stage4_full"
)

REPORT_DIR = Path(
    "reports/stage4_full"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



class GRUPredictor(nn.Module):

    def __init__(
        self,
        input_size=8,
        hidden=128,
        layers=1,
        dropout=0.0,
        future=10
    ):
        super().__init__()

        self.future = future


        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.head = nn.Linear(
            hidden,
            future * 2
        )


    def forward(self,x):

        _, h = self.gru(x)

        out = self.dropout(
            h[-1]
        )

        out = self.head(out)


        return out.reshape(
            -1,
            self.future,
            2
        )



def ADE(pred, target):

    error = torch.sqrt(
        ((pred-target)**2).sum(dim=-1)
    )

    return error.mean()



def FDE(pred, target):

    error = torch.sqrt(
        ((pred[:,-1]-target[:,-1])**2)
        .sum(dim=-1)
    )

    return error.mean()



def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn
):

    model.train()

    total = 0


    for x,y in loader:

        pred = model(x)

        loss = loss_fn(
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


    return total / len(loader)



def validate(
    model,
    loader,
    loss_fn
):

    model.eval()


    total = 0
    ade_total = 0
    fde_total = 0


    with torch.no_grad():

        for x,y in loader:

            pred = model(x)


            loss = loss_fn(
                pred,
                y
            )


            total += loss.item()

            ade_total += ADE(
                pred,
                y
            ).item()


            fde_total += FDE(
                pred,
                y
            ).item()


    n = len(loader)


    return (
        total/n,
        ade_total/n,
        fde_total/n
    )



EXPERIMENTS = []


# Phase 1 - architecture

for layers in [1,2,3]:

    for hidden in [64,128,256]:

        EXPERIMENTS.append(
            {
                "name":
                f"layers{layers}_hidden{hidden}",

                "layers":
                layers,

                "hidden":
                hidden,

                "dropout":
                0.2,

                "batch":
                32,

                "lr":
                1e-3
            }
        )



# Phase 2 dropout

for d in [0.0,0.1,0.3,0.5]:

    EXPERIMENTS.append(
        {
            "name":
            f"dropout_{d}",

            "layers":
            2,

            "hidden":
            128,

            "dropout":
            d,

            "batch":
            32,

            "lr":
            1e-3
        }
    )



# Phase 3 batch

for b in [16,32,64]:

    EXPERIMENTS.append(
        {
            "name":
            f"batch_{b}",

            "layers":
            2,

            "hidden":
            128,

            "dropout":
            0.2,

            "batch":
            b,

            "lr":
            1e-3
        }
    )



# Phase 4 learning rate

for lr in [1e-2,1e-3,5e-4,1e-4]:

    EXPERIMENTS.append(
        {
            "name":
            f"lr_{lr}",

            "layers":
            2,

            "hidden":
            128,

            "dropout":
            0.2,

            "batch":
            32,

            "lr":
            lr
        }
    )




TRAIN_DATA = (
    "/home/agni/waymo/"
    "data/paired_womd_lidar_v1_3_0/"
    "training/motion"
)

VAL_DATA = (
    "/home/agni/waymo/"
    "data/paired_womd_lidar_v1_3_0/"
    "validation/motion"
)


train_dataset = WOMDGRUDataset(
    TRAIN_DATA,
    max_files=200
)


val_dataset = WOMDGRUDataset(
    VAL_DATA,
    max_files=150
)


print(
    "TRAIN DATASET:",
    len(train_dataset)
)

print(
    "VAL DATASET:",
    len(val_dataset)
)



all_results = []



for exp in EXPERIMENTS:


    print(
        "\n===================="
    )

    print(
        "RUNNING",
        exp["name"]
    )


    start = time.time()


    model = GRUPredictor(
        hidden=exp["hidden"],
        layers=exp["layers"],
        dropout=exp["dropout"]
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=exp["batch"],
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=exp["batch"],
        shuffle=False
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp["lr"],
        weight_decay=1e-4
    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5
    )


    loss_fn = nn.MSELoss()


    best_val = float("inf")

    best_metrics = None

    patience = 10
    epochs_without_improvement = 0
    best_epoch = 0


    train_history=[]

    val_history=[]


    for epoch in range(100):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn
        )


        val_loss, ade, fde = validate(
            model,
            val_loader,
            loss_fn
        )


        scheduler.step(
            val_loss
        )


        train_history.append(
            train_loss
        )

        val_history.append(
            val_loss
        )


        if val_loss < best_val:

            best_val = val_loss

            epochs_without_improvement = 0

            best_epoch = epoch + 1

            best_metrics = {
                "ADE":ade,
                "FDE":fde
            }

        else:

            epochs_without_improvement += 1


        if epochs_without_improvement >= patience:

            print(
                "EARLY STOP at epoch",
                epoch+1,
                flush=True
            )

            break


            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR /
                f"{exp['name']}.pt"
            )


        print(
            exp["name"],
            epoch+1,
            train_loss,
            val_loss,
            ade,
            fde,
            flush=True
        )



    result = {

        **exp,

        "best_val_loss":
        best_val,

        "ADE":
        best_metrics["ADE"],

        "FDE":
        best_metrics["FDE"],

        "epochs":
        100,

        "time_sec":
        time.time()-start,

        "train_history":
        train_history,

        "val_history":
        val_history

    }


    all_results.append(
        result
    )


    with open(
        REPORT_DIR /
        "stage4_full_results.json",
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )



print(
    "\nALL EXPERIMENTS FINISHED"
)
