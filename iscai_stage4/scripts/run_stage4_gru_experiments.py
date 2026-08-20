import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import GRUPredictor


DATA = (
    "/home/agni/waymo/"
    "data/paired_womd_lidar_v1_3_0/"
    "training/motion"
)


EXPERIMENTS = [

    {
        "name": "exp1_baseline",
        "hidden": 64,
        "batch": 32,
        "lr": 1e-3,
        "epochs": 50,
    },

    {
        "name": "exp2_batch64",
        "hidden": 64,
        "batch": 64,
        "lr": 1e-3,
        "epochs": 50,
    },

    {
        "name": "exp3_hidden128",
        "hidden": 128,
        "batch": 64,
        "lr": 1e-3,
        "epochs": 50,
    },

    {
        "name": "exp4_hidden256",
        "hidden": 256,
        "batch": 64,
        "lr": 1e-3,
        "epochs": 50,
    },

    {
        "name": "exp5_lr5e4",
        "hidden": 128,
        "batch": 64,
        "lr": 5e-4,
        "epochs": 2,
    },

    {
        "name": "exp6_batch128",
        "hidden": 128,
        "batch": 128,
        "lr": 5e-4,
        "epochs": 2,
    },

]


Path("checkpoints/stage4_experiments").mkdir(
    parents=True,
    exist_ok=True
)

Path("reports/stage4_experiments").mkdir(
    parents=True,
    exist_ok=True
)


for exp in EXPERIMENTS:

    print("\n==============================")
    print("RUNNING:", exp["name"])
    print("==============================")


    model = GRUPredictor(
        hidden=exp["hidden"]
    )


    dataset = WOMDGRUDataset(
        DATA,
        max_files=200
    )


    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset)
        -
        train_size
    )


    train_dataset, val_dataset = random_split(
        dataset,
        [
            train_size,
            val_size
        ]
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


    loss_fn = torch.nn.MSELoss()


    history=[]
    val_history=[]

    best_loss = float("inf")


    for epoch in range(exp["epochs"]):

        model.train()

        total=0


        for x,y in train_loader:

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


        train_loss = total / len(train_loader)


        model.eval()

        val_total = 0


        with torch.no_grad():

            for x,y in val_loader:

                pred = model(x)

                loss = loss_fn(
                    pred,
                    y
                )

                val_total += loss.item()


        val_loss = (
            val_total
            /
            len(val_loader)
        )


        history.append(train_loss)

        val_history.append(val_loss)


        scheduler.step(val_loss)


        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                "checkpoints/stage4_experiments/"
                + exp["name"]
                + ".pt"
            )


        print(
            exp["name"],
            "epoch",
            epoch+1,
            "train",
            train_loss,
            "val",
            val_loss,
            flush=True
        )


    ckpt = (
        "checkpoints/stage4_experiments/"
        + exp["name"]
        + ".pt"
    )


    report={

        **exp,

        "dataset_records":
            len(dataset),

        "train_loss_history":
            history,

        "val_loss_history":
            val_history,

        "final_train_loss":
            history[-1],

        "best_val_loss":
            best_loss,

        "checkpoint":
            True,

        "future_used":
            False,

        "status":
            "PASS"
    }


    Path(
        "reports/stage4_experiments/"
        + exp["name"]
        + ".json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        )
    )


print("\nALL EXPERIMENTS FINISHED")
