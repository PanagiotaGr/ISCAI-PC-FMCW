from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import (
    GRUPredictor,
    GaussianGRUPredictor,
)

from iscai_stage4.training.gaussian_loss import gaussian_nll


DATA_ROOT = Path("/home/agni/waymo/data")

RESULTS = Path(
    "reports/stage4_auto_cuda/all_results.json"
)

CHECKPOINT = Path(
    "checkpoints/stage4/final_model.pt"
)


def get_best_config():

    results = json.loads(
        RESULTS.read_text()
    )

    best = min(
        results,
        key=lambda x: x["metrics"]["loss"]
    )

    return best



def main():

    print("==============================")
    print("STAGE 4 FINAL MODEL TRAINING")
    print("==============================")


    best = get_best_config()

    cfg = best["config"]

    print("Selected:")
    print(best["name"])
    print(cfg)


    gaussian = (
        cfg["model"] == "gaussian"
    )


    dataset = WOMDGRUDataset(
        root=DATA_ROOT,
        history=10,
        future=10,
        max_files=200,
    )


    loader = DataLoader(
        dataset,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=4,
    )


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    if gaussian:

        model = GaussianGRUPredictor(
            hidden=cfg["hidden"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
            future=10,
        )

    else:

        model = GRUPredictor(
            hidden=cfg["hidden"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
            future=10,
        )


    model.to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"]
    )


    model.train()


    epochs = 50


    for epoch in range(epochs):

        total = 0.0


        for x,y in loader:

            x = x.to(device)
            y = y.to(device)


            optimizer.zero_grad()


            out = model(x)


            if gaussian:

                loss = gaussian_nll(
                    out,
                    y
                )

            else:

                loss = nn.functional.mse_loss(
                    out,
                    y
                )


            loss.backward()

            optimizer.step()


            total += loss.item()


        avg = total / len(loader)


        print(
            f"epoch={epoch+1} loss={avg:.6f}"
        )


    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "gaussian": gaussian,
        },
        CHECKPOINT
    )


    print()
    print("SAVED:")
    print(CHECKPOINT)



if __name__ == "__main__":
    main()
