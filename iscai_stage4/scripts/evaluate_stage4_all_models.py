import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import (
    GRUPredictor,
    GaussianGRUPredictor
)

from iscai_stage4.training.gaussian_loss import gaussian_nll


VAL_DATA = (
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)


RESULTS = Path(
    "reports/stage4_final_comparison.json"
)


CLASSES = [
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST"
]


def compute_ade_fde(pred, target):

    dist = torch.norm(
        pred-target,
        dim=-1
    )

    ade = dist.mean().item()

    fde = dist[:,-1].mean().item()

    return ade, fde



def evaluate_deterministic(
    model,
    loader
):

    model.eval()

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

            ade,fde = compute_ade_fde(
                pred,
                y
            )

            ades.append(ade)
            fdes.append(fde)


    return {
        "ADE":
            sum(ades)/len(ades),

        "FDE":
            sum(fdes)/len(fdes)
    }



def evaluate_gaussian(
    model,
    loader
):

    model.eval()

    ades=[]
    fdes=[]
    nlls=[]


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


            ade,fde = compute_ade_fde(
                out["mu"],
                y
            )


            nll = gaussian_nll(
                out,
                y
            ).item()


            ades.append(ade)
            fdes.append(fde)
            nlls.append(nll)


    return {

        "ADE":
            sum(ades)/len(ades),

        "FDE":
            sum(fdes)/len(fdes),

        "NLL":
            sum(nlls)/len(nlls)
    }



results=[]


for actor in CLASSES:

    print()
    print("================")
    print(actor)
    print("================")


    dataset=WOMDGRUDataset(
        VAL_DATA,
        max_files=150,
        actor_type=actor
    )


    loader=DataLoader(
        dataset,
        batch_size=64,
        shuffle=False
    )


    print(
        "samples",
        len(dataset)
    )


    # -----------------------
    # Deterministic
    # -----------------------

    det_model=GRUPredictor(
        hidden=128,
        layers=2,
        dropout=0.2
    )

    model.to(DEVICE)


    det_ckpt = (
        Path(
            "checkpoints/stage4_full"
        )
        /
        "layers2_hidden256.pt"
    )


    if det_ckpt.exists():

        det_model.load_state_dict(
            torch.load(
                det_ckpt,
                map_location="cpu"
            )
        )


        det_metrics=evaluate_deterministic(
            det_model,
            loader
        )


        results.append(
            {
                "class":actor,
                "model":"deterministic_GRU",
                **det_metrics
            }
        )


    # -----------------------
    # Gaussian
    # -----------------------

    gauss_model=GaussianGRUPredictor(
        hidden=128,
        layers=2,
        dropout=0.2
    )

    model.to(DEVICE)


    gauss_ckpt = (
        Path(
            "checkpoints/stage4_gaussian"
        )
        /
        f"{actor}_gaussian.pt"
    )


    if gauss_ckpt.exists():

        gauss_model.load_state_dict(
            torch.load(
                gauss_ckpt,
                map_location="cpu"
            )
        )


        gauss_metrics=evaluate_gaussian(
            gauss_model,
            loader
        )


        results.append(
            {
                "class":actor,
                "model":"gaussian_GRU",
                **gauss_metrics
            }
        )



RESULTS.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    RESULTS,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print()
print("FINAL COMPARISON DONE")
print(RESULTS)
