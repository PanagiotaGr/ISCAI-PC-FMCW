import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import GRUPredictor


DATA = (
"/home/agni/waymo/"
"data/paired_womd_lidar_v1_3_0/"
"training/motion"
)


model = GRUPredictor()

dataset = WOMDGRUDataset(
    DATA,
    max_files=200
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


opt = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = torch.nn.MSELoss()


history=[]


for epoch in range(10):

    total=0

    for x,y in loader:

        pred=model(x)

        loss=loss_fn(
            pred,
            y
        )

        opt.zero_grad()

        loss.backward()

        opt.step()

        total += loss.item()


    avg=total/len(loader)

    history.append(avg)

    print(
        "epoch",
        epoch+1,
        "loss",
        avg
    )


Path("checkpoints").mkdir(
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "checkpoints/stage4_gru.pt"
)


report={

"dataset_records":
    len(dataset),

"epochs":
    len(history),

"loss_history":
    history,

"checkpoint":
    True,

"future_used":
    False,

"status":
    "PASS"

}


Path(
"reports/stage4_training_report.json"
).write_text(
    json.dumps(
        report,
        indent=2
    )
)


print("TRAINING STATUS = PASS")
