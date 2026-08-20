
import sys
from pathlib import Path

ROOT = Path("/home/agni/waymo")

for stage_src in [
    ROOT / "iscai_stage1" / "src",
    ROOT / "iscai_stage2" / "src",
    ROOT / "iscai_stage3" / "src",
    ROOT / "iscai_stage4" / "src",
    ROOT / "iscai_stage5" / "src",
]:
    value = str(stage_src)

    if value not in sys.path:
        sys.path.insert(0, value)


ROOT = Path("/home/agni/waymo")

for stage_src in [
    ROOT / "iscai_stage1" / "src",
    ROOT / "iscai_stage2" / "src",
    ROOT / "iscai_stage3" / "src",
    ROOT / "iscai_stage4" / "src",
    ROOT / "iscai_stage5" / "src",
]:
    value = str(stage_src)

    if value not in sys.path:
        sys.path.insert(0, value)

import torch

from iscai_stage5.models import LSTMMDN
from iscai_stage5.datasets import WOMDTrajectoryDataset
from iscai_stage5.training.losses import mdn_nll_loss


dataset = WOMDTrajectoryDataset(
    "womd_trajectories.json"
)


model = LSTMMDN(
    hidden_dim=64,
    components=3,
)


past, future = dataset[0]


past = past.unsqueeze(0)


output = model(
    past
)


target = future.unsqueeze(0)


loss = mdn_nll_loss(
    output,
    target,
)


print(
    "output shape =",
    output.shape
)


print(
    "loss =",
    float(loss)
)


print(
    "Stage5 training smoke PASS"
)
