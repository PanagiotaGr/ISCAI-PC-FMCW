
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

from iscai_stage5.datasets import (
    TrajectoryDataset,
    TrajectorySample,
)


samples = [

    TrajectorySample(
        past=torch.zeros(
            10,
            2,
        ),

        future=torch.ones(
            10,
            2,
        ),
    )

]


dataset = TrajectoryDataset(
    samples
)


past, future = dataset[0]


print(
    "dataset size =",
    len(dataset)
)


print(
    "past shape =",
    past.shape
)


print(
    "future shape =",
    future.shape
)


assert past.shape == (
    10,
    2,
)

assert future.shape == (
    10,
    2,
)


print(
    "Stage5 dataset PASS"
)
