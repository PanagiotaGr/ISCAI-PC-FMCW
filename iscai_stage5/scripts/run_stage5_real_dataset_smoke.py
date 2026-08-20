
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

from iscai_stage5.datasets import (
    WOMDTrajectoryDataset,
)


dataset = WOMDTrajectoryDataset(
    "womd_trajectories.json"
)


print(
    "samples =",
    len(dataset)
)


past, future = dataset[0]


print(
    "past shape =",
    past.shape
)


print(
    "future shape =",
    future.shape
)


print(
    "first past =",
    past[0]
)


print(
    "first future =",
    future[0]
)


print(
    "Stage5 real dataset PASS"
)
