
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

from iscai_stage5.models.lstm_mdn import LSTMMDN


model = LSTMMDN(
    hidden_dim=64,
    components=3,
)


x = torch.randn(
    4,
    10,
    2,
)


out = model(x)


print(
    "input =",
    x.shape
)

print(
    "output =",
    out.shape
)

print(
    "Stage5 LSTM-MDN smoke PASS"
)
