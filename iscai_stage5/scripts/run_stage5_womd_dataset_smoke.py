
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

from iscai_stage3.io.womd_reader import (
    read_scenario_by_id,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage5.datasets import (
    build_samples_from_actor,
)


SCENARIO_ID = (
    "b85e1bd6cc8e74c0"
)


(
    scenario,
    _,
    _,
    _,
) = read_scenario_by_id(
    SCENARIO_ID
)


adapted = (
    adapt_causal_womd_scenario(
        scenario
    )
)


total = 0


for actor in adapted.actors:

    samples = build_samples_from_actor(
        actor.artifact.history,
        anchor_index=adapted.anchor_index,
    )

    total += len(samples)


print(
    "scenario =",
    SCENARIO_ID,
)

print(
    "actors =",
    len(adapted.actors),
)

print(
    "samples =",
    total,
)

print(
    "Stage5 WOMD dataset PASS"
)
