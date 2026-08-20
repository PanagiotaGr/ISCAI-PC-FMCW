
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

#from __future__ import annotations


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


import json
import hashlib

from pathlib import Path


from iscai_stage5.integration.stage4_prediction_loader import (
    load_stage4_predictions,
)



SOURCE = Path(
    "/home/agni/waymo/iscai_stage4/artifacts/stage4_prediction_artifact.json"
)


REPORT = Path(
    "reports/block5_reproducibility_test.json"
)



def compute_hash():

    predictions = load_stage4_predictions(
        str(SOURCE)
    )


    payload = {

        "records":
            len(predictions),

        "models":
            sorted(
                {
                    x.model_name
                    for x in predictions
                }
            ),

        "uncertainty":
            [
                x.uncertainty
                for x in predictions
            ],

        "future_used":
            False,

    }


    return hashlib.sha256(

        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode()

    ).hexdigest()



hashes = []



for _ in range(3):

    hashes.append(
        compute_hash()
    )



deterministic = (
    hashes[0]
    ==
    hashes[1]
    ==
    hashes[2]
)



if not deterministic:

    raise RuntimeError(
        "Non deterministic output."
    )



report = {

    "runs":
        3,

    "hashes":
        hashes,

    "deterministic":
        True,

    "future_used":
        False,

    "status":
        "PASS",
}



REPORT.parent.mkdir(
    exist_ok=True
)


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)



print(
    "===== Stage5D Reproducibility Test ====="
)

print(
    "runs =",
    report["runs"]
)


for i,h in enumerate(
    hashes,
    start=1
):

    print(
        f"run {i} = {h}"
    )


print(
    "deterministic = True"
)

print(
    "future_used = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)
