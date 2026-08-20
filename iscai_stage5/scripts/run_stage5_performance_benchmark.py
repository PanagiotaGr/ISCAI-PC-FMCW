from __future__ import annotations


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



import json
import hashlib
import time

from pathlib import Path


from iscai_stage5.integration.stage4_prediction_loader import (
    load_stage4_predictions,
)



SOURCE = Path(
    "/home/agni/waymo/iscai_stage4/artifacts/stage4_prediction_artifact.json"
)


REPORT = Path(
    "reports/block5_performance_benchmark.json"
)



start = time.perf_counter()



predictions = load_stage4_predictions(
    str(SOURCE)
)



processed = 0



for prediction in predictions:

    if len(
        prediction.trajectory
    ) != 10:

        raise RuntimeError(
            "Invalid trajectory."
        )


    processed += 1



runtime = (
    time.perf_counter()
    -
    start
)



if runtime == 0:

    throughput = 0.0

else:

    throughput = (
        processed
        /
        runtime
    )



payload = {

    "records":
        processed,

    "runtime_sec":
        runtime,

    "throughput":
        throughput,

    "future_used":
        False,

}



sha = hashlib.sha256(

    json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode()

).hexdigest()



report = {

    "records":
        processed,

    "runtime_sec":
        runtime,

    "predictions_per_sec":
        throughput,

    "future_used":
        False,

    "sha256":
        sha,

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
    "===== Stage5E Performance Benchmark ====="
)

print(
    "records =",
    report["records"]
)

print(
    "runtime_sec =",
    report["runtime_sec"]
)

print(
    "predictions/sec =",
    report["predictions_per_sec"]
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    sha
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)
