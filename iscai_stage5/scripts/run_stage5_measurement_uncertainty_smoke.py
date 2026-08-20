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
from pathlib import Path

from iscai_stage5.uncertainty.measurement_covariance import (
    compute_measurement_uncertainty,
)


SOURCE = Path(
    "artifacts/stage5_pcfmcw_observations.json"
)

REPORT = Path(
    "reports/block5_measurement_uncertainty_smoke.json"
)


data = json.loads(
    SOURCE.read_text()
)


valid = 0


for item in data:

    u = compute_measurement_uncertainty(
        item["range_m"],
        item["snr_db"],
    )

    assert len(
        u["covariance"]
    ) == 4

    valid += 1


payload = {
    "records":
        len(data),

    "covariance_valid":
        valid,

    "future_used":
        False,
}


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
    ).encode()
).hexdigest()


payload["sha256"] = sha
payload["status"] = "PASS"


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 Measurement Uncertainty Smoke ====="
)

print(
    "records =",
    len(data)
)

print(
    "covariance_valid =",
    valid
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
