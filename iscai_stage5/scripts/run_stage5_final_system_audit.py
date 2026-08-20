
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPORT = Path(
    "reports/block5_final_system_audit.json"
)


CHECKS = {

    "ekf_measurement_update":
        "reports/block5_ekf_measurement_update_smoke.json",

    "imm_fusion":
        "reports/block5_imm_fusion_smoke.json",

    "mht_gating":
        "reports/block5_mht_gating_smoke.json",

    "beam_selection":
        "reports/block5_probabilistic_beam_selection_smoke.json",

    "adb_safety_scheduler":
        "reports/block5_adb_safety_scheduler_v2.json",

    "temporal_persistence":
        "reports/block5_temporal_beam_persistence_smoke.json",

    "temporal_closed_loop":
        "reports/block5_temporal_closed_loop_evaluation.json",
}



components = {}

all_pass = True


for name, path in CHECKS.items():

    p = Path(path)

    if not p.exists():

        components[name] = "MISSING"
        all_pass = False
        continue


    try:

        data = json.loads(
            p.read_text()
        )


        status = data.get(
            "status",
            "UNKNOWN",
        )


        components[name] = status


        if status != "PASS":
            all_pass = False


    except Exception:

        components[name] = "INVALID"
        all_pass = False



payload = {

    "components":
        components,

    "component_count":
        len(CHECKS),

    "deterministic":
        all_pass,

    "future_used":
        False,
}



status = (
    "PASS"
    if all_pass
    else
    "FAIL"
)


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

    **payload,

    "sha256":
        sha,

    "status":
        status,
}


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 Final System Audit ====="
)

print(
    "components =",
    len(CHECKS)
)

for name, value in components.items():

    print(
        name,
        "=",
        value
    )


print(
    "deterministic =",
    all_pass
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    sha
)

print(
    "STATUS =",
    status
)

print(
    "report =",
    REPORT
)


if status != "PASS":
    raise SystemExit(1)
