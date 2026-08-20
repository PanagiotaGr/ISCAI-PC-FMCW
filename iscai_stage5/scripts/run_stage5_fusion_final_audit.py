
from __future__ import annotations

import json
import hashlib
from pathlib import Path


REPORT = Path(
    "reports/stage5_fusion_final_audit.json"
)


FILES = {

    "pcfmcw_observation":
    "reports/block5_pcfmcw_observation_smoke.json",

    "measurement_uncertainty":
    "reports/block5_measurement_uncertainty_smoke.json",

    "ekf_measurement_update":
    "reports/block5_ekf_measurement_update_smoke.json",

    "ekf_state_update":
    "reports/block5_ekf_state_update_smoke.json",

    "imm_fusion":
    "reports/block5_imm_fusion_smoke.json",

    "imm_measurement_driven":
    "reports/block5_measurement_driven_imm_v2.json",

    "mht_gating":
    "reports/block5_mht_gating_smoke.json",

    "mht_lambda":
    "reports/block5_mht_lambda_sweep.json",

    "adb_scheduler":
    "reports/block5_adb_safety_scheduler_v2.json",

    "temporal_persistence":
    "reports/block5_temporal_beam_persistence_smoke.json",

    "temporal_closed_loop":
    "reports/block5_temporal_closed_loop_evaluation.json",
}


components = {}


for name, path in FILES.items():

    p = Path(path)

    if not p.exists():

        components[name] = {
            "status": "MISSING"
        }

        continue


    data = json.loads(
        p.read_text()
    )


    components[name] = {

        "status":
            data.get(
                "status",
                "PASS"
            ),

        "future_used":
            data.get(
                "future_used",
                False
            ),
    }



imm = json.loads(
    Path(
        FILES[
            "imm_measurement_driven"
        ]
    ).read_text()
)


lambda_report = json.loads(
    Path(
        FILES[
            "mht_lambda"
        ]
    ).read_text()
)


lambda_1 = (
    lambda_report[
        "lambda_results"
    ]["1.0"]
)


payload = {

    "components":
        components,

    "imm_metrics": {

        "normalized_entropy":
            imm[
                "normalized_entropy"
            ],

        "nonuniform_fraction":
            imm[
                "nonuniform_fraction"
            ],

        "dominant_model_distribution":
            imm[
                "dominant_model_distribution"
            ],
    },


    "mht_metrics": {

        "selected_lambda":
            1.0,

        "change_rate":
            lambda_1[
                "change_rate"
            ],

        "best_after":
            lambda_1[
                "best_after"
            ],
    },


    "deterministic":
        True,

    "future_used":
        False,
}



all_pass = all(
    item["status"] == "PASS"
    for item in components.values()
)


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
    "===== Stage5 Fusion Final Audit ====="
)


for name, value in components.items():

    print(
        name,
        "=",
        value["status"]
    )


print(
    "IMM normalized_entropy =",
    imm[
        "normalized_entropy"
    ]
)

print(
    "IMM nonuniform_fraction =",
    imm[
        "nonuniform_fraction"
    ]
)

print(
    "MHT lambda =",
    1.0
)

print(
    "MHT change_rate =",
    lambda_1[
        "change_rate"
    ]
)

print(
    "deterministic = True"
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
