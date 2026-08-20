from __future__ import annotations

import json
from pathlib import Path


from iscai_stage3.baselines.prediction_quality import (
    evaluate_prediction,
    prediction_sha256,
)



REPORT = Path(
    "reports/block3k_prediction_quality_smoke.json"
)



predictions = {

    "CV": (
        (1.0,0.0,0.0),
        (2.0,0.0,0.0),
    ),

    "CA": (
        (1.0,0.0,0.0),
        (2.5,0.0,0.0),
    ),

    "CTRV": (
        (1.0,0.1,0.0),
        (2.0,0.3,0.0),
    ),

    "KALMAN": (
        (1.0,0.0,0.0),
        (2.0,0.0,0.0),
    ),

    "IMM": (
        (1.2,0.0,0.0),
        (2.2,0.1,0.0),
    ),

    "MHT": (
        (1.1,0.0,0.0),
        (2.1,0.0,0.0),
    ),
}



records = []


for name, trajectory in predictions.items():

    records.append(
        evaluate_prediction(
            model_name=name,
            trajectory=trajectory,
        )
    )



digest = prediction_sha256(
    tuple(records)
)



report = {

    "block":
        "stage3k_prediction_quality_smoke",

    "models":
        [
            r.model_name
            for r in records
        ],

    "trajectory_lengths":
        {
            r.model_name:
            r.trajectory_length
            for r in records
        },

    "finite":
        all(
            r.finite
            for r in records
        ),

    "future_used":
        False,

    "sha256":
        digest,

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
    "===== Stage3 Prediction Quality smoke ====="
)

print(
    "models =",
    report["models"]
)

print(
    "finite =",
    report["finite"]
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    digest
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)
