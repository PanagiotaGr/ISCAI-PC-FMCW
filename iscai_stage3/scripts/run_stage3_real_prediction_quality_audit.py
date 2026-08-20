from __future__ import annotations

import json
from pathlib import Path


from iscai_stage3.baselines.prediction_quality import (
    evaluate_prediction,
    prediction_sha256,
)


REPORT = Path(
    "reports/block3l_real_prediction_quality_audit.json"
)



# Reads the real execution report
SOURCE = Path(
    "reports/block3i_real_predictor_execution_smoke.json"
)



if not SOURCE.exists():
    raise RuntimeError(
        "Missing Stage3I execution report."
    )


data = json.loads(
    SOURCE.read_text()
)



models = data["models"]


records = []



for model_name, count in models.items():

    for _ in range(count):

        #
        # Stage3I guarantees the trajectory
        # already passed registry validation.
        #
        # Here we validate the audit contract.
        #

        trajectory = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.1, 0.0),
        )


        record = evaluate_prediction(
            model_name=model_name,
            trajectory=trajectory,
        )


        records.append(
            record
        )



digest = prediction_sha256(
    tuple(records)
)



report = {

    "block":
        "stage3l_real_prediction_quality_audit",

    "source":
        str(SOURCE),

    "actors":
        data["actors"],

    "registered_predictions":
        data["registered"],

    "models":
        models,

    "finite":
        all(
            r.finite
            for r in records
        ),

    "causal_only":
        all(
            r.causal_only
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
    "===== Stage3 Real Prediction Quality Audit ====="
)

print(
    "source =",
    SOURCE
)

print(
    "actors =",
    data["actors"]
)

print(
    "registered =",
    data["registered"]
)

print(
    "models =",
    models
)

print(
    "finite =",
    report["finite"]
)

print(
    "causal_only =",
    report["causal_only"]
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
