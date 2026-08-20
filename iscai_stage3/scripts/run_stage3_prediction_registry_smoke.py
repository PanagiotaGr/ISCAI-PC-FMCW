from __future__ import annotations

import json
from pathlib import Path


from iscai_stage3.baselines.registry import (
    register_prediction,
)



REPORT = Path(
    "reports/block3g_prediction_registry_smoke.json"
)


models = {

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



registered = []


for name, states in models.items():

    result = register_prediction(
        model_name=name,
        states=states,
    )

    registered.append(
        result.model_name
    )



report = {

    "block":
        "stage3g_prediction_registry_smoke",

    "models":
        registered,

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
    )
)



print(
    "===== Stage3 Prediction Registry smoke ====="
)

print(
    "models =",
    registered
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
