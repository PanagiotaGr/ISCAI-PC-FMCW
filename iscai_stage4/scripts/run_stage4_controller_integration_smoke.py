from __future__ import annotations


import json
import hashlib

from pathlib import Path


from iscai_stage4.controller.prediction_adapter import (
    prediction_to_controller,
)



ARTIFACT = Path(
    "artifacts/stage4_prediction_artifact.json"
)


REPORT = Path(
    "reports/block4_controller_integration_smoke.json"
)



if not ARTIFACT.exists():

    raise RuntimeError(
        "Missing Stage4 artifact."
    )



data = json.loads(
    ARTIFACT.read_text()
)



candidates = []



for item in data:

    if item["future_used"]:

        raise RuntimeError(
            "Future leakage."
        )


    candidate = prediction_to_controller(
        item
    )


    if not (
        0.0 <= candidate.confidence <= 1.0
    ):

        raise RuntimeError(
            "Invalid confidence."
        )


    if not (
        0.0 <= candidate.utility <= 1.0
    ):

        raise RuntimeError(
            "Invalid utility."
        )


    candidates.append(
        candidate
    )



payload = {

    "candidates":
        len(candidates),

    "models":
        sorted(
            {
                c.model_name
                for c in candidates
            }
        ),

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

    "block":
        "stage4_controller_integration_smoke",

    "candidates":
        len(candidates),

    "models":
        payload["models"],

    "confidence_valid":
        True,

    "utility_valid":
        True,

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
    "===== Stage4 Controller Integration Smoke ====="
)

print(
    "candidates =",
    report["candidates"]
)

print(
    "models =",
    report["models"]
)

print(
    "confidence_valid = YES"
)

print(
    "utility_valid = YES"
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
