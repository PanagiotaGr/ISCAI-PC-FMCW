from __future__ import annotations

import json
import hashlib

from pathlib import Path


FILES = {

    "artifact":
        Path(
            "artifacts/stage4_prediction_artifact.json"
        ),

    "quality":
        Path(
            "reports/block4_prediction_quality_audit.json"
        ),

    "calibration":
        Path(
            "reports/block4_calibration_evaluation.json"
        ),

    "controller":
        Path(
            "reports/block4_controller_integration_smoke.json"
        ),

    "selection":
        Path(
            "reports/block4_prediction_selection_smoke.json"
        ),

    "replay":
        Path(
            "reports/block4_closed_loop_replay_smoke.json"
        ),
}


REPORT = Path(
    "reports/block4_final_full_audit.json"
)



for name, path in FILES.items():

    if not path.exists():

        raise RuntimeError(
            f"Missing {name}: {path}"
        )



artifact = json.loads(
    FILES["artifact"].read_text()
)


quality = json.loads(
    FILES["quality"].read_text()
)


calibration = json.loads(
    FILES["calibration"].read_text()
)


controller = json.loads(
    FILES["controller"].read_text()
)


selection = json.loads(
    FILES["selection"].read_text()
)


replay = json.loads(
    FILES["replay"].read_text()
)



for item in artifact:

    if item["future_used"]:

        raise RuntimeError(
            "Future leakage."
        )



if not all(
    x["status"] == "PASS"
    for x in (
        quality,
        calibration,
        controller,
        selection,
        replay,
    )
):

    raise RuntimeError(
        "Component failure."
    )



models = sorted(
    {
        x["model_name"]
        for x in artifact
    }
)



payload = {

    "components":
        6,

    "prediction_records":
        len(artifact),

    "models":
        models,

    "future_used":
        False,

    "quality_sha":
        quality["sha256"],

    "calibration_sha":
        calibration["sha256"],

    "controller_sha":
        controller["sha256"],

    "selection_sha":
        selection["sha256"],

    "replay_sha":
        replay["sha256"],

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
        "stage4_final_full_audit",

    "components":
        6,

    "predictions":
        len(artifact),

    "models":
        models,

    "quality":
        "PASS",

    "calibration":
        "PASS",

    "controller":
        "PASS",

    "selection":
        "PASS",

    "replay":
        "PASS",

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
    "===== Stage4 Final Full Audit ====="
)

print(
    "components =",
    report["components"]
)

print(
    "predictions =",
    report["predictions"]
)

print(
    "models =",
    report["models"]
)

print(
    "quality = PASS"
)

print(
    "calibration = PASS"
)

print(
    "controller = PASS"
)

print(
    "selection = PASS"
)

print(
    "replay = PASS"
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
