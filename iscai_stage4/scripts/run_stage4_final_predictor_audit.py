from __future__ import annotations


import json
import hashlib

from pathlib import Path



PREDICTION_ARTIFACT = Path(
    "artifacts/stage4_prediction_artifact.json"
)


QUALITY_REPORT = Path(
    "reports/block4_prediction_quality_audit.json"
)


CALIBRATION_REPORT = Path(
    "reports/block4_calibration_evaluation.json"
)


REPORT = Path(
    "reports/block4_final_predictor_audit.json"
)



for path in (
    PREDICTION_ARTIFACT,
    QUALITY_REPORT,
    CALIBRATION_REPORT,
):

    if not path.exists():

        raise RuntimeError(
            f"Missing required file: {path}"
        )



predictions = json.loads(
    PREDICTION_ARTIFACT.read_text()
)


quality = json.loads(
    QUALITY_REPORT.read_text()
)


calibration = json.loads(
    CALIBRATION_REPORT.read_text()
)



models = sorted(
    {
        x["model_name"]
        for x in predictions
    }
)



if quality["future_used"]:

    raise RuntimeError(
        "Future leakage."
    )


if calibration["future_used"]:

    raise RuntimeError(
        "Calibration future leakage."
    )



actor_ids = {
    (
        x.get("scenario_id"),
        x.get("track_index"),
    )
    for x in predictions
}


expected_prediction_count = (
    len(actor_ids)
    *
    len(models)
)


if len(predictions) != expected_prediction_count:

    raise RuntimeError(
        "Unexpected prediction count: "
        f"got={len(predictions)}, "
        f"expected={expected_prediction_count}, "
        f"actors={len(actor_ids)}, "
        f"models={len(models)}"
    )



if quality["status"] != "PASS":

    raise RuntimeError(
        "Quality audit failed."
    )



if calibration["status"] != "PASS":

    raise RuntimeError(
        "Calibration failed."
    )



payload = {

    "prediction_records":
        len(predictions),

    "models":
        models,

    "quality":
        quality["sha256"],

    "calibration":
        calibration["sha256"],

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
        "stage4_final_predictor_audit",

    "components":
        3,

    "prediction_records":
        len(predictions),

    "models":
        models,

    "quality_audit":
        "PASS",

    "calibration":
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
    "===== Stage4 Final Predictor Audit ====="
)

print(
    "components =",
    report["components"]
)

print(
    "prediction_records =",
    report["prediction_records"]
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
