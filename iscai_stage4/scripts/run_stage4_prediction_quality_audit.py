from __future__ import annotations


import json
import hashlib
import math

from pathlib import Path



ARTIFACT = Path(
    "artifacts/stage4_prediction_artifact.json"
)


REPORT = Path(
    "reports/block4_prediction_quality_audit.json"
)



EXPECTED_MODELS = {

    "DETERMINISTIC_GRU",

    "GAUSSIAN_GRU",

    "GMM_cross_wait",

    "GMM_lane_change",

    "GMM_stop_decelerate",

    "GMM_straight",

    "GMM_turn",
}



if not ARTIFACT.exists():

    raise RuntimeError(
        "Missing Stage4 artifact."
    )



data = json.loads(
    ARTIFACT.read_text()
)



if len(data) == 0:

    raise RuntimeError(
        "Empty artifact."
    )



models = set()

horizons = set()



for record in data:


    models.add(
        record["model_name"]
    )


    trajectory = record[
        "trajectory"
    ]


    horizons.add(
        len(trajectory)
    )


    if record[
        "future_used"
    ]:

        raise RuntimeError(
            "Future leakage detected."
        )


    uncertainty = record[
        "uncertainty"
    ]


    if not (
        0.0 <= uncertainty <= 1.0
    ):

        raise RuntimeError(
            "Invalid uncertainty."
        )


    for point in trajectory:

        if len(point) != 3:

            raise RuntimeError(
                "Invalid point dimension."
            )


        if not all(
            math.isfinite(x)
            for x in point
        ):

            raise RuntimeError(
                "Non finite trajectory."
            )



if models != EXPECTED_MODELS:

    raise RuntimeError(
        "Model coverage mismatch."
    )



if horizons != {10}:

    raise RuntimeError(
        "Invalid prediction horizon."
    )



payload = {

    "records":
        len(data),

    "models":
        sorted(models),

    "horizon":
        10,

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
        "stage4_prediction_quality_audit",

    "records":
        len(data),

    "models":
        sorted(models),

    "horizon":
        10,

    "finite":
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
    "===== Stage4 Prediction Quality Audit ====="
)

print(
    "records =",
    report["records"]
)

print(
    "models =",
    report["models"]
)

print(
    "horizon =",
    report["horizon"]
)

print(
    "finite = True"
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
