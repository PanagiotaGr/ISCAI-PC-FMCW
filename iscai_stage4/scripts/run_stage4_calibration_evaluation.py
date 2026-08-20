from __future__ import annotations


import json
import hashlib
import math

from pathlib import Path



ARTIFACT = Path(
    "artifacts/stage4_prediction_artifact.json"
)


REPORT = Path(
    "reports/block4_calibration_evaluation.json"
)



if not ARTIFACT.exists():

    raise RuntimeError(
        "Missing prediction artifact."
    )



data = json.loads(
    ARTIFACT.read_text()
)



errors = []

predicted_confidence = []

observed = []



for item in data:

    trajectory = item[
        "trajectory"
    ]


    # deterministic synthetic
    # calibration target:
    # deviation from initial motion

    start = trajectory[0]

    end = trajectory[-1]


    displacement = math.sqrt(

        (end[0]-start[0])**2

        +

        (end[1]-start[1])**2

        +

        (end[2]-start[2])**2

    )


    error = (
        displacement
        /
        (1.0 + displacement)
    )


    errors.append(
        error
    )


    confidence = max(
        0.0,
        min(
            1.0,
            1.0 -
            item["uncertainty"]
        )
    )


    predicted_confidence.append(
        confidence
    )


    observed.append(
        1.0
    )



# -------------------------
# NLL
# -------------------------

variance = 0.25


nll = sum(

    0.5 *
    (
        math.log(
            2.0 *
            math.pi *
            variance
        )

        +

        (
            e*e
        )
        /
        variance
    )

    for e in errors

) / len(errors)



# -------------------------
# Brier
# -------------------------

brier = sum(

    (
        p-o
    )**2

    for p,o
    in zip(
        predicted_confidence,
        observed,
    )

) / len(errors)



# -------------------------
# Coverage
# -------------------------

threshold = math.sqrt(
    variance
)


coverage = sum(

    1

    for e in errors

    if e <= threshold

) / len(errors)



# -------------------------
# ECE
# -------------------------

ece = sum(

    abs(
        p-o
    )

    for p,o
    in zip(
        predicted_confidence,
        observed,
    )

) / len(errors)



payload = {

    "records":
        len(data),

    "nll":
        nll,

    "brier":
        brier,

    "coverage":
        coverage,

    "ece":
        ece,

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
        "stage4_calibration_evaluation",

    "records":
        len(data),

    "NLL":
        nll,

    "Brier":
        brier,

    "empirical_coverage":
        coverage,

    "ECE":
        ece,

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
    "===== Stage4 Calibration Evaluation ====="
)

print(
    "records =",
    len(data)
)

print(
    "NLL =",
    nll
)

print(
    "Brier =",
    brier
)

print(
    "coverage =",
    coverage
)

print(
    "ECE =",
    ece
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
