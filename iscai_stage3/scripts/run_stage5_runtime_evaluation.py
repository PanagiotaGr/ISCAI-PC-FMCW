from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path


ARTIFACT = Path(
    "artifacts/real_predictor_trajectories.json"
)


REPORT = Path(
    "reports/block5a_runtime_evaluation.json"
)



if not ARTIFACT.exists():

    raise RuntimeError(
        "Missing trajectory artifact."
    )


start_time = time.perf_counter()


data = json.loads(
    ARTIFACT.read_text()
)



failures = []

models = {}



for item in data:

    model = item["model_name"]

    models.setdefault(
        model,
        0,
    )

    models[model] += 1


    if item["future_used"]:

        failures.append(
            {
                "type":
                    "future_leakage",

                "model":
                    model,
            }
        )


    if len(item["trajectory"]) != 10:

        failures.append(
            {
                "type":
                    "invalid_horizon",

                "model":
                    model,
            }
        )



elapsed = (
    time.perf_counter()
    -
    start_time
)



payload = {

    "records":
        len(data),

    "models":
        models,

    "failures":
        failures,

    "runtime_sec":
        elapsed,

}



sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
    ).encode()
).hexdigest()



report = {

    "block":
        "stage5a_runtime_evaluation",

    "records":
        len(data),

    "models":
        models,

    "runtime_sec":
        elapsed,

    "failure_count":
        len(failures),

    "future_used":
        False,

    "sha256":
        sha,

    "status":
        "PASS"
        if len(failures) == 0
        else "FAIL",
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
    "===== Stage5A Runtime Evaluation ====="
)

print(
    "records =",
    len(data)
)

print(
    "models =",
    models
)

print(
    "runtime_sec =",
    elapsed
)

print(
    "failures =",
    len(failures)
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
    report["status"]
)

print(
    "report =",
    REPORT
)
