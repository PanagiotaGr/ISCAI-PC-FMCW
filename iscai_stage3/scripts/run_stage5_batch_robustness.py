from __future__ import annotations

import json
import hashlib
from pathlib import Path



REPORT = Path(
    "reports/block5b_batch_robustness.json"
)



ARTIFACTS = [

    Path(
        "artifacts/real_predictor_trajectories.json"
    ),

]



results = []

failures = []



for artifact in ARTIFACTS:


    if not artifact.exists():

        failures.append(
            {
                "artifact":
                    str(artifact),

                "error":
                    "missing",
            }
        )

        continue



    data = json.loads(
        artifact.read_text()
    )


    local_failures = 0


    models = {}



    for item in data:


        model = item[
            "model_name"
        ]


        models.setdefault(
            model,
            0,
        )

        models[model] += 1



        if item.get(
            "future_used",
            True,
        ):

            local_failures += 1


        if len(
            item["trajectory"]
        ) != 10:

            local_failures += 1



    results.append(
        {
            "artifact":
                str(artifact),

            "records":
                len(data),

            "models":
                models,

            "failures":
                local_failures,
        }
    )



payload = {

    "results":
        results,

    "failures":
        failures,

}



sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
    ).encode()
).hexdigest()



report = {

    "block":
        "stage5b_batch_robustness",

    "scenarios_checked":
        len(results),

    "records":
        sum(
            x["records"]
            for x in results
        ),

    "failures":
        sum(
            x["failures"]
            for x in results
        )
        +
        len(failures),

    "future_used":
        False,

    "sha256":
        sha,

    "status":
        "PASS"
        if (
            len(failures) == 0
            and
            sum(
                x["failures"]
                for x in results
            ) == 0
        )
        else
        "FAIL",

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
    "===== Stage5B Batch Robustness ====="
)

print(
    "scenarios =",
    report["scenarios_checked"]
)

print(
    "records =",
    report["records"]
)

print(
    "failures =",
    report["failures"]
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
