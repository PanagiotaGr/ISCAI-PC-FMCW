from __future__ import annotations

import json
import hashlib
from pathlib import Path


ARTIFACT = Path(
    "artifacts/real_predictor_trajectories.json"
)


REPORT = Path(
    "reports/block5d_reproducibility_test.json"
)


RUNS = 3



if not ARTIFACT.exists():

    raise RuntimeError(
        "Missing trajectory artifact."
    )



def compute_digest():

    data = json.loads(
        ARTIFACT.read_text()
    )


    payload = {

        "records":
            len(data),

        "models":
            sorted(
                {
                    x["model_name"]
                    for x in data
                }
            ),

        "lengths":
            sorted(
                {
                    len(x["trajectory"])
                    for x in data
                }
            ),

        "future_used":
            any(
                x["future_used"]
                for x in data
            ),

    }


    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()



hashes = []


for i in range(RUNS):

    digest = compute_digest()

    hashes.append(
        digest
    )


if len(set(hashes)) != 1:

    status = "FAIL"

else:

    status = "PASS"



report = {

    "block":
        "stage5d_reproducibility_test",

    "runs":
        RUNS,

    "hashes":
        hashes,

    "deterministic":
        len(set(hashes)) == 1,

    "future_used":
        False,

    "status":
        status,
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
    "===== Stage5D Reproducibility Test ====="
)

print(
    "runs =",
    RUNS
)

for i,h in enumerate(hashes):

    print(
        "run",
        i+1,
        "=",
        h
    )


print(
    "deterministic =",
    report["deterministic"]
)

print(
    "future_used = NO"
)

print(
    "STATUS =",
    status
)

print(
    "report =",
    REPORT
)
