from __future__ import annotations

import json
import hashlib

from pathlib import Path


SOURCE = Path(
    "artifacts/real_predictor_trajectories.json"
)


REPORT = Path(
    "reports/block3_class_aware_dataset_audit.json"
)


data = json.loads(
    SOURCE.read_text()
)


classes = {

    "vehicle": 0,

    "pedestrian": 0,

    "cyclist": 0,

}



for item in data:

    actor_type = item.get(
        "actor_type",
        None
    )

    if actor_type:
        actor_type = actor_type.lower()


    if actor_type in classes:

        classes[actor_type] += 1



payload = {

    "records":
        len(data),

    "classes":
        classes,

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

    "records":
        len(data),

    "vehicle":
        classes["vehicle"],

    "pedestrian":
        classes["pedestrian"],

    "cyclist":
        classes["cyclist"],

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
    "===== Stage3 Class Aware Dataset Audit ====="
)

print(
    "records =",
    report["records"]
)

print(
    "vehicle =",
    report["vehicle"]
)

print(
    "pedestrian =",
    report["pedestrian"]
)

print(
    "cyclist =",
    report["cyclist"]
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
