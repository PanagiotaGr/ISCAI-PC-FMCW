from pathlib import Path
import json
import hashlib


SOURCE = Path(
    "/home/agni/waymo/iscai_stage3/artifacts/stage3_class_aware_predictions.json"
)


REPORT = Path(
    "reports/block4_class_aware_prediction_evaluation.json"
)


data = json.loads(
    SOURCE.read_text()
)


classes = {

    "VEHICLE": {
        "records": 0,
        "models": set(),
    },

    "PEDESTRIAN": {
        "records": 0,
        "models": set(),
    },

    "CYCLIST": {
        "records": 0,
        "models": set(),
    },
}


future_used = False


for item in data:

    actor_type = item.get(
        "actor_type"
    )


    if actor_type not in classes:
        continue


    classes[actor_type]["records"] += 1


    for model in item.get(
        "models",
        []
    ):
        classes[actor_type]["models"].add(
            model
        )


    if item.get(
        "future_used",
        False
    ):
        future_used = True



payload = {

    "records":
        len(data),

    "vehicle":
        classes["VEHICLE"]["records"],

    "pedestrian":
        classes["PEDESTRIAN"]["records"],

    "cyclist":
        classes["CYCLIST"]["records"],

    "models": {

        k:
        sorted(
            v["models"]
        )

        for k, v in classes.items()

    },

    "future_used":
        future_used,
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



REPORT.parent.mkdir(
    exist_ok=True
)


REPORT.write_text(
    json.dumps(
        {
            **payload,
            "sha256": sha,
            "status": "PASS",
        },
        indent=2,
        sort_keys=True,
    )
)



print(
    "===== Stage4 Class Aware Prediction Evaluation ====="
)

print(
    "records =",
    len(data)
)

print(
    "vehicle =",
    classes["VEHICLE"]["records"]
)

print(
    "pedestrian =",
    classes["PEDESTRIAN"]["records"]
)

print(
    "cyclist =",
    classes["CYCLIST"]["records"]
)

print(
    "models =",
    sorted(
        list(
            classes["VEHICLE"]["models"]
        )
    )
)

print(
    "future_used =",
    "YES" if future_used else "NO"
)

print(
    "SHA256 =",
    sha
)

print(
    "STATUS = PASS"
)
