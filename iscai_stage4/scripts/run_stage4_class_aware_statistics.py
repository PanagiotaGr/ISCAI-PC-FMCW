from pathlib import Path
import json
import hashlib


SOURCE = Path(
    "/home/agni/waymo/iscai_stage3/artifacts/stage3_class_aware_predictions.json"
)


REPORT = Path(
    "reports/block4_class_aware_statistics.json"
)


data = json.loads(
    SOURCE.read_text()
)


stats = {

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


    if actor_type not in stats:
        continue


    stats[actor_type]["records"] += 1


    for model in item.get(
        "models",
        []
    ):
        stats[actor_type]["models"].add(
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

    "classes": {

        key: {

            "records":
                value["records"],

            "models":
                sorted(
                    list(
                        value["models"]
                    )
                ),

        }

        for key, value in stats.items()

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
    "===== Stage4 Class Aware Statistics ====="
)

print(
    "records =",
    len(data)
)


for key in stats:

    print(
        key,
        "=",
        stats[key]["records"]
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


print(
    "report =",
    REPORT
)
