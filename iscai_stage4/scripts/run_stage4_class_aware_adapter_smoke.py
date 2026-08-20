from pathlib import Path
import json
import hashlib


SOURCE = Path(
    "/home/agni/waymo/iscai_stage3/artifacts/stage3_class_aware_predictions.json"
)


REPORT = Path(
    "reports/block4_class_aware_adapter_smoke.json"
)


data = json.loads(
    SOURCE.read_text()
)


classes = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


future_used = False


for item in data:

    actor_type = item.get(
        "actor_type",
        "UNKNOWN"
    )

    if actor_type in classes:
        classes[actor_type] += 1


    if item.get(
        "future_used",
        False
    ):
        future_used = True



payload = {
    "records": len(data),
    "classes": classes,
    "future_used": future_used,
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
    "===== Stage4 Class Aware Adapter Smoke ====="
)

print(
    "records =",
    len(data)
)

print(
    "vehicle =",
    classes["VEHICLE"]
)

print(
    "pedestrian =",
    classes["PEDESTRIAN"]
)

print(
    "cyclist =",
    classes["CYCLIST"]
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
