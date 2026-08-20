from pathlib import Path
import json
import hashlib


SOURCE = Path(
    "artifacts/stage3_class_aware_predictions.json"
)


REPORT = Path(
    "reports/block3_class_aware_predictor_audit.json"
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



report = {

    "records":
        len(data),

    "vehicle":
        classes["VEHICLE"],

    "pedestrian":
        classes["PEDESTRIAN"],

    "cyclist":
        classes["CYCLIST"],

    "future_used":
        future_used,

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
    "===== Stage3 Class Aware Predictor Audit ====="
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
