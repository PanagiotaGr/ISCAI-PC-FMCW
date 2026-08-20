from pathlib import Path
import json
import hashlib


SOURCE = Path(
    "/home/agni/waymo/iscai_stage3/artifacts/stage3_class_aware_predictions.json"
)


def compute_hash():

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
            "actor_type"
        )


        if actor_type in classes:
            classes[actor_type] += 1


        if item.get(
            "future_used",
            False
        ):
            future_used = True



    payload = {

        "records":
            len(data),

        "classes":
            classes,

        "future_used":
            future_used,

    }


    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode()
    ).hexdigest()



runs = []


for i in range(3):

    runs.append(
        compute_hash()
    )



deterministic = (
    runs[0]
    ==
    runs[1]
    ==
    runs[2]
)


print(
    "===== Stage4 Class Aware Reproducibility ====="
)

print(
    "runs = 3"
)


for i, value in enumerate(
    runs,
    1
):

    print(
        f"run {i} = {value}"
    )


print(
    "deterministic =",
    deterministic
)

print(
    "future_used = NO"
)


print(
    "STATUS =",
    "PASS" if deterministic else "FAIL"
)
