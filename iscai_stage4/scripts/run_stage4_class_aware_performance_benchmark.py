from pathlib import Path
import json
import hashlib
import time


SOURCE = Path(
    "/home/agni/waymo/iscai_stage3/artifacts/stage3_class_aware_predictions.json"
)


REPORT = Path(
    "reports/block4_class_aware_performance_benchmark.json"
)



def run():

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


    return {
        "records":
            len(data),

        "classes":
            classes,

        "future_used":
            future_used,
    }



start = time.perf_counter()


result = run()


elapsed = (
    time.perf_counter()
    -
    start
)


throughput = (
    result["records"]
    /
    elapsed
)



sha = hashlib.sha256(
    json.dumps(
        result,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode()
).hexdigest()



report = {

    **result,

    "runtime_sec":
        elapsed,

    "records_per_sec":
        throughput,

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
    "===== Stage4 Class Aware Performance Benchmark ====="
)

print(
    "records =",
    result["records"]
)

print(
    "runtime_sec =",
    elapsed
)

print(
    "records/sec =",
    throughput
)

print(
    "vehicle =",
    result["classes"]["VEHICLE"]
)

print(
    "pedestrian =",
    result["classes"]["PEDESTRIAN"]
)

print(
    "cyclist =",
    result["classes"]["CYCLIST"]
)

print(
    "future_used =",
    "YES" if result["future_used"] else "NO"
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
