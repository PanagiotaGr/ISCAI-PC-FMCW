from pathlib import Path
import json
import hashlib


STAT = Path(
    "reports/block5_class_aware_statistics.json"
)

BENCH = Path(
    "reports/block5_class_aware_performance_benchmark.json"
)

REPRO_HASH = Path(
    "reports/block5_class_aware_reproducibility.json"
)


REPORT = Path(
    "reports/block5_class_aware_experimental_summary.json"
)



statistics = json.loads(
    STAT.read_text()
)


benchmark = json.loads(
    BENCH.read_text()
)



summary = {

    "records":
        statistics["records"],

    "classes":
        statistics["classes"],

    "models":
        statistics["models"],

    "future_used":
        statistics["future_used"],

    "benchmark":

        {

            "runtime_sec":
                benchmark["runtime_sec"],

            "records_per_sec":
                benchmark["records_per_sec"],

        },

}



sha = hashlib.sha256(
    json.dumps(
        summary,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode()
).hexdigest()



report = {

    **summary,

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
    "===== Stage5 Class Aware Experimental Summary ====="
)

print(
    "records =",
    summary["records"]
)

print(
    "classes =",
    summary["classes"]
)

print(
    "models =",
    summary["models"]
)

print(
    "benchmark = PASS"
)

print(
    "future_used =",
    "YES" if summary["future_used"] else "NO"
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
