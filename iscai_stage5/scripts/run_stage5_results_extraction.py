
from pathlib import Path
import json
import hashlib


SOURCE = Path(
    "reports/stage5_scientific_evaluation_summary.json"
)

REPORT = Path(
    "reports/stage5_results_table.json"
)


data = json.loads(
    SOURCE.read_text()
)


results = {

    "system": {
        "status":
            data["system_status"],

        "deterministic":
            data["deterministic"],

        "future_used":
            data["future_used"],
    },


    "IMM": {

        "normalized_entropy":
            data["IMM"]
            ["normalized_entropy"],

        "nonuniform_fraction":
            data["IMM"]
            ["nonuniform_fraction"],

        "dominant_models":
            data["IMM"]
            ["dominant_models"],
    },


    "MHT": {

        "selected_lambda":
            data["MHT"]
            ["selected_lambda"],

        "ranking_change_rate":
            data["MHT"]
            ["change_rate"],

        "best_after":
            data["MHT"]
            ["best_after"],
    },


    "Beam_Scheduler": {

        "V1_class_balance":
            data["beam_scheduler"]
            ["V1_class_balance"],

        "V2_class_balance":
            data["beam_scheduler"]
            ["V2_class_balance"],

        "V1_vehicle_coverage":
            data["beam_scheduler"]
            ["V1_vehicle_coverage"],

        "V2_vehicle_coverage":
            data["beam_scheduler"]
            ["V2_vehicle_coverage"],

    },


    "Class_Aware": {

        "classes":
            data["class_analysis"],
    },
}


sha = hashlib.sha256(
    json.dumps(
        results,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode()
).hexdigest()


output = {

    "results":
        results,

    "sha256":
        sha,

    "status":
        "PASS",
}


REPORT.write_text(
    json.dumps(
        output,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 Results Extraction ====="
)

print(
    "IMM entropy =",
    results["IMM"]
    ["normalized_entropy"]
)

print(
    "MHT change rate =",
    results["MHT"]
    ["ranking_change_rate"]
)

print(
    "V2 balance =",
    results["Beam_Scheduler"]
    ["V2_class_balance"]
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
