from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


REPORTS = {
    "5A_adapter": Path(
        "reports/block5_stage4_adapter_smoke.json"
    ),
    "5B_evaluation": Path(
        "reports/block5_probabilistic_evaluation_smoke.json"
    ),
    "5C_statistics": Path(
        "reports/block5_statistics_evaluation.json"
    ),
    "5D_reproducibility": Path(
        "reports/block5_reproducibility_test.json"
    ),
    "5E_benchmark": Path(
        "reports/block5_performance_benchmark.json"
    ),
    "5F_summary": Path(
        "reports/block5_experimental_summary.json"
    ),
}


OUTPUT = Path(
    "reports/block5_final_audit.json"
)


# --------------------------------------------------
# Load all Stage5 reports
# --------------------------------------------------

loaded = {}

for name, path in REPORTS.items():

    if not path.exists():
        raise RuntimeError(
            f"Missing Stage5 report: {path}"
        )

    loaded[name] = json.loads(
        path.read_text()
    )


# --------------------------------------------------
# Validate PASS status
# --------------------------------------------------

for name, report in loaded.items():

    if report.get("status") != "PASS":
        raise RuntimeError(
            f"{name} did not PASS."
        )


# --------------------------------------------------
# Causality / future leakage
# --------------------------------------------------

for name, report in loaded.items():

    if report.get("future_used") is not False:
        raise RuntimeError(
            f"Future leakage status invalid in {name}."
        )


# --------------------------------------------------
# Cross-report consistency
# --------------------------------------------------

adapter = loaded["5A_adapter"]
evaluation = loaded["5B_evaluation"]
statistics = loaded["5C_statistics"]
reproducibility = loaded["5D_reproducibility"]
benchmark = loaded["5E_benchmark"]
summary = loaded["5F_summary"]


prediction_record_counts = {
    adapter["records"],
    evaluation["records"],
    statistics["records"],
    benchmark["records"],
    summary["records"],
}


if len(prediction_record_counts) != 1:
    raise RuntimeError(
        "Prediction-domain record mismatch."
    )


records = next(
    iter(prediction_record_counts)
)


if records <= 0:
    raise RuntimeError(
        "No Stage5 prediction records."
    )


models = adapter["models"]


if sorted(evaluation["model_counts"].keys()) != sorted(models):
    raise RuntimeError(
        "Model mismatch between adapter and evaluation."
    )


if sorted(statistics["models"]) != sorted(models):
    raise RuntimeError(
        "Model mismatch in statistics."
    )


if sorted(summary["models"]) != sorted(models):
    raise RuntimeError(
        "Model mismatch in experimental summary."
    )


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

if reproducibility["deterministic"] is not True:
    raise RuntimeError(
        "Stage5 is not deterministic."
    )


hashes = reproducibility["hashes"]


if len(hashes) != reproducibility["runs"]:
    raise RuntimeError(
        "Invalid reproducibility run count."
    )


if len(set(hashes)) != 1:
    raise RuntimeError(
        "Reproducibility hashes differ."
    )


# --------------------------------------------------
# Numerical validation
# --------------------------------------------------

if evaluation["finite"] is not True:
    raise RuntimeError(
        "Evaluation contains non-finite values."
    )


throughput = benchmark[
    "predictions_per_sec"
]

runtime = benchmark[
    "runtime_sec"
]


if not math.isfinite(runtime) or runtime < 0.0:
    raise RuntimeError(
        "Invalid benchmark runtime."
    )


if not math.isfinite(throughput) or throughput < 0.0:
    raise RuntimeError(
        "Invalid benchmark throughput."
    )


# --------------------------------------------------
# Experimental summary consistency
# --------------------------------------------------

if summary["passed_blocks"] != 5:
    raise RuntimeError(
        "Experimental summary block count mismatch."
    )


if summary["deterministic"] is not True:
    raise RuntimeError(
        "Experimental summary is not deterministic."
    )


# --------------------------------------------------
# Deterministic audit digest
#
# Do NOT include runtime/throughput here because
# benchmark timing naturally changes between runs.
# --------------------------------------------------

payload = {
    "components": len(REPORTS),
    "records": records,
    "models": sorted(models),
    "reproducibility_hash": hashes[0],
    "adapter_sha256": adapter["sha256"],
    "evaluation_sha256": evaluation["sha256"],
    "statistics_sha256": statistics["sha256"],
    "future_used": False,
}


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


final_report = {
    "block": "stage5_final_audit",
    "components": len(REPORTS),
    "records": records,
    "models": sorted(models),
    "adapter": "PASS",
    "evaluation": "PASS",
    "statistics": "PASS",
    "reproducibility": "PASS",
    "benchmark": "PASS",
    "experimental_summary": "PASS",
    "deterministic": True,
    "future_used": False,
    "sha256": sha,
    "status": "PASS",
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT.write_text(
    json.dumps(
        final_report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5G Final Audit ====="
)

print(
    "components =",
    final_report["components"],
)

print(
    "records =",
    final_report["records"],
)

print(
    "models =",
    final_report["models"],
)

print("adapter = PASS")
print("evaluation = PASS")
print("statistics = PASS")
print("reproducibility = PASS")
print("benchmark = PASS")
print("experimental_summary = PASS")

print(
    "deterministic = True"
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    sha,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    OUTPUT,
)
