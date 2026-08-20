#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# PATHS
# ============================================================

REPORT_DIR = Path("reports/stage4_auto_cuda")

RESULT_FILES = [
    REPORT_DIR / "all_results.json",
    REPORT_DIR / "worker_0_results.json",
    REPORT_DIR / "worker_1_results.json",
]

OUTPUT = REPORT_DIR / "stage4_final_scientific_audit.json"

TARGET_EXPERIMENTS = 48


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"WARNING: cannot read {path}: {exc}")
        return None


def metric(result, key):
    return (
        result
        .get("metrics", {})
        .get(key, float("inf"))
    )


def best_by(results, key):
    if not results:
        return None

    valid = [
        r for r in results
        if metric(r, key) != float("inf")
    ]

    if not valid:
        return None

    return min(
        valid,
        key=lambda r: metric(r, key)
    )


def compact_result(result):
    if result is None:
        return None

    return {
        "name": result.get("name"),
        "class": result.get("class"),
        "config": result.get("config"),
        "metrics": result.get("metrics"),
        "time_seconds": result.get("time"),
    }


# ============================================================
# LOAD + DEDUPLICATE RESULTS
# ============================================================

all_results = {}

print("=" * 64)
print("STAGE 4 FINAL SCIENTIFIC AUDIT")
print("=" * 64)

for path in RESULT_FILES:

    data = load_json(path)

    if data is None:
        print(f"{path.name}: NOT FOUND")
        continue

    if not isinstance(data, list):
        print(f"{path.name}: INVALID FORMAT")
        continue

    print(f"{path.name}: {len(data)} results")

    for result in data:

        name = result.get("name")

        if not name:
            continue

        all_results[name] = result


results = list(all_results.values())


# ============================================================
# SPLIT MODEL TYPES
# ============================================================

deterministic = [
    r for r in results
    if r.get("config", {}).get("model")
    == "deterministic"
]

gaussian = [
    r for r in results
    if r.get("config", {}).get("model")
    == "gaussian"
]


# ============================================================
# BEST MODELS
# ============================================================

best_det_loss = best_by(
    deterministic,
    "loss"
)

best_det_ade = best_by(
    deterministic,
    "ADE"
)

best_det_fde = best_by(
    deterministic,
    "FDE"
)


best_gauss_loss = best_by(
    gaussian,
    "loss"
)

best_gauss_ade = best_by(
    gaussian,
    "ADE"
)

best_gauss_fde = best_by(
    gaussian,
    "FDE"
)


# ============================================================
# COMPLETION STATUS
# ============================================================

completed = len(results)

search_complete = (
    completed >= TARGET_EXPERIMENTS
)

deterministic_available = (
    len(deterministic) > 0
)

gaussian_available = (
    len(gaussian) > 0
)


# Calibration and classical-baseline comparison are deliberately
# not inferred from training loss files.
#
# They require dedicated evaluation artifacts.

calibration_candidates = [
    REPORT_DIR / "stage4_calibration.json",
    REPORT_DIR / "stage4_calibration_report.json",
    REPORT_DIR / "stage4_gaussian_calibration.json",
]

baseline_candidates = [
    REPORT_DIR / "stage4_baseline_comparison.json",
    REPORT_DIR / "stage4_classical_baselines.json",
]

calibration_file = next(
    (
        str(p)
        for p in calibration_candidates
        if p.exists()
    ),
    None,
)

baseline_file = next(
    (
        str(p)
        for p in baseline_candidates
        if p.exists()
    ),
    None,
)

calibration_available = (
    calibration_file is not None
)

baseline_comparison_available = (
    baseline_file is not None
)


# Stage-4 scientific completion requires all components.
scientific_complete = all([
    search_complete,
    deterministic_available,
    gaussian_available,
    calibration_available,
    baseline_comparison_available,
])


if scientific_complete:
    status = "READY_FOR_FINAL_ACCEPTANCE_CHECK"
else:
    status = "INCOMPLETE"


# ============================================================
# REPORT
# ============================================================

report = {

    "status": status,

    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "experiment_search": {
        "target": TARGET_EXPERIMENTS,
        "completed_unique": completed,
        "remaining": max(
            0,
            TARGET_EXPERIMENTS - completed
        ),
        "progress_fraction": (
            completed / TARGET_EXPERIMENTS
        ),
        "search_complete": search_complete,
    },

    "model_counts": {
        "deterministic": len(deterministic),
        "gaussian": len(gaussian),
    },

    "best_deterministic": {
        "validation_loss":
            compact_result(best_det_loss),
        "ADE":
            compact_result(best_det_ade),
        "FDE":
            compact_result(best_det_fde),
    },

    "best_gaussian": {
        "NLL_or_validation_loss":
            compact_result(best_gauss_loss),
        "ADE":
            compact_result(best_gauss_ade),
        "FDE":
            compact_result(best_gauss_fde),
    },

    "required_scientific_components": {

        "deterministic_predictor":
            deterministic_available,

        "probabilistic_gaussian_predictor":
            gaussian_available,

        "trajectory_search_complete":
            search_complete,

        "calibration_evaluation_available":
            calibration_available,

        "classical_baseline_comparison_available":
            baseline_comparison_available,
    },

    "artifacts": {
        "calibration":
            calibration_file,

        "classical_baselines":
            baseline_file,
    },

    "acceptance_note": (
        "Do not mark Stage 4 PASS from hyperparameter-search "
        "results alone. Final acceptance requires probabilistic "
        "trajectory evaluation, uncertainty calibration, and "
        "comparison against classical baseline(s)."
    ),
}


OUTPUT.write_text(
    json.dumps(
        report,
        indent=2
    )
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 64)
print("EXPERIMENT STATUS")
print("=" * 64)

print(
    f"COMPLETED : {completed}/{TARGET_EXPERIMENTS}"
)

print(
    f"REMAINING : "
    f"{max(0, TARGET_EXPERIMENTS-completed)}"
)

print(
    f"PROGRESS  : "
    f"{100*completed/TARGET_EXPERIMENTS:.1f}%"
)

print()

print(
    "DETERMINISTIC:",
    len(deterministic)
)

print(
    "GAUSSIAN     :",
    len(gaussian)
)


print()
print("=" * 64)
print("BEST DETERMINISTIC")
print("=" * 64)

for label, result in [
    ("LOSS", best_det_loss),
    ("ADE", best_det_ade),
    ("FDE", best_det_fde),
]:

    if result is None:
        print(label, ": NONE")
        continue

    print()
    print(label)
    print(" ", result["name"])
    print(
        " ",
        result.get("metrics")
    )


print()
print("=" * 64)
print("BEST GAUSSIAN")
print("=" * 64)

for label, result in [
    ("NLL/LOSS", best_gauss_loss),
    ("ADE", best_gauss_ade),
    ("FDE", best_gauss_fde),
]:

    if result is None:
        print(label, ": NONE")
        continue

    print()
    print(label)
    print(" ", result["name"])
    print(
        " ",
        result.get("metrics")
    )


print()
print("=" * 64)
print("SCIENTIFIC COMPLETION CHECK")
print("=" * 64)

checks = report[
    "required_scientific_components"
]

for key, value in checks.items():
    print(
        f"{key:42s}",
        "OK" if value else "MISSING"
    )


print()
print("STATUS:", status)
print("REPORT:", OUTPUT)
