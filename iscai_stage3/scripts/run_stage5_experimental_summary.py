from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


REPORT = Path(
    "reports/block5f_experimental_summary.json"
)


SOURCES = {
    "stage4h":
        Path(
            "reports/block4h_final_closed_loop_audit.json"
        ),

    "stage5a":
        Path(
            "reports/block5a_runtime_evaluation.json"
        ),

    "stage5b":
        Path(
            "reports/block5b_batch_robustness.json"
        ),

    "stage5c":
        Path(
            "reports/block5c_statistics_evaluation.json"
        ),

    "stage5d":
        Path(
            "reports/block5d_reproducibility_test.json"
        ),

    "stage5e":
        Path(
            "reports/block5e_performance_benchmark.json"
        ),
}


loaded = {}


for name, path in SOURCES.items():

    if not path.exists():
        raise RuntimeError(
            f"Missing required report: {path}"
        )

    loaded[name] = json.loads(
        path.read_text()
    )


# --------------------------------------------------
# 1. Every upstream block must have passed.
# --------------------------------------------------

failed_blocks = [
    name
    for name, data in loaded.items()
    if data.get("status") != "PASS"
]


if failed_blocks:
    raise RuntimeError(
        "Upstream block failure: "
        +
        ", ".join(failed_blocks)
    )


# --------------------------------------------------
# 2. Causal / future-leakage gate.
# --------------------------------------------------

future_blocks = [
    name
    for name, data in loaded.items()
    if data.get("future_used", False)
]


if future_blocks:
    raise RuntimeError(
        "Future leakage reported by: "
        +
        ", ".join(future_blocks)
    )


# --------------------------------------------------
# 3. Cross-report record consistency.
# --------------------------------------------------

prediction_count = int(
    loaded["stage4h"]["prediction_count"]
)

adb_targets = int(
    loaded["stage4h"]["adb_targets"]
)

runtime_records = int(
    loaded["stage5a"]["records"]
)

batch_records = int(
    loaded["stage5b"]["records"]
)

statistics_records = int(
    loaded["stage5c"]["records"]
)

benchmark_records = int(
    loaded["stage5e"]["records"]
)


record_counts = {
    prediction_count,
    adb_targets,
    runtime_records,
    batch_records,
    statistics_records,
    benchmark_records,
}


if len(record_counts) != 1:
    raise RuntimeError(
        "Cross-report record-count mismatch: "
        +
        str(
            sorted(record_counts)
        )
    )


# --------------------------------------------------
# 4. Model coverage.
# --------------------------------------------------

model_stats = loaded[
    "stage5c"
]["models"]


models = sorted(
    model_stats.keys()
)


expected_models = {
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
    "IMM",
    "MHT",
}


if set(models) != expected_models:
    raise RuntimeError(
        "Unexpected model coverage: "
        +
        str(models)
    )


model_counts = {
    model:
        int(
            model_stats[model]["count"]
        )
    for model in models
}


if sum(
    model_counts.values()
) != statistics_records:

    raise RuntimeError(
        "Model counts do not sum to record count."
    )


# --------------------------------------------------
# 5. Robustness / determinism gates.
# --------------------------------------------------

robustness_failures = int(
    loaded["stage5b"]["failures"]
)


if robustness_failures != 0:
    raise RuntimeError(
        "Robustness failures detected."
    )


deterministic = bool(
    loaded["stage5d"]["deterministic"]
)


if not deterministic:
    raise RuntimeError(
        "Reproducibility gate failed."
    )


# --------------------------------------------------
# 6. Closed-loop validation.
# --------------------------------------------------

constraints_passed = bool(
    loaded["stage4h"]["constraints_passed"]
)


if not constraints_passed:
    raise RuntimeError(
        "Controller constraints failed."
    )


selected_target = int(
    loaded["stage4h"]["selected_target"]
)

beam_id = int(
    loaded["stage4h"]["beam_id"]
)

replay_actions = int(
    loaded["stage4h"]["replay_actions"]
)


# --------------------------------------------------
# 7. Performance sanity.
# --------------------------------------------------

runtime_sec = float(
    loaded["stage5e"]["total_runtime_sec"]
)

throughput = float(
    loaded["stage5e"]["trajectories_per_sec"]
)


if (
    not math.isfinite(runtime_sec)
    or
    runtime_sec <= 0.0
):
    raise RuntimeError(
        "Invalid benchmark runtime."
    )


if (
    not math.isfinite(throughput)
    or
    throughput <= 0.0
):
    raise RuntimeError(
        "Invalid benchmark throughput."
    )


# --------------------------------------------------
# 8. Build deterministic audit chain.
#
# Do NOT hash measured runtime values here:
# wall-clock timings are intentionally nondeterministic.
# --------------------------------------------------

source_hashes = {
    name:
        data.get("sha256")
    for name, data in loaded.items()
}


audit_payload = {
    "records":
        prediction_count,

    "models":
        models,

    "model_counts":
        model_counts,

    "selected_target":
        selected_target,

    "beam_id":
        beam_id,

    "replay_actions":
        replay_actions,

    "constraints_passed":
        constraints_passed,

    "robustness_failures":
        robustness_failures,

    "deterministic":
        deterministic,

    "future_used":
        False,

    "source_hashes":
        source_hashes,
}


digest = hashlib.sha256(
    json.dumps(
        audit_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


report = {
    "block":
        "stage5f_experimental_summary",

    "source_blocks":
        list(SOURCES.keys()),

    "passed_blocks":
        len(SOURCES),

    "records":
        prediction_count,

    "models":
        models,

    "model_counts":
        model_counts,

    "selected_target":
        selected_target,

    "beam_id":
        beam_id,

    "replay_actions":
        replay_actions,

    "constraints_passed":
        constraints_passed,

    "robustness_failures":
        robustness_failures,

    "deterministic":
        deterministic,

    "benchmark_runtime_sec":
        runtime_sec,

    "benchmark_trajectories_per_sec":
        throughput,

    "future_used":
        False,

    "audit_sha256":
        digest,

    "status":
        "PASS",
}


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5F Experimental Summary ====="
)

print(
    "passed_blocks =",
    report["passed_blocks"],
)

print(
    "records =",
    report["records"],
)

print(
    "models =",
    report["models"],
)

print(
    "selected_target =",
    report["selected_target"],
)

print(
    "beam_id =",
    report["beam_id"],
)

print(
    "replay_actions =",
    report["replay_actions"],
)

print(
    "constraints = PASS"
)

print(
    "robustness_failures =",
    report["robustness_failures"],
)

print(
    "deterministic =",
    report["deterministic"],
)

print(
    "benchmark_runtime_sec =",
    report["benchmark_runtime_sec"],
)

print(
    "trajectories/sec =",
    report["benchmark_trajectories_per_sec"],
)

print(
    "future_used = NO"
)

print(
    "AUDIT SHA256 =",
    digest,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)
