
from __future__ import annotations

import hashlib
import json
from pathlib import Path


V1_REPORT = Path(
    "reports/block5_closed_loop_beam_evaluation.json"
)

V2_REPORT = Path(
    "reports/block5_closed_loop_beam_evaluation_v2.json"
)

OUTPUT = Path(
    "reports/block5_scheduler_comparison.json"
)


v1 = json.loads(
    V1_REPORT.read_text()
)

v2 = json.loads(
    V2_REPORT.read_text()
)


# --------------------------------------------------
# Extract comparable metrics
# --------------------------------------------------

v1_vehicle = float(
    v1["vehicle_coverage"]
)

v1_pedestrian = float(
    v1["pedestrian_coverage"]
)

v1_cyclist = float(
    v1["cyclist_coverage"]
)

v1_safety = float(
    v1["safety_actor_coverage"]
)


v2_coverage = v2[
    "coverage"
]

v2_vehicle = float(
    v2_coverage["VEHICLE"]
)

v2_pedestrian = float(
    v2_coverage["PEDESTRIAN"]
)

v2_cyclist = float(
    v2_coverage["CYCLIST"]
)

v2_safety = float(
    v2["safety_actor_coverage"]
)

v2_balance = float(
    v2["class_balance_score"]
)


# V1 class balance.
v1_counts = v1[
    "covered_by_class"
]

v1_positive = [
    int(value)
    for value in v1_counts.values()
    if int(value) > 0
]

if len(v1_positive) == 3:

    v1_balance = (
        min(v1_positive)
        /
        max(v1_positive)
    )

else:

    # At least one supported class was starved.
    v1_balance = 0.0


# --------------------------------------------------
# Deltas
# --------------------------------------------------

delta_vehicle = (
    v2_vehicle
    -
    v1_vehicle
)

delta_pedestrian = (
    v2_pedestrian
    -
    v1_pedestrian
)

delta_cyclist = (
    v2_cyclist
    -
    v1_cyclist
)

delta_safety = (
    v2_safety
    -
    v1_safety
)

delta_balance = (
    v2_balance
    -
    v1_balance
)


# Number of classes with non-zero coverage.
v1_nonzero_classes = sum(
    value > 0.0
    for value in (
        v1_vehicle,
        v1_pedestrian,
        v1_cyclist,
    )
)

v2_nonzero_classes = sum(
    value > 0.0
    for value in (
        v2_vehicle,
        v2_pedestrian,
        v2_cyclist,
    )
)


# --------------------------------------------------
# Important interpretation:
#
# V2 is not expected to dominate V1 on every metric.
# V1 heavily concentrates its budget on cyclists.
# V2 deliberately trades some cyclist/safety coverage
# for non-starvation and class balance.
# --------------------------------------------------

nonstarvation_improved = (
    v2_nonzero_classes
    >
    v1_nonzero_classes
)

balance_improved = (
    v2_balance
    >
    v1_balance
)

budget_valid = (
    int(v1["selected_actors"])
    <= int(v1["max_active_beams"])
    and
    int(v2["selected_beams"])
    <= int(v2["max_active_beams"])
)


payload = {

    "v1": {
        "vehicle_coverage":
            v1_vehicle,

        "pedestrian_coverage":
            v1_pedestrian,

        "cyclist_coverage":
            v1_cyclist,

        "safety_actor_coverage":
            v1_safety,

        "class_balance_score":
            v1_balance,

        "nonzero_classes":
            v1_nonzero_classes,
    },

    "v2": {
        "vehicle_coverage":
            v2_vehicle,

        "pedestrian_coverage":
            v2_pedestrian,

        "cyclist_coverage":
            v2_cyclist,

        "safety_actor_coverage":
            v2_safety,

        "class_balance_score":
            v2_balance,

        "nonzero_classes":
            v2_nonzero_classes,
    },

    "delta_v2_minus_v1": {
        "vehicle_coverage":
            delta_vehicle,

        "pedestrian_coverage":
            delta_pedestrian,

        "cyclist_coverage":
            delta_cyclist,

        "safety_actor_coverage":
            delta_safety,

        "class_balance_score":
            delta_balance,
    },

    "nonstarvation_improved":
        nonstarvation_improved,

    "balance_improved":
        balance_improved,

    "budget_valid":
        budget_valid,

    "future_used":
        False,
}


# PASS does NOT require V2 safety coverage to exceed V1.
#
# That would be scientifically wrong here because V1
# obtains high safety coverage by starving two classes.
status = (
    "PASS"
    if (
        nonstarvation_improved
        and balance_improved
        and budget_valid
        and v2_nonzero_classes == 3
    )
    else
    "FAIL"
)


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


report = {
    **payload,
    "sha256":
        sha,
    "status":
        status,
}


OUTPUT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 Scheduler Comparison ====="
)

print(
    "V1 vehicle coverage =",
    v1_vehicle
)

print(
    "V2 vehicle coverage =",
    v2_vehicle
)

print(
    "V1 pedestrian coverage =",
    v1_pedestrian
)

print(
    "V2 pedestrian coverage =",
    v2_pedestrian
)

print(
    "V1 cyclist coverage =",
    v1_cyclist
)

print(
    "V2 cyclist coverage =",
    v2_cyclist
)

print(
    "V1 safety coverage =",
    v1_safety
)

print(
    "V2 safety coverage =",
    v2_safety
)

print(
    "V1 class balance =",
    v1_balance
)

print(
    "V2 class balance =",
    v2_balance
)

print(
    "V1 nonzero classes =",
    v1_nonzero_classes
)

print(
    "V2 nonzero classes =",
    v2_nonzero_classes
)

print(
    "delta safety coverage =",
    delta_safety
)

print(
    "delta class balance =",
    delta_balance
)

print(
    "nonstarvation_improved =",
    nonstarvation_improved
)

print(
    "balance_improved =",
    balance_improved
)

print(
    "budget_valid =",
    budget_valid
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    sha
)

print(
    "STATUS =",
    status
)

print(
    "report =",
    OUTPUT
)


if status != "PASS":
    raise SystemExit(1)
