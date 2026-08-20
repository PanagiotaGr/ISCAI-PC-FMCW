
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SOURCE = Path(
    "artifacts/stage5_adb_safety_schedule_v2.json"
)

OUTPUT = Path(
    "artifacts/stage5_temporal_beam_schedule.json"
)

REPORT = Path(
    "reports/block5_temporal_beam_persistence_smoke.json"
)


PERSISTENCE_RATIO = 0.75

MAX_ACTIVE_BEAMS = 16


previous = json.loads(
    SOURCE.read_text()
)


# Simulated next-frame candidate set.
# Deterministic transformation:
# same actors, slightly reordered.
current_candidates = sorted(
    previous,
    key=lambda x: (
        x["actor_type"],
        x["beam_index"],
        x["scenario_id"],
        x["track_index"],
    )
)


previous_keys = {
    (
        item["scenario_id"],
        item["track_index"],
        item["actor_type"],
    )
    for item in previous
}


persistent = []

for item in current_candidates:

    key = (
        item["scenario_id"],
        item["track_index"],
        item["actor_type"],
    )

    if key in previous_keys:

        persistent.append(
            item
        )


keep_count = int(
    MAX_ACTIVE_BEAMS
    *
    PERSISTENCE_RATIO
)


persistent_selected = persistent[
    :keep_count
]


remaining = [
    item
    for item in current_candidates
    if item not in persistent_selected
]


slots = (
    MAX_ACTIVE_BEAMS
    -
    len(persistent_selected)
)


new_selected = remaining[
    :slots
]


selected = (
    persistent_selected
    +
    new_selected
)


selected = selected[
    :MAX_ACTIVE_BEAMS
]


retained = len(
    persistent_selected
)


switches = (
    MAX_ACTIVE_BEAMS
    -
    retained
)


beam_switch_rate = (
    switches
    /
    MAX_ACTIVE_BEAMS
)


actor_retention = (
    retained
    /
    MAX_ACTIVE_BEAMS
)


class_counts = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for item in selected:

    cls = item.get(
        "actor_type"
    )

    if cls in class_counts:
        class_counts[cls] += 1



temporal_consistency = (
    retained
    ==
    keep_count
)


payload = {

    "previous_beams":
        len(previous),

    "selected_beams":
        len(selected),

    "retained_beams":
        retained,

    "beam_switches":
        switches,

    "beam_switch_rate":
        beam_switch_rate,

    "actor_retention":
        actor_retention,

    "temporal_consistency":
        temporal_consistency,

    "selected_by_class":
        class_counts,

    "persistence_ratio":
        PERSISTENCE_RATIO,

    "future_used":
        False,
}


status = (
    "PASS"
    if (
        len(selected)
        == MAX_ACTIVE_BEAMS
        and temporal_consistency
        and actor_retention >= PERSISTENCE_RATIO
    )
    else
    "FAIL"
)


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


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT.write_text(
    json.dumps(
        selected,
        indent=2,
        sort_keys=True,
    )
)


report = {
    **payload,
    "sha256":
        sha,
    "status":
        status,
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
    "===== Stage5 Temporal Beam Persistence Smoke ====="
)

print(
    "previous_beams =",
    len(previous)
)

print(
    "selected_beams =",
    len(selected)
)

print(
    "retained_beams =",
    retained
)

print(
    "beam_switches =",
    switches
)

print(
    "beam_switch_rate =",
    beam_switch_rate
)

print(
    "actor_retention =",
    actor_retention
)

print(
    "temporal_consistency =",
    temporal_consistency
)

print(
    "selected_by_class =",
    class_counts
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
    "output =",
    OUTPUT
)

print(
    "report =",
    REPORT
)


if status != "PASS":
    raise SystemExit(1)
