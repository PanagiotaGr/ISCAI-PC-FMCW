from __future__ import annotations

import json
import hashlib
from pathlib import Path



REPORT = Path(
    "reports/block4h_final_closed_loop_audit.json"
)


SOURCES = {

    "prediction_quality":
        "reports/block3l_real_prediction_quality_audit.json",

    "adb_prediction":
        "reports/block4b_real_adb_from_prediction_smoke.json",

    "closed_loop":
        "reports/block4e_full_closed_loop_smoke.json",

    "temporal_replay":
        "reports/block4f_temporal_closed_loop_replay.json",

    "constraints":
        "reports/block4g_controller_constraint_smoke.json",
}



loaded = {}



for name, path in SOURCES.items():

    p = Path(path)

    if not p.exists():

        raise RuntimeError(
            f"Missing report: {path}"
        )


    loaded[name] = json.loads(
        p.read_text()
    )



# -------------------------
# Cross validation
# -------------------------


if loaded["closed_loop"]["future_used"]:

    raise RuntimeError(
        "Future leakage in closed loop."
    )


if loaded["temporal_replay"]["future_used"]:

    raise RuntimeError(
        "Future leakage in replay."
    )


if not loaded["constraints"]["beam_ids_valid"]:

    raise RuntimeError(
        "Invalid beam ids."
    )


if not loaded["constraints"]["time_contiguous"]:

    raise RuntimeError(
        "Non contiguous controller timeline."
    )



chain = {

    "prediction_count":
        loaded["closed_loop"]["prediction_count"],

    "adb_targets":
        loaded["adb_prediction"]["targets"],

    "closed_loop_target":
        loaded["closed_loop"]["selected_target"],

    "beam_id":
        loaded["closed_loop"]["beam_id"],

    "replay_actions":
        loaded["temporal_replay"]["actions"],

    "constraint_actions":
        loaded["constraints"]["actions"],

    "future_used":
        False,

}



digest = hashlib.sha256(
    json.dumps(
        chain,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()



report = {

    "block":
        "stage4h_final_closed_loop_audit",

    "components":
        list(SOURCES.keys()),

    "prediction_count":
        chain["prediction_count"],

    "adb_targets":
        chain["adb_targets"],

    "selected_target":
        chain["closed_loop_target"],

    "beam_id":
        chain["beam_id"],

    "replay_actions":
        chain["replay_actions"],

    "constraints_passed":
        True,

    "future_used":
        False,

    "sha256":
        digest,

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
    "===== Stage4H Final Closed Loop Audit ====="
)

print(
    "components =",
    len(SOURCES)
)

print(
    "predictions =",
    report["prediction_count"]
)

print(
    "adb_targets =",
    report["adb_targets"]
)

print(
    "selected_target =",
    report["selected_target"]
)

print(
    "beam_id =",
    report["beam_id"]
)

print(
    "replay_actions =",
    report["replay_actions"]
)

print(
    "constraints = PASS"
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    digest
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)
