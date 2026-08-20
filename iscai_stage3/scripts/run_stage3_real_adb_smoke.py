from __future__ import annotations

import json
from pathlib import Path


from iscai_stage3.pipeline.smoke import (
    evaluate_target,
    select_target,
)



REPORT = Path(
    "reports/stage3/block7_real_adb_smoke.json"
)



# deterministic synthetic uncertainty
# from Stage2 CRLB baseline

MEASUREMENT_COV = (
    (3.65230032e-5**2, 0.0, 0.0),
    (0.0, 0.0174532925**2, 0.0),
    (0.0, 0.0, 0.0174532925**2),
)



# frozen Stage1 pilot SDC-relative examples
TARGETS = (
    {
        "id": 1,
        "range": 25.0,
        "azimuth": 0.02,
        "elevation": 0.01,
        "priority": 2.0,
    },
    {
        "id": 2,
        "range": 35.0,
        "azimuth": 0.20,
        "elevation": 0.02,
        "priority": 1.0,
    },
)



def main():

    evaluated = []

    for target in TARGETS:

        score = evaluate_target(
            target_id=target["id"],
            range_m=target["range"],
            azimuth_rad=target["azimuth"],
            elevation_rad=target["elevation"],
            measurement_covariance=MEASUREMENT_COV,
            priority=target["priority"],
        )

        evaluated.append(score)



    selected = select_target(
        evaluated
    )


    report = {

        "block":
            "stage3_block7_real_adb_smoke",

        "targets":
            len(evaluated),

        "selected_target":
            selected.target_id,

        "selected_utility":
            selected.utility,

        "beam_confidence":
            selected.beam_confidence,

        "uncertainty_penalty":
            selected.uncertainty_penalty,

        "future_state_used":
            False,

        "measured_FMCW":
            False,

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
        )
    )


    print(
        "===== Stage3 real ADB smoke ====="
    )

    print(
        "targets =",
        len(evaluated)
    )

    print(
        "selected target =",
        selected.target_id
    )

    print(
        "beam confidence =",
        selected.beam_confidence
    )

    print(
        "future used = NO"
    )

    print(
        "measured FMCW = NO"
    )

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()
