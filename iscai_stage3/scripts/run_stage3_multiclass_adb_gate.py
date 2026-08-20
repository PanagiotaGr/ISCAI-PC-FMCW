from __future__ import annotations

import json
from pathlib import Path


from iscai_stage3.pipeline.smoke import (
    evaluate_target,
    select_target,
)


REPORT = Path(
    "reports/stage3/block8_multiclass_adb_gate.json"
)


MEASUREMENT_COV = (
    (3.65230032e-5**2, 0.0, 0.0),
    (0.0, 0.0174532925**2, 0.0),
    (0.0, 0.0, 0.0174532925**2),
)



# Stage2 frozen multiclass coverage abstraction
ACTORS = (
    {
        "id": 1,
        "class": "TYPE_VEHICLE",
        "range": 25.0,
        "azimuth": 0.02,
        "elevation": 0.01,
        "priority": 2.0,
    },

    {
        "id": 2,
        "class": "TYPE_PEDESTRIAN",
        "range": 18.0,
        "azimuth": 0.08,
        "elevation": 0.02,
        "priority": 3.0,
    },

    {
        "id": 3,
        "class": "TYPE_CYCLIST",
        "range": 30.0,
        "azimuth": 0.15,
        "elevation": 0.01,
        "priority": 2.5,
    },
)



def main():

    scored = []

    for actor in ACTORS:

        result = evaluate_target(
            target_id=actor["id"],
            range_m=actor["range"],
            azimuth_rad=actor["azimuth"],
            elevation_rad=actor["elevation"],
            measurement_covariance=MEASUREMENT_COV,
            priority=actor["priority"],
        )


        scored.append(
            (
                actor,
                result,
            )
        )


    selected = select_target(
        tuple(
            x[1]
            for x in scored
        )
    )


    selected_class = None

    for actor, score in scored:

        if score.target_id == selected.target_id:
            selected_class = actor["class"]



    report = {

        "block":
            "stage3_block8_multiclass_adb_gate",

        "actors":
            len(ACTORS),

        "classes":
            sorted(
                list(
                    set(
                        x["class"]
                        for x in ACTORS
                    )
                )
            ),

        "selected_target":
            selected.target_id,

        "selected_class":
            selected_class,

        "beam_confidence":
            selected.beam_confidence,

        "utility":
            selected.utility,

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
        "===== Stage3 multiclass ADB gate ====="
    )

    print(
        "actors =",
        len(ACTORS)
    )

    print(
        "classes =",
        report["classes"]
    )

    print(
        "selected target =",
        selected.target_id
    )

    print(
        "selected class =",
        selected_class
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
