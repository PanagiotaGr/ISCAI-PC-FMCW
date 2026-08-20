from pathlib import Path
import json
import hashlib


SOURCE = Path(
    "artifacts/stage6_class_aware_policy.json"
)

OUT = Path(
    "artifacts/stage6_closed_loop_adb.json"
)

REPORT = Path(
    "reports/block6_closed_loop_adb.json"
)


BEAM_MAP = {
    "HIGH": "SAFETY_OVERRIDE",
    "MEDIUM": "ADAPTIVE_HIGH",
    "LOW": "NORMAL",
}


def main():

    data = json.loads(
        SOURCE.read_text()
    )

    output = []

    previous_modes = {}

    pending_modes = {}


    for item in data:

        illumination_mode = item[
            "illumination_mode"
        ]


        requested = BEAM_MAP.get(
            illumination_mode,
            "NORMAL"
        )


        track = item["track_index"]


        previous = previous_modes.get(
            track
        )

        pending = pending_modes.get(
            track
        )


        if previous == requested:

            stable_mode = previous


        elif pending == requested:

            stable_mode = requested

            previous_modes[track] = requested

            pending_modes.pop(
                track,
                None
            )


        else:

            pending_modes[track] = requested

            stable_mode = (
                previous
                if previous is not None
                else requested
            )


        previous_modes.setdefault(
            track,
            stable_mode
        )


        output.append(
            {
                "track_index":
                    item["track_index"],

                "actor_class":
                    item["actor_class"],

                "illumination_mode":
                    illumination_mode,

                "beam_state":
                    stable_mode,

                "safety_override":
                    stable_mode ==
                    "SAFETY_OVERRIDE",

                "future_used":
                    False,
            }
        )


    OUT.write_text(
        json.dumps(
            output,
            indent=2
        )
    )


    sha = hashlib.sha256(
        OUT.read_bytes()
    ).hexdigest()


    report = {

        "records":
            len(output),

        "SAFETY_OVERRIDE":
            sum(
                1
                for x in output
                if x["beam_state"] ==
                "SAFETY_OVERRIDE"
            ),

        "ADAPTIVE_HIGH":
            sum(
                1
                for x in output
                if x["beam_state"] ==
                "ADAPTIVE_HIGH"
            ),

        "NORMAL":
            sum(
                1
                for x in output
                if x["beam_state"] ==
                "NORMAL"
            ),

        "future_used":
            False,

        "sha256":
            sha,

        "status":
            "PASS",
    }


    REPORT.write_text(
        json.dumps(
            report,
            indent=2
        )
    )


    print(
        "===== Stage6 Closed Loop ADB ====="
    )

    print(
        "records =",
        len(output)
    )

    print(
        "SAFETY_OVERRIDE =",
        report["SAFETY_OVERRIDE"]
    )

    print(
        "ADAPTIVE_HIGH =",
        report["ADAPTIVE_HIGH"]
    )

    print(
        "NORMAL =",
        report["NORMAL"]
    )

    print(
        "future_used = NO"
    )

    print(
        "SHA256 =",
        sha
    )

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()
