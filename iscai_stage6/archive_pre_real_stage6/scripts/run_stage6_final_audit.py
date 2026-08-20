from pathlib import Path
import json
import hashlib


REPORTS = {

    "projection":
        Path(
            "reports/block6_future_box_projection.json"
        ),

    "illumination":
        Path(
            "reports/block6_illumination_mask.json"
        ),

    "policy":
        Path(
            "reports/block6_class_aware_policy.json"
        ),

    "adb":
        Path(
            "reports/block6_closed_loop_adb.json"
        ),
}


OUT = Path(
    "reports/block6_final_audit.json"
)


def load(path):

    if not path.exists():
        raise RuntimeError(
            f"Missing report {path}"
        )

    return json.loads(
        path.read_text()
    )


def main():

    reports = {}

    for name, path in REPORTS.items():

        reports[name] = load(path)


    for name, report in reports.items():

        if report.get(
            "future_used",
            False
        ):

            raise RuntimeError(
                f"Future leakage in {name}"
            )


        if report.get(
            "status"
        ) != "PASS":

            raise RuntimeError(
                f"Failed block {name}"
            )


    records = {
        x.get("records")
        for x in reports.values()
    }


    if len(records) != 1:

        raise RuntimeError(
            "Record mismatch"
        )


    record_count = next(
        iter(records)
    )


    if record_count <= 0:

        raise RuntimeError(
            "No records"
        )


    payload = {

        "components":
            len(reports),

        "records":
            record_count,

        "blocks":
            {
                k:
                "PASS"
                for k in reports
            },

        "future_used":
            False,

        "status":
            "PASS",
    }


    sha = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True
        ).encode()
    ).hexdigest()


    payload["sha256"] = sha


    OUT.write_text(
        json.dumps(
            payload,
            indent=2
        )
    )


    print(
        "===== Stage6 Final Audit ====="
    )

    print(
        "components =",
        payload["components"]
    )

    print(
        "records =",
        record_count
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

    print(
        "report =",
        OUT
    )


if __name__ == "__main__":
    main()
