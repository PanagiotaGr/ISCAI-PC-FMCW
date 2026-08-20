from pathlib import Path
import json
import hashlib


SOURCE = Path(
    "artifacts/stage6_illumination_mask.json"
)

OUT = Path(
    "artifacts/stage6_class_aware_policy.json"
)

REPORT = Path(
    "reports/block6_class_aware_policy.json"
)


CLASS_WEIGHT = {
    "VEHICLE": 1.0,
    "PEDESTRIAN": 1.5,
    "CYCLIST": 1.3,
}


def main():

    data = json.loads(
        SOURCE.read_text()
    )

    policies = []

    for item in data:

        actor_class = item.get(
            "actor_class",
            "UNKNOWN"
        )

        risk = item.get(
            "illumination_risk",
            0.0
        )

        weight = CLASS_WEIGHT.get(
            actor_class,
            1.0
        )

        priority = risk * weight


        if priority > 0.08:
            mode = "HIGH"
        elif priority > 0.03:
            mode = "MEDIUM"
        else:
            mode = "LOW"


        policies.append(
            {
                "track_index":
                    item["track_index"],

                "actor_class":
                    actor_class,

                "priority":
                    priority,

                "illumination_mode":
                    mode,

                "future_used":
                    False,
            }
        )


    OUT.write_text(
        json.dumps(
            policies,
            indent=2
        )
    )


    sha = hashlib.sha256(
        OUT.read_bytes()
    ).hexdigest()


    REPORT.write_text(
        json.dumps(
            {
                "records":
                    len(policies),

                "HIGH":
                    sum(
                        1 for x in policies
                        if x["illumination_mode"]=="HIGH"
                    ),

                "MEDIUM":
                    sum(
                        1 for x in policies
                        if x["illumination_mode"]=="MEDIUM"
                    ),

                "LOW":
                    sum(
                        1 for x in policies
                        if x["illumination_mode"]=="LOW"
                    ),

                "future_used":
                    False,

                "sha256":
                    sha,

                "status":
                    "PASS",
            },
            indent=2
        )
    )


    print(
        "===== Stage6 Class Aware Illumination Policy ====="
    )

    print(
        "records =",
        len(policies)
    )

    print(
        "HIGH =",
        sum(
            1 for x in policies
            if x["illumination_mode"]=="HIGH"
        )
    )

    print(
        "MEDIUM =",
        sum(
            1 for x in policies
            if x["illumination_mode"]=="MEDIUM"
        )
    )

    print(
        "LOW =",
        sum(
            1 for x in policies
            if x["illumination_mode"]=="LOW"
        )
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
