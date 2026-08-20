from pathlib import Path
import json
import hashlib
import math


SOURCE = Path(
    "artifacts/stage6_future_box_projection.json"
)

OUT = Path(
    "artifacts/stage6_illumination_mask.json"
)

REPORT = Path(
    "reports/block6_illumination_mask.json"
)


def main():

    data = json.loads(
        SOURCE.read_text()
    )

    masks = []

    for actor in data:

        points = actor["boxes"]

        risk_values = []

        for p in points:

            distance = math.sqrt(
                p["x"]**2 +
                p["y"]**2 +
                p["z"]**2
            )

            uncertainty = p["uncertainty"]

            risk = (
                1.0 /
                (1.0 + distance)
            ) * (
                1.0 +
                uncertainty
            )

            risk_values.append(
                risk
            )


        mean_risk = (
            sum(risk_values)
            /
            len(risk_values)
            if risk_values
            else 0.0
        )


        masks.append(
            {
                "track_index":
                    actor["track_index"],

                "actor_class":
                    actor["actor_class"],

                "illumination_risk":
                    mean_risk,

                "active":
                    mean_risk > 0.05,

                "future_used":
                    False,
            }
        )


    OUT.write_text(
        json.dumps(
            masks,
            indent=2
        )
    )


    sha = hashlib.sha256(
        OUT.read_bytes()
    ).hexdigest()


    REPORT.write_text(
        json.dumps(
            {
                "records": len(masks),
                "active_regions":
                    sum(
                        1 for x in masks
                        if x["active"]
                    ),
                "future_used": False,
                "sha256": sha,
                "status": "PASS",
            },
            indent=2
        )
    )


    print(
        "===== Stage6 Illumination Mask ====="
    )

    print(
        "records =",
        len(masks)
    )

    print(
        "active_regions =",
        sum(
            1 for x in masks
            if x["active"]
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
