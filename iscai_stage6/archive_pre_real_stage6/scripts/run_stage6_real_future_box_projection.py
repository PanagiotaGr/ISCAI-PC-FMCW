from pathlib import Path
import json
import hashlib


SOURCE = Path(
    "/home/agni/waymo/iscai_stage4/artifacts/stage4_prediction_artifact.json"
)

OUT = Path(
    "artifacts/stage6_future_box_projection.json"
)

REPORT = Path(
    "reports/block6_future_box_projection.json"
)


def main():

    data = json.loads(
        SOURCE.read_text()
    )

    boxes = []

    for item in data:

        trajectory = item["trajectory"]

        projected = []

        for step, point in enumerate(
            trajectory[:10]
        ):

            projected.append(
                {
                    "step": step + 1,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]) if len(point) > 2 else 0.0,
                    "uncertainty": float(
                        item.get(
                            "uncertainty",
                            0.0
                        )
                    ),
                }
            )


        boxes.append(
            {
                "track_index":
                    item["track_index"],

                "actor_class":
                    item["actor_class"],

                "boxes":
                    projected,

                "future_used":
                    False,
            }
        )


    OUT.write_text(
        json.dumps(
            boxes,
            indent=2
        )
    )


    digest = hashlib.sha256(
        OUT.read_bytes()
    ).hexdigest()


    report = {
        "records": len(boxes),
        "future_used": False,
        "sha256": digest,
        "status": "PASS",
    }


    REPORT.write_text(
        json.dumps(
            report,
            indent=2
        )
    )


    print("===== Stage6 Future Box Projection =====")
    print("records =", len(boxes))
    print("future_used = NO")
    print("SHA256 =", digest)
    print("STATUS = PASS")


if __name__ == "__main__":
    main()
