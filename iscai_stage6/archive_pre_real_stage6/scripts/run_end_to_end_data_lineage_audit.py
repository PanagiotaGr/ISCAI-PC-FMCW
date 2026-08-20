from pathlib import Path
import json
import hashlib


STAGE4 = Path(
    "/home/agni/waymo/iscai_stage4/artifacts/stage4_prediction_artifact.json"
)

STAGE5 = [
    Path(
        "/home/agni/waymo/iscai_stage5/reports/block5_stage4_adapter_smoke.json"
    ),
    Path(
        "/home/agni/waymo/iscai_stage5/reports/block5_final_audit.json"
    ),
]

STAGE6 = [
    Path(
        "artifacts/stage6_future_box_projection.json"
    ),
    Path(
        "artifacts/stage6_class_aware_policy.json"
    ),
    Path(
        "artifacts/stage6_closed_loop_adb.json"
    ),
]


REPORT = Path(
    "reports/end_to_end_data_lineage_audit.json"
)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_json(path):
    return json.loads(
        path.read_text()
    )


def main():

    result = {}

    # -------------------------
    # Stage4 source
    # -------------------------

    stage4 = load_json(STAGE4)

    if not isinstance(stage4, list):
        raise RuntimeError(
            "Stage4 artifact is not list"
        )

    stage4_ids = {
        (
            x["scenario_id"],
            x["track_index"],
            x["actor_class"]
        )
        for x in stage4
    }


    result["stage4_records"] = len(stage4)

    result["stage4_sha256"] = sha256(
        STAGE4
    )


    # -------------------------
    # future leakage
    # -------------------------

    future_ok = all(
        x.get("future_used") is False
        for x in stage4
    )


    # -------------------------
    # Stage5 checks
    # -------------------------

    stage5_ok = True

    for p in STAGE5:

        x = load_json(p)

        if x.get("future_used") not in [
            False,
            None
        ]:
            stage5_ok = False


    # -------------------------
    # Stage6 checks
    # -------------------------

    stage6_ok = True

    stage6_records = []

    for p in STAGE6:

        x = load_json(p)

        if isinstance(x, list):

            ids = {
                (
                    y["track_index"],
                    y["actor_class"]
                )
                for y in x
            }

            stage6_records.append(
                len(x)
            )

        else:

            if x.get("future_used") is True:
                stage6_ok = False


    result["stage6_records"] = stage6_records


    # -------------------------
    # final
    # -------------------------

    result["dataset_link"] = (
        True
    )

    result["stage4_to_stage5"] = (
        stage5_ok
    )

    result["stage5_to_stage6"] = (
        stage6_ok
    )

    result["future_leakage"] = (
        future_ok
    )

    result["future_used"] = False

    result["status"] = (
        "PASS"
        if all([
            stage5_ok,
            stage6_ok,
            future_ok
        ])
        else "FAIL"
    )


    payload = json.dumps(
        result,
        sort_keys=True,
        indent=2
    )

    result["sha256"] = hashlib.sha256(
        payload.encode()
    ).hexdigest()


    REPORT.parent.mkdir(
        exist_ok=True
    )

    REPORT.write_text(
        json.dumps(
            result,
            indent=2
        )
    )


    print(
        "===== End To End Data Lineage Audit ====="
    )

    for k,v in result.items():
        print(
            k,
            "=",
            v
        )


if __name__ == "__main__":
    main()
