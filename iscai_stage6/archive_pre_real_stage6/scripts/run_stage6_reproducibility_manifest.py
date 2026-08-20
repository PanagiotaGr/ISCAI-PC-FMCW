from pathlib import Path
import json
import hashlib


FILES = [

"artifacts/stage6_future_box_projection.json",
"artifacts/stage6_class_aware_policy.json",
"artifacts/stage6_closed_loop_adb.json",

"reports/block6_future_box_projection.json",
"reports/block6_illumination_mask.json",
"reports/block6_class_aware_policy.json",
"reports/block6_closed_loop_adb.json",
"reports/block6_final_audit.json",
]


OUT = Path(
"reports/stage6_reproducibility_manifest.json"
)


def sha(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()



def main():

    artifacts = []

    all_pass = True
    all_future_unused = True


    for f in FILES:

        p = Path(f)

        if not p.exists():
            raise RuntimeError(
                f"Missing {f}"
            )


        x = json.loads(
            p.read_text()
        )


        if isinstance(x, dict):

            status = x.get(
                "status",
                "ARTIFACT"
            )

            future = x.get(
                "future_used",
                False
            )

        elif isinstance(x, list):

            status = "ARTIFACT"

            future = any(
                item.get(
                    "future_used",
                    False
                )
                for item in x
                if isinstance(item, dict)
            )

        else:

            raise RuntimeError(
                f"Unsupported JSON type in {f}"
            )


        if status not in [
            "PASS",
            "ARTIFACT"
        ]:
            all_pass = False


        if future:
            all_future_unused = False


        artifacts.append(
            {
                "file": f,
                "status": status,
                "future_used": future,
                "sha256": sha(p),
            }
        )


    payload = {

        "block":
            "stage6_reproducibility_manifest",

        "artifact_count":
            len(artifacts),

        "all_status_pass":
            all_pass,

        "all_future_unused":
            all_future_unused,

        "future_used":
            False,

        "artifacts":
            artifacts,

        "status":
            "PASS"
            if all_pass and all_future_unused
            else "FAIL",
    }


    payload["sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True
        ).encode()
    ).hexdigest()


    OUT.write_text(
        json.dumps(
            payload,
            indent=2
        )
    )


    print(
        "===== Stage6 Reproducibility Manifest ====="
    )

    print(
        "artifacts =",
        len(artifacts)
    )

    print(
        "all_status_pass =",
        all_pass
    )

    print(
        "all_future_unused =",
        all_future_unused
    )

    print(
        "SHA256 =",
        payload["sha256"]
    )

    print(
        "STATUS =",
        payload["status"]
    )


if __name__ == "__main__":
    main()
