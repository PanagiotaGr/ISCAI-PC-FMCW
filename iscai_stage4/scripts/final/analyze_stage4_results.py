import json
from pathlib import Path


BASE = Path(
    "reports/stage4_auto_cuda"
)

OUT = Path(
    "reports/stage4_final/stage4_summary.json"
)


def load_results():

    results = {}

    for name in [
        "all_results.json",
        "worker_0_results.json",
        "worker_1_results.json",
    ]:

        p = BASE / name

        if not p.exists():
            continue

        data = json.loads(
            p.read_text()
        )

        for r in data:
            results[r["name"]] = r

    return list(results.values())


def main():

    results = load_results()

    print("==============================")
    print("STAGE 4 FINAL ANALYSIS")
    print("==============================")

    print(
        "Unique experiments:",
        len(results)
    )


    deterministic = [
        r for r in results
        if r["config"]["model"]
        ==
        "deterministic"
    ]


    gaussian = [
        r for r in results
        if r["config"]["model"]
        ==
        "gaussian"
    ]


    print()
    print(
        "Deterministic:",
        len(deterministic)
    )

    print(
        "Gaussian:",
        len(gaussian)
    )


    if deterministic:

        best_det = min(
            deterministic,
            key=lambda r:
            r["metrics"]["loss"]
        )

        print()
        print("BEST DETERMINISTIC")
        print(
            best_det["name"]
        )
        print(
            best_det["metrics"]
        )


    if gaussian:

        best_gau = min(
            gaussian,
            key=lambda r:
            r["metrics"]["loss"]
        )

        print()
        print("BEST GAUSSIAN")
        print(
            best_gau["name"]
        )
        print(
            best_gau["metrics"]
        )


    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    OUT.write_text(
        json.dumps(
            {
                "experiments": len(results),
                "deterministic": len(deterministic),
                "gaussian": len(gaussian),
            },
            indent=2
        )
    )


    print()
    print("REPORT:")
    print(OUT)


if __name__ == "__main__":
    main()
