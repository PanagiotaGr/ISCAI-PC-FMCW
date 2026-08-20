import json
from collections import defaultdict
from pathlib import Path


REPORT_DIR = Path("reports/stage4_auto_cuda")

RESULTS_FILE = REPORT_DIR / "all_results.json"

OUTPUT_FILE = REPORT_DIR / "stage4_selected_models.json"


def load_results():

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing results file: {RESULTS_FILE}"
        )

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise RuntimeError(
            "all_results.json must contain a list"
        )

    return results


def valid_result(r):

    return (
        isinstance(r, dict)
        and "name" in r
        and "config" in r
        and "metrics" in r
        and "loss" in r["metrics"]
        and "ADE" in r["metrics"]
        and "FDE" in r["metrics"]
    )


def main():

    results = [
        r
        for r in load_results()
        if valid_result(r)
    ]

    print("TOTAL VALID RESULTS:", len(results))

    if not results:
        raise RuntimeError(
            "No valid Stage-4 results found."
        )

    grouped = defaultdict(list)

    for r in results:

        model_type = r["config"]["model"]

        grouped[model_type].append(r)


    selected = {}


    for model_type, runs in grouped.items():

        by_loss = min(
            runs,
            key=lambda r: r["metrics"]["loss"]
        )

        by_ade = min(
            runs,
            key=lambda r: r["metrics"]["ADE"]
        )

        by_fde = min(
            runs,
            key=lambda r: r["metrics"]["FDE"]
        )

        selected[model_type] = {
            "best_validation_loss": by_loss,
            "best_ADE": by_ade,
            "best_FDE": by_fde,
        }


    # Global best runs

    selected["global"] = {

        "best_validation_loss": min(
            results,
            key=lambda r: r["metrics"]["loss"]
        ),

        "best_ADE": min(
            results,
            key=lambda r: r["metrics"]["ADE"]
        ),

        "best_FDE": min(
            results,
            key=lambda r: r["metrics"]["FDE"]
        ),
    }


    with open(
        OUTPUT_FILE,
        "w"
    ) as f:

        json.dump(
            selected,
            f,
            indent=2
        )


    print("\n==============================")
    print("STAGE 4 MODEL SELECTION")
    print("==============================")

    for model_type in [
        "deterministic",
        "gaussian"
    ]:

        if model_type not in selected:
            continue

        print(
            f"\nMODEL: {model_type}"
        )

        for criterion, result in (
            selected[model_type].items()
        ):

            m = result["metrics"]

            print(
                f"{criterion:22s} "
                f"{result['name']}"
            )

            print(
                f"    loss={m['loss']:.6f} "
                f"ADE={m['ADE']:.6f} "
                f"FDE={m['FDE']:.6f}"
            )


    print(
        "\nSaved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
