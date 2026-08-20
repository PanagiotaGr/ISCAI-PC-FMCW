from pathlib import Path
import json


INPUT = Path(
    "reports/stage4_auto_cuda/"
    "gaussian_calibration_comparison.json"
)

OUTPUT = Path(
    "reports/stage4_auto_cuda/"
    "gaussian_calibration_final_real.json"
)


if not INPUT.exists():
    raise FileNotFoundError(INPUT)

report = json.loads(INPUT.read_text())

levels = report["confidence_levels"]

final_results = []

for result in report["results"]:

    coverage = result["coverage"]

    reliability = []
    brier_by_level = {}

    abs_errors = []

    for p in levels:

        key = str(int(round(p * 100)))

        q = float(coverage[key])

        # Binary calibration event:
        #
        # prediction: confidence region contains target
        # forecast probability: p
        # observed empirical frequency: q
        #
        # Mean Brier:
        # E[(p-I)^2]
        # = q(1-p)^2 + (1-q)p^2

        brier = (
            q * (1.0 - p) ** 2
            +
            (1.0 - q) * p ** 2
        )

        error = q - p

        brier_by_level[key] = brier

        reliability.append(
            {
                "nominal_probability": p,
                "empirical_coverage": q,
                "calibration_error": error,
                "absolute_error": abs(error),
            }
        )

        abs_errors.append(abs(error))

    ece = sum(abs_errors) / len(abs_errors)

    mean_brier = (
        sum(brier_by_level.values())
        /
        len(brier_by_level)
    )

    final_results.append(
        {
            "label": result["label"],
            "name": result["name"],
            "checkpoint": result["checkpoint"],

            "real_validation_samples":
                result["samples"],

            "real_trajectory_points":
                result["trajectory_points"],

            "NLL": result["NLL"],
            "ADE": result["ADE"],
            "FDE": result["FDE"],

            "coverage": coverage,

            # Calibration ECE across the five
            # confidence-region reliability points.
            "ECE_coverage_levels": ece,

            # Brier score for the binary
            # confidence-region containment events.
            "Brier_by_confidence_level":
                brier_by_level,

            "mean_Brier":
                mean_brier,

            "reliability_diagram_points":
                reliability,

            "mean_std_x":
                result["mean_std_x"],

            "mean_std_y":
                result["mean_std_y"],

            "mean_abs_rho":
                result["mean_abs_rho"],
        }
    )


best_nll = min(
    final_results,
    key=lambda x: x["NLL"],
)

best_calibrated = min(
    final_results,
    key=lambda x: x["ECE_coverage_levels"],
)

best_brier = min(
    final_results,
    key=lambda x: x["mean_Brier"],
)


final = {
    "block":
        "stage4_real_trained_gaussian_calibration",

    "data_source":
        "real_WOMD_validation",

    "prediction_source":
        "trained_Gaussian_GRU_checkpoints",

    "future_used_as_input":
        False,

    "confidence_levels":
        levels,

    "brier_definition":
        (
            "binary confidence-region containment "
            "Brier score at each nominal probability"
        ),

    "ece_definition":
        (
            "mean absolute difference between "
            "nominal probability and empirical "
            "confidence-region coverage"
        ),

    "results":
        final_results,

    "selection":
        {
            "best_NLL":
                best_nll["name"],

            "best_calibrated":
                best_calibrated["name"],

            "best_Brier":
                best_brier["name"],
        },

    "status":
        "PASS",
}


OUTPUT.write_text(
    json.dumps(
        final,
        indent=2,
    )
)

print("=" * 70)
print("FINAL REAL STAGE 4 CALIBRATION")
print("=" * 70)

for r in final_results:

    print()
    print(r["name"])

    print(
        "NLL   =",
        r["NLL"],
    )

    print(
        "ADE   =",
        r["ADE"],
    )

    print(
        "FDE   =",
        r["FDE"],
    )

    print(
        "ECE   =",
        r["ECE_coverage_levels"],
    )

    print(
        "Brier =",
        r["mean_Brier"],
    )

    print("Reliability:")

    for point in r[
        "reliability_diagram_points"
    ]:
        print(
            f"  nominal "
            f"{100*point['nominal_probability']:.1f}%"
            f" -> empirical "
            f"{100*point['empirical_coverage']:.2f}%"
        )


print()
print("=" * 70)
print("SELECTION")
print("=" * 70)

print(
    "BEST NLL:",
    final["selection"]["best_NLL"],
)

print(
    "BEST CALIBRATED:",
    final["selection"]["best_calibrated"],
)

print(
    "BEST BRIER:",
    final["selection"]["best_Brier"],
)

print()
print("Saved:", OUTPUT)
