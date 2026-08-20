import json
from pathlib import Path
from datetime import datetime, timezone


BASE = Path("reports/stage4_auto_cuda")

CALIBRATION = BASE / "gaussian_calibration_final_real.json"
GMM = BASE / "gmm_real_training_result.json"

OUTPUT = BASE / "stage4_final_real_results.json"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required real-results file missing: {path}"
        )

    return json.loads(path.read_text())


cal = load_json(CALIBRATION)
gmm = load_json(GMM)


# ------------------------------------------------------------
# Gaussian results:
# use ONLY values already produced by the real calibration /
# validation evaluation.
# ------------------------------------------------------------

gaussian_results = cal.get("results")

if gaussian_results is None:
    # Allow final calibration files that store the comparison
    # under another top-level key.
    gaussian_results = cal.get("models")

if gaussian_results is None:
    raise RuntimeError(
        "Could not find Gaussian model results "
        "inside the real calibration report."
    )


# ------------------------------------------------------------
# GMM result:
# real WOMD learned model only.
# ------------------------------------------------------------

if gmm.get("data_source") != "real_WOMD":
    raise RuntimeError(
        "GMM report is not marked as real_WOMD."
    )

if gmm.get("future_used_as_input") is not False:
    raise RuntimeError(
        "GMM report does not confirm causal evaluation."
    )


report = {
    "stage": 4,

    "status": "COMPLETE",

    "result_policy": (
        "Only real WOMD trained/validation results. "
        "No heuristic or placeholder metrics."
    ),

    "causal_prediction": True,

    "gaussian": {
        "source":
            str(CALIBRATION),

        "results":
            gaussian_results,

        "selection":
            cal.get("selection"),
    },

    "gmm": {
        "source":
            str(GMM),

        "name":
            gmm["name"],

        "data_source":
            gmm["data_source"],

        "modes":
            gmm["modes"],

        "best_epoch":
            gmm["best_epoch"],

        "epochs_run":
            gmm["epochs_run"],

        "metrics":
            gmm["metrics"],

        "checkpoint":
            gmm["checkpoint"],

        "future_used_as_input":
            gmm["future_used_as_input"],
    },

    "generated_utc":
        datetime.now(timezone.utc).isoformat(),
}


OUTPUT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


print("=" * 70)
print("FINAL STAGE 4 REAL RESULTS")
print("=" * 70)

print()
print("GMM")
print("Name      =", gmm["name"])
print("Modes     =", gmm["modes"])
print("Best epoch=", gmm["best_epoch"])
print("NLL/loss  =", gmm["metrics"]["loss"])
print("minADE    =", gmm["metrics"]["minADE"])
print("minFDE    =", gmm["metrics"]["minFDE"])

print()
print("Gaussian calibration source:")
print(CALIBRATION)

print()
print("Saved:")
print(OUTPUT)
