from pathlib import Path
import json
import numpy as np


INPUT = Path(
    "stage7_real/data/"
    "stage7_joint_actor_metrics.npz"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_joint_confidence_intervals.json"
)

N_BOOT = 10000
SEED = 42
CI_LOW = 2.5
CI_HIGH = 97.5


METRICS = {
    "beam_mean_selected_k":
        "beam_mean_selected_k",

    "beam_mean_covered_mass":
        "beam_mean_covered_mass",

    "beam_mean_overhead_fraction":
        "beam_mean_overhead_fraction",

    "adb_mean_iou":
        "adb_mean_iou",

    "adb_mean_shadow_violation":
        "adb_mean_shadow_violation",

    "adb_mean_overmask":
        "adb_mean_overmask",

    "adb_mean_road_retention":
        "adb_mean_road_retention",

    "adb_mean_oracle_region_visibility":
        "adb_mean_oracle_region_visibility",
}


def bootstrap_mean_ci(
    x,
    rng,
):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    n = len(x)

    if n < 2:
        raise RuntimeError(
            "Need at least two actors "
            "for bootstrap CI"
        )

    if not np.all(
        np.isfinite(x)
    ):
        raise RuntimeError(
            "Non-finite metric values"
        )

    boot = np.empty(
        N_BOOT,
        dtype=np.float64,
    )

    for i in range(N_BOOT):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = float(
            np.mean(
                x[idx]
            )
        )

    return {
        "n":
            int(n),

        "estimate":
            float(
                np.mean(x)
            ),

        "bootstrap_mean":
            float(
                np.mean(boot)
            ),

        "bootstrap_std":
            float(
                np.std(
                    boot,
                    ddof=1,
                )
            ),

        "ci95_low":
            float(
                np.percentile(
                    boot,
                    CI_LOW,
                )
            ),

        "ci95_high":
            float(
                np.percentile(
                    boot,
                    CI_HIGH,
                )
            ),
    }


def main():

    print("=" * 88)
    print("STAGE 7 JOINT 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 88)

    d = np.load(INPUT)

    actor_class = d[
        "actor_class"
    ].astype(str)

    rng = np.random.default_rng(
        SEED
    )

    overall = {}

    for label, field in METRICS.items():

        overall[label] = (
            bootstrap_mean_ci(
                d[field],
                rng,
            )
        )

    by_class = {}

    for cls in sorted(
        np.unique(
            actor_class
        )
    ):

        mask = (
            actor_class
            ==
            cls
        )

        class_results = {}

        for label, field in METRICS.items():

            class_results[
                label
            ] = bootstrap_mean_ci(
                d[field][mask],
                rng,
            )

        by_class[
            str(cls)
        ] = class_results

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "analysis":
            "actor_level_bootstrap_confidence_intervals",

        "source":
            str(INPUT),

        "statistical_unit":
            "actor",

        "actors":
            int(
                len(actor_class)
            ),

        "bootstrap_replicates":
            N_BOOT,

        "seed":
            SEED,

        "confidence_level":
            0.95,

        "interval_method":
            "nonparametric_percentile_bootstrap",

        "overall":
            overall,

        "by_class":
            by_class,

        "scientific_scope": (
            "Confidence intervals are estimated "
            "by nonparametric actor-level bootstrap. "
            "All ten future horizons belonging to an "
            "actor are first aggregated into the actor "
            "metric, preventing individual future steps "
            "from being treated as independent samples."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("===== OVERALL =====")

    for metric, x in overall.items():

        print(
            f"{metric:38s} "
            f"mean={x['estimate']:.8f} "
            f"CI95=["
            f"{x['ci95_low']:.8f}, "
            f"{x['ci95_high']:.8f}]"
        )

    print()
    print("===== BY CLASS =====")

    for cls, metrics in by_class.items():

        print()
        print(cls)

        for metric in (
            "beam_mean_covered_mass",
            "beam_mean_overhead_fraction",
            "adb_mean_iou",
            "adb_mean_shadow_violation",
            "adb_mean_overmask",
            "adb_mean_oracle_region_visibility",
        ):

            x = metrics[
                metric
            ]

            print(
                f"  {metric:36s} "
                f"{x['estimate']:.8f} "
                f"[{x['ci95_low']:.8f}, "
                f"{x['ci95_high']:.8f}]"
            )

    print()
    print("Saved:", OUTPUT)

    print()
    print(
        "STAGE 7 CONFIDENCE INTERVALS = PASS"
    )


if __name__ == "__main__":
    main()
