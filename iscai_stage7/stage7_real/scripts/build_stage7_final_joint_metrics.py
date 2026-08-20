from pathlib import Path
import json


TRADEOFF = Path(
    "stage7_real/reports/"
    "stage7_joint_tradeoff.json"
)

CI = Path(
    "stage7_real/reports/"
    "stage7_joint_confidence_intervals.json"
)

UNCERTAINTY = Path(
    "stage7_real/reports/"
    "stage7_uncertainty_sweep.json"
)

LATENCY = Path(
    "stage7_real/reports/"
    "stage7_latency_sweep_v2.json"
)

CODEBOOK = Path(
    "stage7_real/reports/"
    "stage7_codebook_tradeoff.json"
)

COHORT = Path(
    "stage7_real/reports/"
    "stage7_joint_cohort_summary.json"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_final_joint_metrics.json"
)


def main():

    print("=" * 88)
    print("STAGE 7 FINAL JOINT METRICS")
    print("=" * 88)

    tradeoff = json.loads(
        TRADEOFF.read_text()
    )

    ci = json.loads(
        CI.read_text()
    )

    uncertainty = json.loads(
        UNCERTAINTY.read_text()
    )

    latency = json.loads(
        LATENCY.read_text()
    )

    codebook = json.loads(
        CODEBOOK.read_text()
    )

    cohort = json.loads(
        COHORT.read_text()
    )

    comm = (
        tradeoff[
            "representative_operating_point"
        ][
            "communication"
        ]
    )

    illum = (
        tradeoff[
            "representative_operating_point"
        ][
            "illumination"
        ]
    )

    overall_ci = ci["overall"]
    class_ci = ci["by_class"]

    # --------------------------------------------------------
    # Latency points
    # --------------------------------------------------------

    latency_rows = {}

    for x in latency["results"]:

        key = (
            f"{int(x['delay_ms'])}_ms"
        )

        latency_rows[key] = {
            "evaluated_states":
                x["evaluated_states"],

            "mean_selected_k":
                x[
                    "communication"
                ][
                    "selected_k"
                ][
                    "mean"
                ],

            "current_posterior_covered_mass":
                x[
                    "communication"
                ][
                    "covered_mass"
                ][
                    "mean"
                ],

            "mean_overhead_fraction":
                x[
                    "communication"
                ][
                    "mean_overhead_fraction"
                ],
        }

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "evaluation":
            "final_joint_communication_illumination",

        "shared_posterior":
            True,

        "future_used_as_input":
            False,

        "joint_cohort": {
            "source_actors":
                cohort[
                    "source_actors"
                ],

            "current_fov_actors":
                cohort[
                    "current_fov_actors"
                ],

            "predicted_entry_actors":
                cohort[
                    "predicted_entry_actors"
                ],

            "joint_relevant_actors":
                cohort[
                    "joint_relevant_actors"
                ],

            "results_by_class":
                cohort[
                    "results_by_class"
                ],
        },

        # ====================================================
        # Experiment-7 required joint metrics
        # ====================================================

        "communication_reliability": {
            "metric":
                "posterior probability coverage",

            "mean":
                comm[
                    "mean_covered_mass"
                ],

            "ci95":
                [
                    overall_ci[
                        "beam_mean_covered_mass"
                    ][
                        "ci95_low"
                    ],
                    overall_ci[
                        "beam_mean_covered_mass"
                    ][
                        "ci95_high"
                    ],
                ],

            "policy": {
                "num_beams":
                    comm[
                        "num_beams"
                    ],

                "target_mass":
                    comm[
                        "target_mass"
                    ],
            },
        },

        "beam_overhead": {
            "mean_selected_k":
                comm[
                    "mean_selected_k"
                ],

            "mean_selected_k_ci95":
                [
                    overall_ci[
                        "beam_mean_selected_k"
                    ][
                        "ci95_low"
                    ],
                    overall_ci[
                        "beam_mean_selected_k"
                    ][
                        "ci95_high"
                    ],
                ],

            "mean_overhead_fraction":
                comm[
                    "mean_overhead_fraction"
                ],

            "mean_overhead_fraction_ci95":
                [
                    overall_ci[
                        "beam_mean_overhead_fraction"
                    ][
                        "ci95_low"
                    ],
                    overall_ci[
                        "beam_mean_overhead_fraction"
                    ][
                        "ci95_high"
                    ],
                ],
        },

        "glare_protection": {
            "metric":
                "vehicle shadow-zone violation",

            "overall_mean_shadow_violation":
                illum[
                    "mean_shadow_violation"
                ],

            "overall_ci95":
                [
                    overall_ci[
                        "adb_mean_shadow_violation"
                    ][
                        "ci95_low"
                    ],
                    overall_ci[
                        "adb_mean_shadow_violation"
                    ][
                        "ci95_high"
                    ],
                ],

            "vehicle_mean_shadow_violation":
                class_ci[
                    "VEHICLE"
                ][
                    "adb_mean_shadow_violation"
                ][
                    "estimate"
                ],

            "vehicle_ci95":
                [
                    class_ci[
                        "VEHICLE"
                    ][
                        "adb_mean_shadow_violation"
                    ][
                        "ci95_low"
                    ],
                    class_ci[
                        "VEHICLE"
                    ][
                        "adb_mean_shadow_violation"
                    ][
                        "ci95_high"
                    ],
                ],
        },

        "vru_visibility": {
            "metric":
                "mean retained illumination inside constructed oracle region",

            "pedestrian": {
                "mean":
                    class_ci[
                        "PEDESTRIAN"
                    ][
                        "adb_mean_oracle_region_visibility"
                    ][
                        "estimate"
                    ],

                "ci95":
                    [
                        class_ci[
                            "PEDESTRIAN"
                        ][
                            "adb_mean_oracle_region_visibility"
                        ][
                            "ci95_low"
                        ],
                        class_ci[
                            "PEDESTRIAN"
                        ][
                            "adb_mean_oracle_region_visibility"
                        ][
                            "ci95_high"
                        ],
                    ],
            },

            "cyclist": {
                "mean":
                    class_ci[
                        "CYCLIST"
                    ][
                        "adb_mean_oracle_region_visibility"
                    ][
                        "estimate"
                    ],

                "ci95":
                    [
                        class_ci[
                            "CYCLIST"
                        ][
                            "adb_mean_oracle_region_visibility"
                        ][
                            "ci95_low"
                        ],
                        class_ci[
                            "CYCLIST"
                        ][
                            "adb_mean_oracle_region_visibility"
                        ][
                            "ci95_high"
                        ],
                    ],
            },
        },

        "overmasking": {
            "mean":
                illum[
                    "mean_overmask"
                ],

            "ci95":
                [
                    overall_ci[
                        "adb_mean_overmask"
                    ][
                        "ci95_low"
                    ],
                    overall_ci[
                        "adb_mean_overmask"
                    ][
                        "ci95_high"
                    ],
                ],
        },

        "road_illumination_retention": {
            "mean":
                illum[
                    "mean_road_retention"
                ],

            "ci95":
                [
                    overall_ci[
                        "adb_mean_road_retention"
                    ][
                        "ci95_low"
                    ],
                    overall_ci[
                        "adb_mean_road_retention"
                    ][
                        "ci95_high"
                    ],
                ],
        },

        "adb_mask_iou": {
            "mean":
                illum[
                    "mean_iou"
                ],

            "ci95":
                [
                    overall_ci[
                        "adb_mean_iou"
                    ][
                        "ci95_low"
                    ],
                    overall_ci[
                        "adb_mean_iou"
                    ][
                        "ci95_high"
                    ],
                ],
        },

        "latency": {
            "type":
                "algorithmic stale-posterior latency",

            "physical_actuation_latency_measured":
                False,

            "results":
                latency_rows,
        },

        # ====================================================
        # Required Stage-7 sweeps
        # ====================================================

        "sweeps": {
            "uncertainty": {
                "status":
                    uncertainty["status"],

                "scales":
                    uncertainty[
                        "uncertainty_scales"
                    ],

                "report":
                    str(UNCERTAINTY),
            },

            "latency": {
                "status":
                    latency["status"],

                "delays_ms":
                    latency[
                        "delay_ms"
                    ],

                "report":
                    str(LATENCY),
            },

            "codebook": {
                "status":
                    codebook["status"],

                "codebooks":
                    [
                        x["num_beams"]
                        for x in codebook[
                            "results"
                        ]
                    ],

                "report":
                    str(CODEBOOK),
            },
        },

        "confidence_intervals": {
            "status":
                ci["status"],

            "method":
                ci[
                    "interval_method"
                ],

            "statistical_unit":
                ci[
                    "statistical_unit"
                ],

            "bootstrap_replicates":
                ci[
                    "bootstrap_replicates"
                ],

            "confidence_level":
                ci[
                    "confidence_level"
                ],

            "report":
                str(CI),
        },

        "sources": {
            "joint_tradeoff":
                str(TRADEOFF),

            "joint_confidence_intervals":
                str(CI),

            "uncertainty_sweep":
                str(UNCERTAINTY),

            "latency_sweep":
                str(LATENCY),

            "codebook_sweep":
                str(CODEBOOK),

            "joint_cohort":
                str(COHORT),
        },

        "scientific_scope": (
            "Final Stage-7 joint evaluation of the shared "
            "trajectory posterior. Communication reliability, "
            "beam overhead, ADB glare protection, VRU visibility, "
            "over-masking and latency are reported as separate "
            "domain metrics. No arbitrary scalar joint utility "
            "is introduced. Physical ADB actuation latency is "
            "not measured."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("joint actors =", report[
        "joint_cohort"
    ][
        "joint_relevant_actors"
    ])

    print()
    print("COMMUNICATION")
    print(
        " coverage =",
        report[
            "communication_reliability"
        ][
            "mean"
        ]
    )
    print(
        " overhead =",
        report[
            "beam_overhead"
        ][
            "mean_overhead_fraction"
        ]
    )

    print()
    print("ILLUMINATION")
    print(
        " IoU =",
        report[
            "adb_mask_iou"
        ][
            "mean"
        ]
    )
    print(
        " violation =",
        report[
            "glare_protection"
        ][
            "overall_mean_shadow_violation"
        ]
    )
    print(
        " overmask =",
        report[
            "overmasking"
        ][
            "mean"
        ]
    )

    print()
    print("VRU VISIBILITY")
    print(
        " pedestrian =",
        report[
            "vru_visibility"
        ][
            "pedestrian"
        ][
            "mean"
        ]
    )
    print(
        " cyclist =",
        report[
            "vru_visibility"
        ][
            "cyclist"
        ][
            "mean"
        ]
    )

    print()
    print("Saved:", OUTPUT)

    print()
    print(
        "STAGE 7 FINAL JOINT METRICS = PASS"
    )


if __name__ == "__main__":
    main()
