from pathlib import Path
import json


VEHICLE_COMM = Path(
    "stage7_real/reports/"
    "stage7_vehicle_communication.json"
)

OLD_FINAL = Path(
    "stage7_real/reports/"
    "stage7_final_joint_metrics.json"
)

CI = Path(
    "stage7_real/reports/"
    "stage7_joint_confidence_intervals.json"
)

BEAM = Path(
    "stage7_real/reports/"
    "stage7_joint_beam_evaluation.json"
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

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_final_joint_metrics_pdf_aligned.json"
)


def load(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text()
    )


def main():

    print("=" * 92)
    print("STAGE 7 FINAL JOINT METRICS — PDF-ALIGNED")
    print("=" * 92)

    vehicle = load(VEHICLE_COMM)
    old = load(OLD_FINAL)
    ci = load(CI)
    beam = load(BEAM)
    uncertainty = load(UNCERTAINTY)
    latency = load(LATENCY)
    codebook = load(CODEBOOK)

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert vehicle["status"] == "PASS"
    assert vehicle["receiver_scope"] == "VEHICLE"

    assert old["status"] == "PASS"
    assert ci["status"] == "PASS"
    assert beam["status"] == "PASS"
    assert uncertainty["status"] == "PASS"
    assert latency["status"] == "PASS"
    assert codebook["status"] == "PASS"

    joint_n = int(
        old[
            "joint_cohort"
        ][
            "joint_relevant_actors"
        ]
    )

    if joint_n != 167:
        raise RuntimeError(
            f"Unexpected joint cohort size: {joint_n}"
        )

    # --------------------------------------------------------
    # Primary communication = connected VEHICLE only
    # --------------------------------------------------------

    primary_comm = {
        "scope":
            "VEHICLE",

        "role":
            "primary_connected_receiver",

        "operating_point":
            vehicle[
                "operating_point"
            ],

        "communication_reliability":
            vehicle[
                "communication_reliability"
            ],

        "beam_overhead":
            vehicle[
                "beam_overhead"
            ],

        "source":
            str(VEHICLE_COMM),
    }

    # --------------------------------------------------------
    # Illumination / safety = complete joint-relevant cohort
    # --------------------------------------------------------

    illumination = {
        "scope":
            "ALL_JOINT_RELEVANT_ACTORS",

        "actors":
            joint_n,

        "adb_mask_iou":
            old[
                "adb_mask_iou"
            ],

        "glare_protection":
            old[
                "glare_protection"
            ],

        "overmasking":
            old[
                "overmasking"
            ],

        "road_illumination_retention":
            old[
                "road_illumination_retention"
            ],

        "vru_visibility":
            old[
                "vru_visibility"
            ],
    }

    # --------------------------------------------------------
    # Preserve previous all-class communication result,
    # but explicitly mark it supplementary.
    # --------------------------------------------------------

    supplementary_comm = {
        "scope":
            "VEHICLE_PEDESTRIAN_CYCLIST",

        "role":
            "supplementary_connected_vru_inclusive_analysis",

        "not_primary_endpoint":
            True,

        "communication_reliability":
            old[
                "communication_reliability"
            ],

        "beam_overhead":
            old[
                "beam_overhead"
            ],

        "source":
            str(OLD_FINAL),

        "interpretation": (
            "This all-class communication result is retained "
            "for completeness and exploratory connected-VRU "
            "analysis. It is not used as the primary "
            "communication endpoint."
        ),
    }

    # --------------------------------------------------------
    # Sweep scope declarations
    # --------------------------------------------------------

    sweeps = {
        "uncertainty": {
            "status":
                uncertainty["status"],

            "source":
                str(UNCERTAINTY),

            "scope_note": (
                "Existing Stage-7 uncertainty sensitivity "
                "analysis. Its original evaluated cohort is "
                "preserved and is not relabelled as a "
                "vehicle-only experiment."
            ),
        },

        "latency": {
            "status":
                latency["status"],

            "source":
                str(LATENCY),

            "physical_actuator_latency_measured":
                False,

            "scope_note": (
                "Algorithmic stale-posterior sensitivity. "
                "This is not a physical headlamp-actuation "
                "latency measurement."
            ),
        },

        "codebook": {
            "status":
                codebook["status"],

            "source":
                str(CODEBOOK),

            "scope_note": (
                "Existing beam-codebook sensitivity analysis. "
                "Its original cohort is preserved."
            ),
        },
    }

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "evaluation":
            "pdf_aligned_joint_communication_illumination",

        "shared_posterior":
            True,

        "future_used_as_input":
            False,

        "protocol_alignment": {
            "primary_communication_receiver":
                "VEHICLE",

            "illumination_safety_actor_scope":
                [
                    "VEHICLE",
                    "PEDESTRIAN",
                    "CYCLIST",
                ],

            "pedestrian_cyclist_primary_role":
                "ADB_and_safety",

            "connected_vru_communication":
                "supplementary_only",
        },

        "joint_cohort":
            old[
                "joint_cohort"
            ],

        "primary_communication":
            primary_comm,

        "illumination_and_safety":
            illumination,

        "supplementary_communication":
            supplementary_comm,

        "sweeps":
            sweeps,

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

            "primary_vehicle_communication_ci_source":
                str(CI),

            "illumination_ci_source":
                str(CI),
        },

        "limitations": {
            "physical_adb_actuation_latency_measured":
                False,

            "latency_interpretation":
                (
                    "Reported latency sensitivity measures "
                    "degradation under stale trajectory "
                    "posteriors transformed into the current "
                    "headlamp frame. It does not constitute "
                    "physical actuator-latency measurement."
                ),
        },

        "sources": {
            "vehicle_communication":
                str(VEHICLE_COMM),

            "original_joint_final":
                str(OLD_FINAL),

            "joint_beam_evaluation":
                str(BEAM),

            "confidence_intervals":
                str(CI),

            "uncertainty_sweep":
                str(UNCERTAINTY),

            "latency_sweep":
                str(LATENCY),

            "codebook_sweep":
                str(CODEBOOK),
        },

        "scientific_scope": (
            "The primary communication endpoint is evaluated "
            "on connected VEHICLE receivers. Illumination and "
            "safety evaluation uses the complete joint-relevant "
            "VEHICLE/PEDESTRIAN/CYCLIST cohort. Pedestrian and "
            "cyclist visibility are retained as class-specific "
            "ADB safety metrics. Previously computed all-class "
            "communication metrics are retained only as a "
            "supplementary connected-VRU-inclusive analysis. "
            "No physical ADB actuation latency is claimed."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("PROTOCOL")
    print(
        "  primary communication =",
        report[
            "protocol_alignment"
        ][
            "primary_communication_receiver"
        ]
    )
    print(
        "  illumination actors =",
        joint_n
    )

    pc = report[
        "primary_communication"
    ]

    print()
    print("PRIMARY VEHICLE COMMUNICATION")
    print(
        "  coverage =",
        pc[
            "communication_reliability"
        ][
            "mean"
        ]
    )
    print(
        "  coverage CI95 =",
        pc[
            "communication_reliability"
        ][
            "ci95"
        ]
    )
    print(
        "  mean K =",
        pc[
            "beam_overhead"
        ][
            "mean_selected_k"
        ]
    )
    print(
        "  overhead =",
        pc[
            "beam_overhead"
        ][
            "mean_overhead_fraction"
        ]
    )

    illum = report[
        "illumination_and_safety"
    ]

    print()
    print("ILLUMINATION / SAFETY")
    print(
        "  IoU =",
        illum[
            "adb_mask_iou"
        ][
            "mean"
        ]
    )
    print(
        "  shadow violation =",
        illum[
            "glare_protection"
        ][
            "overall_mean_shadow_violation"
        ]
    )
    print(
        "  pedestrian visibility =",
        illum[
            "vru_visibility"
        ][
            "pedestrian"
        ][
            "mean"
        ]
    )
    print(
        "  cyclist visibility =",
        illum[
            "vru_visibility"
        ][
            "cyclist"
        ][
            "mean"
        ]
    )

    print()
    print(
        "supplementary all-class communication retained =",
        True
    )

    print()
    print("Saved:", OUTPUT)

    print()
    print(
        "STAGE 7 PDF-ALIGNED FINAL JOINT METRICS = PASS"
    )


if __name__ == "__main__":
    main()
