from pathlib import Path
import json


BEAM = Path(
    "stage7_real/reports/"
    "stage7_joint_beam_evaluation.json"
)

CI = Path(
    "stage7_real/reports/"
    "stage7_joint_confidence_intervals.json"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_vehicle_communication.json"
)

REP_BEAMS = 64
REP_TARGET = 0.95


def find_operating_point(rows):

    matches = [
        x for x in rows
        if int(x["num_beams"]) == REP_BEAMS
        and abs(
            float(x["target_mass"])
            - REP_TARGET
        ) < 1e-12
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "64-beam / 0.95 vehicle operating point"
        )

    return matches[0]


def main():

    print("=" * 88)
    print("STAGE 7 MAIN VEHICLE COMMUNICATION EVALUATION")
    print("=" * 88)

    beam = json.loads(
        BEAM.read_text()
    )

    ci = json.loads(
        CI.read_text()
    )

    vehicle = beam[
        "results_by_class"
    ][
        "VEHICLE"
    ]

    # Support either a direct list or an
    # adaptive_topk list inside the class block.
    if isinstance(vehicle, list):
        rows = vehicle
    elif "adaptive_topk" in vehicle:
        rows = vehicle[
            "adaptive_topk"
        ]
    else:
        raise RuntimeError(
            "Unexpected VEHICLE beam-result structure: "
            + str(vehicle.keys())
        )

    op = find_operating_point(
        rows
    )

    vehicle_ci = ci[
        "by_class"
    ][
        "VEHICLE"
    ]

    coverage_ci = vehicle_ci[
        "beam_mean_covered_mass"
    ]

    overhead_ci = vehicle_ci[
        "beam_mean_overhead_fraction"
    ]

    k_ci = vehicle_ci[
        "beam_mean_selected_k"
    ]

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "evaluation":
            "main_connected_vehicle_communication",

        "receiver_scope":
            "VEHICLE",

        "receiver_semantics": (
            "Primary communication evaluation is restricted "
            "to connected vehicle receivers. Pedestrians and "
            "cyclists remain ADB/safety-relevant actors and "
            "their communication results, when reported, are "
            "supplementary rather than part of the primary "
            "communication endpoint."
        ),

        "shared_posterior":
            True,

        "future_used_as_input":
            False,

        "operating_point": {
            "num_beams":
                REP_BEAMS,

            "target_mass":
                REP_TARGET,
        },

        "communication_reliability": {
            "metric":
                "posterior probability coverage",

            "mean":
                op[
                    "covered_mass"
                ][
                    "mean"
                ],

            "ci95":
                [
                    coverage_ci[
                        "ci95_low"
                    ],
                    coverage_ci[
                        "ci95_high"
                    ],
                ],
        },

        "beam_overhead": {
            "mean_selected_k":
                op[
                    "selected_k"
                ][
                    "mean"
                ],

            "mean_selected_k_ci95":
                [
                    k_ci[
                        "ci95_low"
                    ],
                    k_ci[
                        "ci95_high"
                    ],
                ],

            "mean_overhead_fraction":
                op[
                    "mean_overhead_fraction"
                ],

            "mean_overhead_fraction_ci95":
                [
                    overhead_ci[
                        "ci95_low"
                    ],
                    overhead_ci[
                        "ci95_high"
                    ],
                ],
        },

        "sources": {
            "beam_evaluation":
                str(BEAM),

            "confidence_intervals":
                str(CI),
        },

        "scientific_scope": (
            "This report defines the primary Stage-7 "
            "communication endpoint on VEHICLE receivers. "
            "The all-class joint cohort is retained for "
            "illumination and safety evaluation. Existing "
            "pedestrian/cyclist communication calculations "
            "are not discarded, but are treated as "
            "supplementary connected-VRU analyses."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("receiver scope = VEHICLE")
    print(
        "beams =",
        REP_BEAMS
    )
    print(
        "target mass =",
        REP_TARGET
    )

    print()
    print("COMMUNICATION RELIABILITY")
    print(
        "coverage =",
        report[
            "communication_reliability"
        ][
            "mean"
        ]
    )
    print(
        "CI95 =",
        report[
            "communication_reliability"
        ][
            "ci95"
        ]
    )

    print()
    print("BEAM OVERHEAD")
    print(
        "mean K =",
        report[
            "beam_overhead"
        ][
            "mean_selected_k"
        ]
    )
    print(
        "overhead =",
        report[
            "beam_overhead"
        ][
            "mean_overhead_fraction"
        ]
    )
    print(
        "overhead CI95 =",
        report[
            "beam_overhead"
        ][
            "mean_overhead_fraction_ci95"
        ]
    )

    print()
    print("Saved:", OUTPUT)

    print()
    print(
        "STAGE 7 VEHICLE COMMUNICATION = PASS"
    )


if __name__ == "__main__":
    main()
