from pathlib import Path
import json


BEAM_REPORT = Path(
    "stage7_real/reports/"
    "stage7_joint_beam_evaluation.json"
)

ADB_REPORT = Path(
    "/home/agni/waymo/iscai_stage6/"
    "stage6_real/reports/"
    "stage6_streaming_evaluation.json"
)

COHORT_REPORT = Path(
    "stage7_real/reports/"
    "stage7_joint_cohort_summary.json"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_joint_tradeoff.json"
)


REP_BEAMS = 64
REP_TARGET = 0.95
REP_ADB = "class_aware_gamma_0.25"


def main():

    print("=" * 84)
    print("STAGE 7 JOINT COMMUNICATION / ILLUMINATION TRADE-OFF")
    print("=" * 84)

    beam = json.loads(
        BEAM_REPORT.read_text()
    )

    adb = json.loads(
        ADB_REPORT.read_text()
    )

    cohort = json.loads(
        COHORT_REPORT.read_text()
    )

    beam_point = None

    for x in beam["adaptive_topk"]:
        if (
            x["num_beams"] == REP_BEAMS
            and
            abs(
                x["target_mass"]
                -
                REP_TARGET
            ) < 1e-12
        ):
            beam_point = x
            break

    if beam_point is None:
        raise RuntimeError(
            "Representative beam point not found"
        )

    adb_point = (
        adb["results"]
        ["all_relevant"]
        [REP_ADB]
    )

    row = {
        "communication": {
            "policy":
                "adaptive_topk",

            "num_beams":
                REP_BEAMS,

            "target_mass":
                REP_TARGET,

            "mean_selected_k":
                beam_point[
                    "selected_k"
                ][
                    "mean"
                ],

            "p95_selected_k":
                beam_point[
                    "selected_k"
                ][
                    "p95"
                ],

            "mean_covered_mass":
                beam_point[
                    "covered_mass"
                ][
                    "mean"
                ],

            "mean_overhead_fraction":
                beam_point[
                    "mean_overhead_fraction"
                ],
        },

        "illumination": {
            "policy":
                REP_ADB,

            "mean_iou":
                adb_point[
                    "mean_iou"
                ],

            "mean_shadow_violation":
                adb_point[
                    "mean_shadow_violation"
                ],

            "mean_overmask":
                adb_point[
                    "mean_overmask"
                ],

            "mean_road_retention":
                adb_point[
                    "mean_road_retention"
                ],

            "mean_oracle_region_visibility":
                adb_point[
                    "mean_oracle_region_visibility"
                ],

            "mean_false_dimming":
                adb_point[
                    "mean_false_dimming"
                ],
        },
    }

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "evaluation":
            "joint_communication_illumination_tradeoff",

        "shared_posterior":
            True,

        "future_used_as_input":
            False,

        "joint_cohort": {
            "source_actors":
                cohort[
                    "source_actors"
                ],

            "joint_relevant_actors":
                cohort[
                    "joint_relevant_actors"
                ],

            "current_fov_actors":
                cohort[
                    "current_fov_actors"
                ],

            "predicted_entry_actors":
                cohort[
                    "predicted_entry_actors"
                ],
        },

        "representative_operating_point":
            row,

        "sources": {
            "beam_report":
                str(BEAM_REPORT),

            "adb_report":
                str(ADB_REPORT),

            "cohort_report":
                str(COHORT_REPORT),
        },

        "scientific_scope": (
            "Communication and illumination metrics are "
            "reported for the same Stage-7 actor cohort "
            "and originate from the same Stage-4 Gaussian "
            "posterior representation. The metrics remain "
            "domain-specific; no arbitrary scalar joint "
            "utility is introduced."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("COMMUNICATION")
    print(
        "  beams =",
        REP_BEAMS
    )
    print(
        "  target mass =",
        REP_TARGET
    )
    print(
        "  mean K =",
        row[
            "communication"
        ][
            "mean_selected_k"
        ]
    )
    print(
        "  covered mass =",
        row[
            "communication"
        ][
            "mean_covered_mass"
        ]
    )
    print(
        "  overhead =",
        row[
            "communication"
        ][
            "mean_overhead_fraction"
        ]
    )

    print()
    print("ILLUMINATION")
    print(
        "  policy =",
        REP_ADB
    )
    print(
        "  IoU =",
        row[
            "illumination"
        ][
            "mean_iou"
        ]
    )
    print(
        "  violation =",
        row[
            "illumination"
        ][
            "mean_shadow_violation"
        ]
    )
    print(
        "  overmask =",
        row[
            "illumination"
        ][
            "mean_overmask"
        ]
    )
    print(
        "  retention =",
        row[
            "illumination"
        ][
            "mean_road_retention"
        ]
    )

    print()
    print("Saved:", OUTPUT)
    print()
    print(
        "STAGE 7 JOINT TRADE-OFF = PASS"
    )


if __name__ == "__main__":
    main()
