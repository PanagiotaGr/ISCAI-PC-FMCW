from pathlib import Path
import json
import numpy as np


SLICES = Path(
    "stage7_real/data/"
    "stage7_failure_case_slices.npz"
)

METRICS = Path(
    "stage7_real/data/"
    "stage7_joint_actor_metrics.npz"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_failure_case_analysis.json"
)


SLICE_NAMES = [
    "near_beam_boundary",
    "near_fov_edge",
    "predicted_entry",
    "close_range",
    "high_predicted_angular_change",
    "vehicle",
    "pedestrian",
    "cyclist",
    "close_range_and_high_angular_change",
    "beam_boundary_and_high_angular_change",
    "fov_edge_or_predicted_entry",
]


ADB_FIELDS = [
    "adb_mean_iou",
    "adb_mean_shadow_violation",
    "adb_mean_overmask",
    "adb_mean_road_retention",
    "adb_mean_oracle_region_visibility",
]


COMM_FIELDS = [
    "beam_mean_selected_k",
    "beam_mean_covered_mass",
    "beam_mean_overhead_fraction",
]


def mean(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if len(x) == 0:
        return None

    return float(
        np.mean(x)
    )


def summarize_fields(
    d,
    mask,
    fields,
):

    result = {
        "actors":
            int(mask.sum())
    }

    for field in fields:

        result[field] = mean(
            d[field][mask]
        )

    return result


def delta(
    value,
    reference,
):

    if (
        value is None
        or
        reference is None
    ):
        return None

    return float(
        value - reference
    )


def main():

    print("=" * 96)
    print("STAGE 7 FAILURE-CASE / SAFETY-CRITICAL ANALYSIS")
    print("=" * 96)

    s = np.load(SLICES)
    m = np.load(METRICS)

    # --------------------------------------------------------
    # Alignment audit
    # --------------------------------------------------------

    if len(
        s["scenario_id"]
    ) != len(
        m["scenario_id"]
    ):
        raise RuntimeError(
            "Slice/metric actor-count mismatch"
        )

    if not np.array_equal(
        s["scenario_id"].astype(str),
        m["scenario_id"].astype(str),
    ):
        raise RuntimeError(
            "scenario_id alignment mismatch"
        )

    if not np.array_equal(
        s["track_index"],
        m["track_index"],
    ):
        raise RuntimeError(
            "track_index alignment mismatch"
        )

    classes = m[
        "actor_class"
    ].astype(str)

    all_mask = np.ones(
        len(classes),
        dtype=bool,
    )

    vehicle_mask = (
        classes == "VEHICLE"
    )

    # --------------------------------------------------------
    # Overall references
    # --------------------------------------------------------

    overall_adb = summarize_fields(
        m,
        all_mask,
        ADB_FIELDS,
    )

    overall_vehicle_comm = (
        summarize_fields(
            m,
            vehicle_mask,
            COMM_FIELDS,
        )
    )

    overall_allclass_comm = (
        summarize_fields(
            m,
            all_mask,
            COMM_FIELDS,
        )
    )

    rows = {}

    print()
    print("OVERALL REFERENCES")
    print(
        "  all actors =",
        int(all_mask.sum())
    )
    print(
        "  vehicle communication actors =",
        int(vehicle_mask.sum())
    )

    for name in SLICE_NAMES:

        if name not in s.files:
            raise RuntimeError(
                f"Missing slice: {name}"
            )

        mask = s[
            name
        ].astype(bool)

        # ====================================================
        # ADB / safety:
        # complete slice
        # ====================================================

        adb = summarize_fields(
            m,
            mask,
            ADB_FIELDS,
        )

        # ====================================================
        # Primary communication:
        # VEHICLE receivers inside the slice
        # ====================================================

        comm_vehicle_mask = (
            mask
            &
            vehicle_mask
        )

        comm_vehicle = (
            summarize_fields(
                m,
                comm_vehicle_mask,
                COMM_FIELDS,
            )
        )

        # ====================================================
        # Supplementary connected-VRU-inclusive communication
        # ====================================================

        comm_all = (
            summarize_fields(
                m,
                mask,
                COMM_FIELDS,
            )
        )

        comparison = {
            "adb_vs_overall": {
                field:
                    delta(
                        adb[field],
                        overall_adb[field],
                    )
                for field in ADB_FIELDS
            },

            "vehicle_communication_vs_overall_vehicle": {
                field:
                    delta(
                        comm_vehicle[field],
                        overall_vehicle_comm[field],
                    )
                for field in COMM_FIELDS
            },
        }

        rows[name] = {
            "slice_actors":
                int(mask.sum()),

            "actor_fraction":
                float(mask.mean()),

            "class_counts": {
                cls:
                    int(
                        np.sum(
                            mask
                            &
                            (classes == cls)
                        )
                    )
                for cls in (
                    "VEHICLE",
                    "PEDESTRIAN",
                    "CYCLIST",
                )
            },

            "primary_vehicle_communication":
                comm_vehicle,

            "adb_and_safety":
                adb,

            "supplementary_all_class_communication":
                comm_all,

            "comparison_to_overall":
                comparison,
        }

    # --------------------------------------------------------
    # Rank potentially difficult slices.
    #
    # We rank descriptively, not as a statistical hypothesis
    # test.
    # --------------------------------------------------------

    vehicle_comm_rank = []

    adb_violation_rank = []

    adb_iou_rank = []

    for name, row in rows.items():

        vc = row[
            "primary_vehicle_communication"
        ]

        if (
            vc["actors"] > 0
            and
            vc[
                "beam_mean_covered_mass"
            ] is not None
        ):
            vehicle_comm_rank.append(
                (
                    name,
                    vc["actors"],
                    vc[
                        "beam_mean_covered_mass"
                    ],
                )
            )

        adb = row[
            "adb_and_safety"
        ]

        if adb["actors"] > 0:

            adb_violation_rank.append(
                (
                    name,
                    adb["actors"],
                    adb[
                        "adb_mean_shadow_violation"
                    ],
                )
            )

            adb_iou_rank.append(
                (
                    name,
                    adb["actors"],
                    adb[
                        "adb_mean_iou"
                    ],
                )
            )

    vehicle_comm_rank.sort(
        key=lambda x:
            x[2]
    )

    adb_violation_rank.sort(
        key=lambda x:
            x[2],
        reverse=True,
    )

    adb_iou_rank.sort(
        key=lambda x:
            x[2]
    )

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "analysis":
            "failure_case_and_safety_critical_subsets",

        "actors":
            int(len(classes)),

        "protocol": {
            "primary_communication_scope":
                "VEHICLE",

            "adb_safety_scope":
                "all actors in each slice",

            "all_class_communication":
                "supplementary_only",
        },

        "overall_reference": {
            "adb_and_safety":
                overall_adb,

            "primary_vehicle_communication":
                overall_vehicle_comm,

            "supplementary_all_class_communication":
                overall_allclass_comm,
        },

        "results":
            rows,

        "descriptive_rankings": {
            "lowest_vehicle_communication_coverage":
                [
                    {
                        "slice":
                            x[0],

                        "vehicle_actors":
                            int(x[1]),

                        "mean_covered_mass":
                            float(x[2]),
                    }
                    for x in vehicle_comm_rank
                ],

            "highest_adb_shadow_violation":
                [
                    {
                        "slice":
                            x[0],

                        "actors":
                            int(x[1]),

                        "mean_shadow_violation":
                            float(x[2]),
                    }
                    for x in adb_violation_rank
                ],

            "lowest_adb_iou":
                [
                    {
                        "slice":
                            x[0],

                        "actors":
                            int(x[1]),

                        "mean_iou":
                            float(x[2]),
                    }
                    for x in adb_iou_rank
                ],
        },

        "limitations": {
            "not_directly_available": [
                "lane-change label",
                "braking label",
                "intersection-turn label",
                "partial-occlusion field",
                "temporary-missing-track field",
                "LiDAR point-count field",
                "dynamic-blocker field",
            ],

            "interpretation": (
                "Unavailable failure labels are not "
                "fabricated. Geometry- and class-based "
                "subsets are evaluated directly."
            ),
        },

        "sources": {
            "slices":
                str(SLICES),

            "actor_metrics":
                str(METRICS),
        },

        "scientific_scope": (
            "Failure-case performance is reported using "
            "predefined geometry/class-based subsets. "
            "Communication metrics use connected VEHICLE "
            "receivers as the primary endpoint, while "
            "ADB/safety metrics use all relevant actors."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("=" * 96)
    print("FAILURE-CASE RESULTS")
    print("=" * 96)

    for name in SLICE_NAMES:

        row = rows[name]

        adb = row[
            "adb_and_safety"
        ]

        comm = row[
            "primary_vehicle_communication"
        ]

        print()
        print(name)
        print(
            "  actors =",
            row[
                "slice_actors"
            ],
            row[
                "class_counts"
            ]
        )

        print(
            "  vehicle communication actors =",
            comm["actors"]
        )

        if comm["actors"] > 0:
            print(
                "  vehicle coverage =",
                comm[
                    "beam_mean_covered_mass"
                ]
            )
            print(
                "  vehicle mean K =",
                comm[
                    "beam_mean_selected_k"
                ]
            )
            print(
                "  vehicle overhead =",
                comm[
                    "beam_mean_overhead_fraction"
                ]
            )

        print(
            "  ADB IoU =",
            adb[
                "adb_mean_iou"
            ]
        )
        print(
            "  shadow violation =",
            adb[
                "adb_mean_shadow_violation"
            ]
        )
        print(
            "  overmask =",
            adb[
                "adb_mean_overmask"
            ]
        )
        print(
            "  visibility =",
            adb[
                "adb_mean_oracle_region_visibility"
            ]
        )

    print()
    print("=" * 96)
    print("DESCRIPTIVE WORST SLICES")
    print("=" * 96)

    print()
    print("Lowest vehicle communication coverage")

    for x in report[
        "descriptive_rankings"
    ][
        "lowest_vehicle_communication_coverage"
    ][:5]:
        print(
            " ",
            x[
                "slice"
            ],
            "N=",
            x[
                "vehicle_actors"
            ],
            "coverage=",
            x[
                "mean_covered_mass"
            ],
        )

    print()
    print("Highest ADB shadow violation")

    for x in report[
        "descriptive_rankings"
    ][
        "highest_adb_shadow_violation"
    ][:5]:
        print(
            " ",
            x[
                "slice"
            ],
            "N=",
            x[
                "actors"
            ],
            "violation=",
            x[
                "mean_shadow_violation"
            ],
        )

    print()
    print("Lowest ADB IoU")

    for x in report[
        "descriptive_rankings"
    ][
        "lowest_adb_iou"
    ][:5]:
        print(
            " ",
            x[
                "slice"
            ],
            "N=",
            x[
                "actors"
            ],
            "IoU=",
            x[
                "mean_iou"
            ],
        )

    print()
    print("Saved:", OUTPUT)

    print()
    print(
        "STAGE 7 FAILURE-CASE ANALYSIS = PASS"
    )


if __name__ == "__main__":
    main()
