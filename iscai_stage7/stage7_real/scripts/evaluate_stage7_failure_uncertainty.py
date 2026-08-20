from pathlib import Path
import json
import importlib.util
import numpy as np


COMMON = Path(
    "stage7_real/data/"
    "stage7_common_probabilistic_input.npz"
)

COHORT = Path(
    "stage7_real/data/"
    "stage7_joint_cohort.npz"
)

SLICES = Path(
    "stage7_real/data/"
    "stage7_failure_case_slices.npz"
)

ORACLE = Path(
    "/home/agni/waymo/iscai_stage6/"
    "stage6_real/data/"
    "real_stage6_oracle_future_boxes.npz"
)

STAGE6_EVAL = Path(
    "/home/agni/waymo/iscai_stage6/"
    "stage6_real/scripts/"
    "evaluate_stage6_streaming.py"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_failure_uncertainty_analysis.json"
)


SCALES = [
    1.0,
    1.5,
    2.0,
]

SLICE_NAMES = [
    "predicted_entry",
    "high_predicted_angular_change",
    "close_range_and_high_angular_change",
    "beam_boundary_and_high_angular_change",
    "fov_edge_or_predicted_entry",
    "vehicle",
    "pedestrian",
    "cyclist",
]

NUM_BEAMS = 64
TARGET_MASS = 0.95
ADB_GAMMA = 0.25
MC_SAMPLES = 500
SEED = 42


def load_stage6_module():

    spec = importlib.util.spec_from_file_location(
        "stage6_eval",
        STAGE6_EVAL,
    )

    m = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(m)

    return m


def sample_gaussian(
    mu,
    std,
    rho,
    n,
    rng,
):

    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)

    sx = float(std[0])
    sy = float(std[1])

    r = float(
        np.clip(
            rho,
            -0.999,
            0.999,
        )
    )

    xs = (
        float(mu[0])
        +
        sx * z1
    )

    ys = (
        float(mu[1])
        +
        sy
        *
        (
            r * z1
            +
            np.sqrt(
                1.0 - r*r
            )
            *
            z2
        )
    )

    return xs, ys


def beam_probabilities(
    angles,
    num_beams,
):

    normalized = (
        angles + np.pi
    ) / (
        2.0 * np.pi
    )

    idx = np.floor(
        normalized
        *
        num_beams
    ).astype(np.int64)

    idx = np.clip(
        idx,
        0,
        num_beams - 1,
    )

    counts = np.bincount(
        idx,
        minlength=num_beams,
    ).astype(np.float64)

    if counts.sum() > 0:
        counts /= counts.sum()

    return counts


def adaptive_topk(
    p,
    target,
):

    ranked = np.argsort(
        p
    )[::-1]

    selected = []
    mass = 0.0

    for b in ranked:

        selected.append(
            int(b)
        )

        mass += float(
            p[b]
        )

        if mass >= target:
            break

    return (
        selected,
        mass,
    )


def mean(x):

    if len(x) == 0:
        return None

    return float(
        np.mean(
            np.asarray(
                x,
                dtype=np.float64,
            )
        )
    )


def main():

    print("=" * 96)
    print("STAGE 7 FAILURE-SLICE UNCERTAINTY RESPONSE")
    print("=" * 96)

    m = load_stage6_module()

    d = np.load(COMMON)
    c = np.load(COHORT)
    s = np.load(SLICES)
    o = np.load(ORACLE)

    relevant = c[
        "joint_relevant"
    ].astype(bool)

    global_idx = np.flatnonzero(
        relevant
    )

    classes = d[
        "actor_class"
    ][global_idx].astype(str)

    mu = d[
        "trajectory_mean_xy_H_m"
    ][global_idx]

    std_base = d[
        "trajectory_std_xy_H_m"
    ][global_idx]

    rho = d[
        "trajectory_rho_xy_H"
    ][global_idx]

    source_idx = c[
        "stage6_source_index"
    ][global_idx]

    gt_center = o[
        "future_center_H_m"
    ][source_idx]

    gt_heading = o[
        "future_heading_H_rad"
    ][source_idx]

    gt_lwh = o[
        "future_lwh_m"
    ][source_idx]

    pred_heading = o[
        "current_heading_H_rad"
    ][source_idx]

    pred_lwh = o[
        "current_lwh_m"
    ][source_idx]

    results = {}

    for slice_name in SLICE_NAMES:

        if slice_name not in s.files:
            raise RuntimeError(
                f"Missing slice: {slice_name}"
            )

        slice_mask = s[
            slice_name
        ].astype(bool)

        actor_ids = np.flatnonzero(
            slice_mask
        )

        print()
        print("=" * 96)
        print(
            slice_name,
            "actors=",
            len(actor_ids),
        )
        print("=" * 96)

        slice_results = []

        for scale in SCALES:

            rng_beam = np.random.default_rng(
                SEED
            )

            rng_adb = np.random.default_rng(
                SEED
            )

            ks = []
            masses = []

            ious = []
            violations = []
            overmasks = []
            retentions = []

            for actor in actor_ids:

                cls = classes[
                    actor
                ]

                config = m.CLASS_CONFIGS[
                    cls
                ]

                for t in range(
                    mu.shape[1]
                ):

                    std = (
                        std_base[
                            actor,
                            t
                        ]
                        *
                        scale
                    )

                    # --------------------------------------
                    # Communication
                    # --------------------------------------

                    xs, ys = sample_gaussian(
                        mu[
                            actor,
                            t,
                        ],
                        std,
                        rho[
                            actor,
                            t
                        ],
                        MC_SAMPLES,
                        rng_beam,
                    )

                    angles = np.arctan2(
                        ys,
                        xs,
                    )

                    probs = beam_probabilities(
                        angles,
                        NUM_BEAMS,
                    )

                    selected, mass = adaptive_topk(
                        probs,
                        TARGET_MASS,
                    )

                    if cls == "VEHICLE":
                        ks.append(
                            len(selected)
                        )

                        masses.append(
                            mass
                        )

                    # --------------------------------------
                    # ADB
                    # --------------------------------------

                    class_margin_scale = float(
                        config[
                            "lateral_margin_scale"
                        ]
                    )

                    actor_std_all = (
                        std_base[
                            actor
                        ]
                        *
                        scale
                    )

                    if cls == "VEHICLE":

                        vm = (
                            m.vehicle_dynamic_margin_scale(
                                mu[
                                    actor,
                                    :,
                                    :2
                                ],
                                actor_std_all,
                                rho[
                                    actor
                                ],
                                t,
                                class_margin_scale,
                                dt_s=0.1,
                            )
                        )

                        class_margin_scale = float(
                            vm[
                                "scale"
                            ]
                        )

                    elif cls == "CYCLIST":

                        cm = (
                            m.cyclist_dynamic_margin_scale(
                                mu[
                                    actor,
                                    :,
                                    :2
                                ],
                                float(
                                    pred_heading[
                                        actor
                                    ]
                                ),
                                t,
                                class_margin_scale,
                                dt_s=0.1,
                            )
                        )

                        class_margin_scale = float(
                            cm[
                                "scale"
                            ]
                        )

                    occupancy = (
                        m.probabilistic_occupancy(
                            mu[
                                actor,
                                t,
                                :2
                            ],
                            std,
                            rho[
                                actor,
                                t
                            ],
                            float(
                                pred_lwh[
                                    actor,
                                    0
                                ]
                            ),
                            float(
                                pred_lwh[
                                    actor,
                                    1
                                ]
                            ),
                            float(
                                pred_heading[
                                    actor
                                ]
                            ),
                            class_margin_scale,
                            rng_adb,
                        )
                    )

                    deterministic_L = (
                        m.intensity_map(
                            mu[
                                actor,
                                t,
                                :2
                            ],
                            float(
                                pred_lwh[
                                    actor,
                                    0
                                ]
                            ),
                            float(
                                pred_lwh[
                                    actor,
                                    1
                                ]
                            ),
                            float(
                                pred_heading[
                                    actor
                                ]
                            ),
                        )
                    )

                    deterministic_D = (
                        1.0
                        -
                        deterministic_L
                    )

                    class_mask = (
                        occupancy
                        >=
                        ADB_GAMMA
                    )

                    max_dimming = (
                        1.0
                        -
                        float(
                            config[
                                "floor"
                            ]
                        )
                    )

                    class_D = (
                        deterministic_D.copy()
                    )

                    class_D[
                        class_mask
                    ] = np.maximum(
                        class_D[
                            class_mask
                        ],
                        max_dimming,
                    )

                    class_D = np.minimum(
                        class_D,
                        max_dimming,
                    )

                    oracle_L = (
                        m.intensity_map(
                            gt_center[
                                actor,
                                t,
                                :2
                            ],
                            float(
                                gt_lwh[
                                    actor,
                                    t,
                                    0
                                ]
                            ),
                            float(
                                gt_lwh[
                                    actor,
                                    t,
                                    1
                                ]
                            ),
                            float(
                                gt_heading[
                                    actor,
                                    t
                                ]
                            ),
                        )
                    )

                    oracle_D = (
                        1.0
                        -
                        oracle_L
                    )

                    ious.append(
                        m.iou(
                            class_D,
                            oracle_D,
                        )
                    )

                    violations.append(
                        m.shadow_violation(
                            class_D,
                            oracle_D,
                        )
                    )

                    overmasks.append(
                        m.overmask(
                            class_D,
                            oracle_D,
                        )
                    )

                    retentions.append(
                        m.road_retention(
                            class_D
                        )
                    )

            row = {
                "uncertainty_scale":
                    scale,

                "vehicle_communication": {
                    "vehicle_future_points":
                        len(ks),

                    "mean_selected_k":
                        mean(ks),

                    "mean_covered_mass":
                        mean(masses),

                    "mean_overhead_fraction":
                        (
                            None
                            if len(ks) == 0
                            else
                            mean(ks)
                            /
                            NUM_BEAMS
                        ),
                },

                "adb_and_safety": {
                    "future_points":
                        len(ious),

                    "mean_iou":
                        mean(ious),

                    "mean_shadow_violation":
                        mean(
                            violations
                        ),

                    "mean_overmask":
                        mean(
                            overmasks
                        ),

                    "mean_road_retention":
                        mean(
                            retentions
                        ),
                },
            }

            slice_results.append(
                row
            )

            print()
            print(
                f"scale={scale:.1f}"
            )

            vc = row[
                "vehicle_communication"
            ]

            if vc[
                "vehicle_future_points"
            ] > 0:
                print(
                    "  vehicle mean K =",
                    vc[
                        "mean_selected_k"
                    ]
                )
                print(
                    "  vehicle coverage =",
                    vc[
                        "mean_covered_mass"
                    ]
                )
                print(
                    "  vehicle overhead =",
                    vc[
                        "mean_overhead_fraction"
                    ]
                )

            adb = row[
                "adb_and_safety"
            ]

            print(
                "  ADB IoU =",
                adb[
                    "mean_iou"
                ]
            )
            print(
                "  violation =",
                adb[
                    "mean_shadow_violation"
                ]
            )
            print(
                "  overmask =",
                adb[
                    "mean_overmask"
                ]
            )
            print(
                "  retention =",
                adb[
                    "mean_road_retention"
                ]
            )

        baseline = (
            slice_results[0]
        )

        derived = []

        for row in slice_results[1:]:

            vc0 = baseline[
                "vehicle_communication"
            ]

            vc = row[
                "vehicle_communication"
            ]

            adb0 = baseline[
                "adb_and_safety"
            ]

            adb = row[
                "adb_and_safety"
            ]

            derived.append(
                {
                    "from_scale":
                        1.0,

                    "to_scale":
                        row[
                            "uncertainty_scale"
                        ],

                    "delta_vehicle_mean_k":
                        (
                            None
                            if vc[
                                "mean_selected_k"
                            ] is None
                            else
                            vc[
                                "mean_selected_k"
                            ]
                            -
                            vc0[
                                "mean_selected_k"
                            ]
                        ),

                    "delta_vehicle_overhead":
                        (
                            None
                            if vc[
                                "mean_overhead_fraction"
                            ] is None
                            else
                            vc[
                                "mean_overhead_fraction"
                            ]
                            -
                            vc0[
                                "mean_overhead_fraction"
                            ]
                        ),

                    "delta_shadow_violation":
                        adb[
                            "mean_shadow_violation"
                        ]
                        -
                        adb0[
                            "mean_shadow_violation"
                        ],

                    "delta_overmask":
                        adb[
                            "mean_overmask"
                        ]
                        -
                        adb0[
                            "mean_overmask"
                        ],

                    "delta_iou":
                        adb[
                            "mean_iou"
                        ]
                        -
                        adb0[
                            "mean_iou"
                        ],
                }
            )

        results[
            slice_name
        ] = {
            "actors":
                int(
                    len(actor_ids)
                ),

            "scales":
                slice_results,

            "change_from_nominal":
                derived,
        }

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "analysis":
            "failure_slice_uncertainty_response",

        "uncertainty_scales":
            SCALES,

        "communication_scope":
            "VEHICLE_only",

        "adb_scope":
            "all_actors_in_slice",

        "results":
            results,

        "interpretation_policy": (
            "A desirable conservative uncertainty response "
            "typically increases vehicle beam-set size and/or "
            "ADB protection while reducing shadow violation, "
            "with explicitly reported costs in overhead, "
            "overmask and IoU. No monotonic improvement is "
            "assumed a priori."
        ),

        "sources": {
            "common":
                str(COMMON),

            "cohort":
                str(COHORT),

            "failure_slices":
                str(SLICES),

            "oracle":
                str(ORACLE),
        },
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("=" * 96)
    print("Saved:", OUTPUT)
    print()
    print(
        "STAGE 7 FAILURE UNCERTAINTY ANALYSIS = PASS"
    )


if __name__ == "__main__":
    main()
