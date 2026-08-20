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

STAGE6_EVAL = Path(
    "/home/agni/waymo/iscai_stage6/"
    "stage6_real/scripts/"
    "evaluate_stage6_streaming.py"
)

STAGE6_ORACLE = Path(
    "/home/agni/waymo/iscai_stage6/"
    "stage6_real/data/"
    "real_stage6_oracle_future_boxes.npz"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_uncertainty_sweep.json"
)

SCALES = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
]

NUM_BEAMS = 64
TARGET_MASS = 0.95
MC_SAMPLES = 500
ADB_GAMMA = 0.25
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
                1.0-r*r
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
        normalized * num_beams
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

    mass = 0.0
    k = 0

    for beam in ranked:

        mass += float(
            p[beam]
        )

        k += 1

        if mass >= target:
            break

    return k, mass


def mean(x):
    return float(
        np.mean(
            np.asarray(
                x,
                dtype=np.float64,
            )
        )
    )


def main():

    print("=" * 84)
    print("STAGE 7 JOINT UNCERTAINTY SWEEP")
    print("=" * 84)

    m = load_stage6_module()

    d = np.load(COMMON)
    cohort = np.load(COHORT)
    oracle = np.load(STAGE6_ORACLE)

    relevant = cohort[
        "joint_relevant"
    ].astype(bool)

    idx = np.flatnonzero(
        relevant
    )

    mu = d[
        "trajectory_mean_xy_H_m"
    ][idx]

    std_base = d[
        "trajectory_std_xy_H_m"
    ][idx]

    rho = d[
        "trajectory_rho_xy_H"
    ][idx]

    classes = d[
        "actor_class"
    ][idx]

    source_idx = cohort[
        "stage6_source_index"
    ][idx]

    gt_center = oracle[
        "future_center_H_m"
    ][source_idx]

    gt_heading = oracle[
        "future_heading_H_rad"
    ][source_idx]

    gt_lwh = oracle[
        "future_lwh_m"
    ][source_idx]

    pred_heading = oracle[
        "current_heading_H_rad"
    ][source_idx]

    pred_lwh = oracle[
        "current_lwh_m"
    ][source_idx]

    rows = []

    for scale in SCALES:

        print()
        print(
            f"uncertainty scale = {scale}"
        )

        rng = np.random.default_rng(
            SEED
        )

        std = (
            std_base
            *
            float(scale)
        )

        beam_k = []
        beam_mass = []

        adb_iou = []
        adb_violation = []
        adb_overmask = []
        adb_retention = []

        n, T, _ = mu.shape

        for actor in range(n):

            cls = str(
                classes[actor]
            )

            config = m.CLASS_CONFIGS[
                cls
            ]

            for t in range(T):

                # --------------------------------------
                # communication branch
                # --------------------------------------

                xs, ys = sample_gaussian(
                    mu[
                        actor,
                        t,
                    ],
                    std[
                        actor,
                        t,
                    ],
                    rho[
                        actor,
                        t,
                    ],
                    MC_SAMPLES,
                    rng,
                )

                angles = np.arctan2(
                    ys,
                    xs,
                )

                probs = beam_probabilities(
                    angles,
                    NUM_BEAMS,
                )

                k, mass = adaptive_topk(
                    probs,
                    TARGET_MASS,
                )

                beam_k.append(k)
                beam_mass.append(mass)

                # --------------------------------------
                # illumination branch
                # --------------------------------------

                class_margin_scale = float(
                    config[
                        "lateral_margin_scale"
                    ]
                )

                if cls == "VEHICLE":

                    vm = (
                        m.vehicle_dynamic_margin_scale(
                            mu[
                                actor,
                                :,
                                :2
                            ],
                            std[
                                actor
                            ],
                            rho[
                                actor
                            ],
                            t,
                            class_margin_scale,
                            dt_s=0.1,
                        )
                    )

                    class_margin_scale = float(
                        vm["scale"]
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
                        cm["scale"]
                    )

                occupancy = (
                    m.probabilistic_occupancy(
                        mu[
                            actor,
                            t,
                            :2
                        ],
                        std[
                            actor,
                            t
                        ],
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
                        rng,
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
                        config["floor"]
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

                adb_iou.append(
                    m.iou(
                        class_D,
                        oracle_D,
                    )
                )

                adb_violation.append(
                    m.shadow_violation(
                        class_D,
                        oracle_D,
                    )
                )

                adb_overmask.append(
                    m.overmask(
                        class_D,
                        oracle_D,
                    )
                )

                adb_retention.append(
                    m.road_retention(
                        class_D
                    )
                )

        row = {
            "uncertainty_scale":
                float(scale),

            "communication": {
                "num_beams":
                    NUM_BEAMS,

                "target_mass":
                    TARGET_MASS,

                "mean_selected_k":
                    mean(beam_k),

                "mean_covered_mass":
                    mean(beam_mass),

                "mean_overhead_fraction":
                    mean(beam_k)
                    /
                    NUM_BEAMS,
            },

            "illumination": {
                "gamma":
                    ADB_GAMMA,

                "mean_iou":
                    mean(adb_iou),

                "mean_shadow_violation":
                    mean(adb_violation),

                "mean_overmask":
                    mean(adb_overmask),

                "mean_road_retention":
                    mean(adb_retention),
            },
        }

        rows.append(row)

        print(
            f"  meanK="
            f"{row['communication']['mean_selected_k']:.4f} "
            f"mass="
            f"{row['communication']['mean_covered_mass']:.6f} "
            f"overhead="
            f"{100*row['communication']['mean_overhead_fraction']:.3f}% "
            f"IoU="
            f"{row['illumination']['mean_iou']:.5f} "
            f"violation="
            f"{100*row['illumination']['mean_shadow_violation']:.3f}% "
            f"overmask="
            f"{100*row['illumination']['mean_overmask']:.3f}% "
            f"retention="
            f"{row['illumination']['mean_road_retention']:.5f}"
        )

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "sweep":
            "predictive_uncertainty_scale",

        "shared_posterior":
            True,

        "joint_relevant_actors":
            int(len(idx)),

        "future_points":
            int(
                len(idx)
                *
                mu.shape[1]
            ),

        "uncertainty_scales":
            SCALES,

        "fixed_communication_policy": {
            "num_beams":
                NUM_BEAMS,

            "target_mass":
                TARGET_MASS,
        },

        "fixed_adb_policy": {
            "type":
                "class_aware",

            "gamma":
                ADB_GAMMA,
        },

        "results":
            rows,

        "scientific_scope": (
            "Only predicted Gaussian standard deviations "
            "are scaled. Predicted means, correlations, "
            "actor cohort, communication policy and ADB "
            "gamma remain fixed."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("Saved:", OUTPUT)
    print()
    print(
        "STAGE 7 UNCERTAINTY SWEEP = PASS"
    )


if __name__ == "__main__":
    main()
