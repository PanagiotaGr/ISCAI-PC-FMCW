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
    "stage7_real/data/"
    "stage7_joint_actor_metrics.npz"
)

REPORT = Path(
    "stage7_real/reports/"
    "stage7_joint_actor_metrics_summary.json"
)


NUM_BEAMS = 64
TARGET_MASS = 0.95
ADB_GAMMA = 0.25

MC_BEAM = 500
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
    probabilities,
    target,
):

    ranked = np.argsort(
        probabilities
    )[::-1]

    selected = []
    mass = 0.0

    for beam in ranked:

        selected.append(
            int(beam)
        )

        mass += float(
            probabilities[
                beam
            ]
        )

        if mass >= target:
            break

    return (
        selected,
        mass,
    )


def mean_or_nan(x):

    a = np.asarray(
        x,
        dtype=np.float64,
    )

    if len(a) == 0:
        return np.nan

    return float(
        np.mean(a)
    )


def main():

    print("=" * 88)
    print("STAGE 7 JOINT ACTOR-LEVEL METRICS")
    print("=" * 88)

    m = load_stage6_module()

    d = np.load(COMMON)
    cohort = np.load(COHORT)
    oracle = np.load(ORACLE)

    relevant = cohort[
        "joint_relevant"
    ].astype(bool)

    selected = np.flatnonzero(
        relevant
    )

    source_idx = cohort[
        "stage6_source_index"
    ][selected]

    scenario_id = d[
        "scenario_id"
    ][selected]

    track_index = d[
        "track_index"
    ][selected]

    classes = d[
        "actor_class"
    ][selected]

    current_fov = cohort[
        "current_fov"
    ][selected]

    predicted_entry = cohort[
        "predicted_entry"
    ][selected]

    mu = d[
        "trajectory_mean_xy_H_m"
    ][selected]

    std = d[
        "trajectory_std_xy_H_m"
    ][selected]

    rho = d[
        "trajectory_rho_xy_H"
    ][selected]

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

    n = len(selected)
    T = mu.shape[1]

    beam_mean_k = np.zeros(
        n,
        dtype=np.float64,
    )

    beam_mean_mass = np.zeros(
        n,
        dtype=np.float64,
    )

    beam_overhead = np.zeros(
        n,
        dtype=np.float64,
    )

    adb_iou = np.zeros(
        n,
        dtype=np.float64,
    )

    adb_violation = np.zeros(
        n,
        dtype=np.float64,
    )

    adb_overmask = np.zeros(
        n,
        dtype=np.float64,
    )

    adb_retention = np.zeros(
        n,
        dtype=np.float64,
    )

    adb_visibility = np.zeros(
        n,
        dtype=np.float64,
    )

    rng_beam = np.random.default_rng(
        SEED
    )

    rng_adb = np.random.default_rng(
        SEED
    )

    for actor in range(n):

        cls = str(
            classes[actor]
        )

        config = m.CLASS_CONFIGS[
            cls
        ]

        ks = []
        masses = []

        ious = []
        violations = []
        overmasks = []
        retentions = []
        visibilities = []

        for t in range(T):

            # ==================================================
            # Communication branch
            # ==================================================

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
                MC_BEAM,
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

            selected_beams, mass = (
                adaptive_topk(
                    probs,
                    TARGET_MASS,
                )
            )

            ks.append(
                len(
                    selected_beams
                )
            )

            masses.append(
                mass
            )

            # ==================================================
            # ADB branch
            # ==================================================

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

            visibilities.append(
                m.oracle_region_visibility(
                    class_D,
                    oracle_D,
                )
            )

        beam_mean_k[
            actor
        ] = mean_or_nan(
            ks
        )

        beam_mean_mass[
            actor
        ] = mean_or_nan(
            masses
        )

        beam_overhead[
            actor
        ] = (
            beam_mean_k[
                actor
            ]
            /
            NUM_BEAMS
        )

        adb_iou[
            actor
        ] = mean_or_nan(
            ious
        )

        adb_violation[
            actor
        ] = mean_or_nan(
            violations
        )

        adb_overmask[
            actor
        ] = mean_or_nan(
            overmasks
        )

        adb_retention[
            actor
        ] = mean_or_nan(
            retentions
        )

        adb_visibility[
            actor
        ] = mean_or_nan(
            visibilities
        )

        if (
            (actor + 1) % 20 == 0
            or
            actor + 1 == n
        ):
            print(
                f"actors={actor+1}/{n}",
                flush=True,
            )

    arrays_to_check = [
        beam_mean_k,
        beam_mean_mass,
        beam_overhead,
        adb_iou,
        adb_violation,
        adb_overmask,
        adb_retention,
        adb_visibility,
    ]

    if not all(
        np.all(
            np.isfinite(x)
        )
        for x in arrays_to_check
    ):
        raise RuntimeError(
            "Non-finite actor-level metric"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT,

        scenario_id=
            scenario_id,

        track_index=
            track_index,

        actor_class=
            classes,

        current_fov=
            current_fov,

        predicted_entry=
            predicted_entry,

        stage6_source_index=
            source_idx,

        beam_mean_selected_k=
            beam_mean_k.astype(
                np.float32
            ),

        beam_mean_covered_mass=
            beam_mean_mass.astype(
                np.float32
            ),

        beam_mean_overhead_fraction=
            beam_overhead.astype(
                np.float32
            ),

        adb_mean_iou=
            adb_iou.astype(
                np.float32
            ),

        adb_mean_shadow_violation=
            adb_violation.astype(
                np.float32
            ),

        adb_mean_overmask=
            adb_overmask.astype(
                np.float32
            ),

        adb_mean_road_retention=
            adb_retention.astype(
                np.float32
            ),

        adb_mean_oracle_region_visibility=
            adb_visibility.astype(
                np.float32
            ),
    )

    class_counts = {
        str(cls):
            int(
                np.sum(
                    classes.astype(str)
                    ==
                    str(cls)
                )
            )
        for cls in np.unique(
            classes
        )
    }

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "actors":
            int(n),

        "future_steps_per_actor":
            int(T),

        "class_counts":
            class_counts,

        "communication_policy": {
            "num_beams":
                NUM_BEAMS,

            "target_mass":
                TARGET_MASS,

            "mc_samples":
                MC_BEAM,
        },

        "adb_policy": {
            "type":
                "class_aware",

            "gamma":
                ADB_GAMMA,
        },

        "aggregate_checks": {
            "beam_mean_selected_k":
                float(
                    beam_mean_k.mean()
                ),

            "beam_mean_covered_mass":
                float(
                    beam_mean_mass.mean()
                ),

            "beam_mean_overhead_fraction":
                float(
                    beam_overhead.mean()
                ),

            "adb_mean_iou":
                float(
                    adb_iou.mean()
                ),

            "adb_mean_shadow_violation":
                float(
                    adb_violation.mean()
                ),

            "adb_mean_overmask":
                float(
                    adb_overmask.mean()
                ),

            "adb_mean_road_retention":
                float(
                    adb_retention.mean()
                ),

            "adb_mean_oracle_region_visibility":
                float(
                    adb_visibility.mean()
                ),
        },

        "statistical_unit":
            "actor",

        "scientific_role": (
            "Actor-level joint communication and "
            "illumination metrics used as the "
            "independent resampling unit for Stage-7 "
            "confidence intervals."
        ),

        "output":
            str(OUTPUT),
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("actors =", n)
    print(
        "class counts =",
        class_counts
    )

    print()
    print("AGGREGATE CHECKS")

    for k, v in report[
        "aggregate_checks"
    ].items():
        print(
            f"  {k:38s} = {v}"
        )

    print()
    print("Saved:", OUTPUT)
    print("Report:", REPORT)

    print()
    print(
        "STAGE 7 ACTOR METRICS = PASS"
    )


if __name__ == "__main__":
    main()
