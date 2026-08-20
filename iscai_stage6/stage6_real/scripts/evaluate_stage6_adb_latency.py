from pathlib import Path
import importlib.util
import json
import time

import numpy as np


EVAL_SCRIPT = Path(
    "stage6_real/scripts/"
    "evaluate_stage6_streaming.py"
)

PRED_FILE = Path(
    "stage6_real/data/"
    "real_stage6_predictions_balanced.npz"
)

ORACLE_FILE = Path(
    "stage6_real/data/"
    "real_stage6_oracle_future_boxes.npz"
)

OUTPUT = Path(
    "stage6_real/reports/"
    "stage6_adb_latency.json"
)


# ============================================================
# Load Stage-6 implementation
# ============================================================

spec = importlib.util.spec_from_file_location(
    "stage6_eval",
    EVAL_SCRIPT,
)

m = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    m
)


def summary_ms(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "samples":
            int(len(x)),

        "mean_ms":
            float(
                1000.0
                *
                x.mean()
            ),

        "median_ms":
            float(
                1000.0
                *
                np.median(x)
            ),

        "p95_ms":
            float(
                1000.0
                *
                np.percentile(
                    x,
                    95
                )
            ),

        "max_ms":
            float(
                1000.0
                *
                x.max()
            ),
    }


def main():

    print("=" * 78)
    print("STAGE 6 REAL ADB GENERATION LATENCY")
    print("=" * 78)

    pred = np.load(
        PRED_FILE
    )

    oracle = np.load(
        ORACLE_FILE
    )

    classes = pred[
        "actor_class"
    ]

    mu = pred[
        "future_mean_H_m"
    ]

    std = pred[
        "future_std_xy_H_m"
    ]

    rho = pred[
        "future_rho_xy_H"
    ]

    length = pred[
        "length_m"
    ]

    width = pred[
        "width_m"
    ]

    pred_heading = pred[
        "current_heading_H_rad"
    ]

    current_center = oracle[
        "current_center_H_m"
    ]

    gt_center = oracle[
        "future_center_H_m"
    ]

    gt_heading = oracle[
        "future_heading_H_rad"
    ]

    gt_lwh = oracle[
        "future_lwh_m"
    ]


    # ========================================================
    # Same causal evaluation cohort as the main Stage-6 run
    # ========================================================

    current_angle = np.degrees(
        np.arctan2(
            current_center[:,1],
            current_center[:,0],
        )
    )

    current_range = np.hypot(
        current_center[:,0],
        current_center[:,1],
    )

    current_fov = (
        (current_center[:,0] > 0)
        &
        (np.abs(current_angle) <= 25.0)
        &
        (current_range <= 180.0)
    )


    pred_angle = np.degrees(
        np.arctan2(
            mu[...,1],
            mu[...,0],
        )
    )

    pred_range = np.hypot(
        mu[...,0],
        mu[...,1],
    )

    predicted_future_fov = (
        (mu[...,0] > 0)
        &
        (np.abs(pred_angle) <= 25.0)
        &
        (pred_range <= 180.0)
    )

    predicted_entry = (
        (~current_fov)
        &
        np.any(
            predicted_future_fov,
            axis=1,
        )
    )

    relevant = np.where(
        current_fov
        |
        predicted_entry
    )[0]


    print(
        "relevant actors =",
        len(relevant)
    )

    print(
        "future states =",
        len(relevant) * 10
    )

    print(
        "MC samples/state =",
        m.MC_SAMPLES
    )


    rng = np.random.default_rng(
        m.SEED
    )


    deterministic_times = []
    uncertainty_times = []
    class_aware_times = []
    oracle_times = []


    # ========================================================
    # Warm-up
    # ========================================================

    if len(relevant):

        actor = int(
            relevant[0]
        )

        cls = str(
            classes[actor]
        )

        config = m.CLASS_CONFIGS[
            cls
        ]

        _ = m.intensity_map(
            mu[
                actor,
                0,
                :2
            ],
            float(
                length[
                    actor
                ]
            ),
            float(
                width[
                    actor
                ]
            ),
            float(
                pred_heading[
                    actor
                ]
            ),
        )

        _ = m.probabilistic_occupancy(
            mu[
                actor,
                0,
                :2
            ],
            std[
                actor,
                0
            ],
            rho[
                actor,
                0
            ],
            float(
                length[
                    actor
                ]
            ),
            float(
                width[
                    actor
                ]
            ),
            float(
                pred_heading[
                    actor
                ]
            ),
            float(
                config[
                    "lateral_margin_scale"
                ]
            ),
            rng,
        )


    # ========================================================
    # Measurement
    # ========================================================

    for pos, actor in enumerate(
        relevant,
        start=1,
    ):

        cls = str(
            classes[
                actor
            ]
        )

        config = m.CLASS_CONFIGS[
            cls
        ]

        max_dimming = (
            1.0
            -
            float(
                config[
                    "floor"
                ]
            )
        )


        for t in range(10):

            # ------------------------------------------------
            # Deterministic predictive map
            # ------------------------------------------------

            tic = time.perf_counter()

            deterministic_L = m.intensity_map(
                mu[
                    actor,
                    t,
                    :2
                ],
                float(
                    length[
                        actor
                    ]
                ),
                float(
                    width[
                        actor
                    ]
                ),
                float(
                    pred_heading[
                        actor
                    ]
                ),
            )

            deterministic_D = (
                1.0
                -
                deterministic_L
            )

            deterministic_times.append(
                time.perf_counter()
                -
                tic
            )


            # ------------------------------------------------
            # Uncertainty-aware class-agnostic
            # ------------------------------------------------

            tic = time.perf_counter()

            occupancy = m.probabilistic_occupancy(
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
                    length[
                        actor
                    ]
                ),
                float(
                    width[
                        actor
                    ]
                ),
                float(
                    pred_heading[
                        actor
                    ]
                ),
                1.0,
                rng,
            )

            mask = (
                occupancy
                >=
                0.25
            )

            uncertainty_D = (
                deterministic_D.copy()
            )

            uncertainty_D[
                mask
            ] = 1.0

            uncertainty_times.append(
                time.perf_counter()
                -
                tic
            )


            # ------------------------------------------------
            # Class-aware predictive
            # ------------------------------------------------

            tic = time.perf_counter()

            occupancy_class = (
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
                        length[
                            actor
                        ]
                    ),
                    float(
                        width[
                            actor
                        ]
                    ),
                    float(
                        pred_heading[
                            actor
                        ]
                    ),
                    float(
                        config[
                            "lateral_margin_scale"
                        ]
                    ),
                    rng,
                )
            )

            class_mask = (
                occupancy_class
                >=
                0.25
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

            class_aware_times.append(
                time.perf_counter()
                -
                tic
            )


            # ------------------------------------------------
            # Constructed oracle reference
            # ------------------------------------------------

            tic = time.perf_counter()

            oracle_L = m.intensity_map(
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

            _ = (
                1.0
                -
                oracle_L
            )

            oracle_times.append(
                time.perf_counter()
                -
                tic
            )


        if (
            pos % 20 == 0
            or
            pos == len(
                relevant
            )
        ):

            print(
                f"actors="
                f"{pos}/"
                f"{len(relevant)}",
                flush=True,
            )


    report = {
        "status":
            "PASS",

        "stage":
            6,

        "data_source":
            "real_WOMD_validation_class_balanced",

        "future_used_as_input":
            False,

        "scope":
            (
                "Measured software ADB map-generation "
                "latency only. No physical actuator latency "
                "is invented."
            ),

        "cohort": {
            "relevant_actors":
                int(
                    len(relevant)
                ),

            "future_states":
                int(
                    len(relevant)
                    *
                    10
                ),
        },

        "configuration": {
            "grid":
                [
                    int(
                        len(
                            m.THETA_DEG
                        )
                    ),
                    int(
                        len(
                            m.RANGE_M
                        )
                    ),
                ],

            "mc_samples_per_state":
                int(
                    m.MC_SAMPLES
                ),

            "latency_gamma_operating_point":
                0.25,

            "gamma_note":
                (
                    "Gamma only changes thresholding cost "
                    "negligibly; 0.25 is used as a representative "
                    "operating point for latency measurement."
                ),
        },

        "latency": {
            "deterministic_predictive_map":
                summary_ms(
                    deterministic_times
                ),

            "uncertainty_aware_map":
                summary_ms(
                    uncertainty_times
                ),

            "class_aware_map":
                summary_ms(
                    class_aware_times
                ),

            "constructed_oracle_reference":
                summary_ms(
                    oracle_times
                ),
        },

        "actuation_latency": {
            "status":
                "NOT_MEASURED",

            "reason":
                (
                    "No physical ADB actuator or validated "
                    "actuator timing model is available."
                ),
        },
    }


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    for name, x in report[
        "latency"
    ].items():

        print(
            f"{name:34s} "
            f"mean={x['mean_ms']:.4f} ms "
            f"p95={x['p95_ms']:.4f} ms "
            f"max={x['max_ms']:.4f} ms"
        )

    print()
    print(
        "Physical actuation latency = NOT MEASURED"
    )

    print(
        "Saved:",
        OUTPUT
    )


if __name__ == "__main__":
    main()
