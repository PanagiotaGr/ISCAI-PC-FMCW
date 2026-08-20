from pathlib import Path
import importlib.util
import json
import time

import numpy as np


# ============================================================
# Load the actual Stage-6 evaluation implementation.
# We benchmark the production evaluation functions directly.
# ============================================================

EVAL_FILE = Path(
    "stage6_real/scripts/evaluate_stage6_streaming.py"
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
    "stage6_adb_computational_latency.json"
)


spec = importlib.util.spec_from_file_location(
    "stage6_eval",
    EVAL_FILE,
)

stage6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage6)


def summary_ms(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "n": int(len(x)),
        "mean_ms": float(np.mean(x)),
        "median_ms": float(np.median(x)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


def timed_call(fn, *args):
    t0 = time.perf_counter_ns()

    result = fn(*args)

    t1 = time.perf_counter_ns()

    elapsed_ms = (
        (t1 - t0)
        / 1_000_000.0
    )

    return result, elapsed_ms


def main():

    print("=" * 78)
    print(
        "STAGE 6 COMPUTATIONAL "
        "ADB GENERATION LATENCY"
    )
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


    # ========================================================
    # Same causal evaluation cohort as Stage-6 evaluation
    # ========================================================

    current_angle = np.degrees(
        np.arctan2(
            current_center[:, 1],
            current_center[:, 0],
        )
    )

    current_range = np.hypot(
        current_center[:, 0],
        current_center[:, 1],
    )

    current_fov = (
        (current_center[:, 0] > 0)
        &
        (np.abs(current_angle) <= 25.0)
        &
        (current_range <= 180.0)
    )

    pred_angle = np.degrees(
        np.arctan2(
            mu[..., 1],
            mu[..., 0],
        )
    )

    pred_range = np.hypot(
        mu[..., 0],
        mu[..., 1],
    )

    predicted_future_fov = (
        (mu[..., 0] > 0)
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

    evaluation_actor = (
        current_fov
        |
        predicted_entry
    )

    actors = np.flatnonzero(
        evaluation_actor
    )


    print(
        "total actors =",
        len(classes),
    )

    print(
        "relevant actors =",
        len(actors),
    )

    print(
        "MC samples =",
        stage6.MC_SAMPLES,
    )

    print(
        "grid =",
        (
            len(stage6.THETA_RAD),
            len(stage6.RANGE_M),
        ),
    )


    # ========================================================
    # Warm-up
    # ========================================================

    warm_actor = int(
        actors[0]
    )

    warm_t = 0

    _ = stage6.intensity_map(
        mu[
            warm_actor,
            warm_t,
            :2
        ],
        float(
            length[
                warm_actor
            ]
        ),
        float(
            width[
                warm_actor
            ]
        ),
        float(
            pred_heading[
                warm_actor
            ]
        ),
    )

    warm_rng = np.random.default_rng(
        12345
    )

    _ = stage6.probabilistic_occupancy(
        mu[
            warm_actor,
            warm_t,
            :2
        ],
        std[
            warm_actor,
            warm_t
        ],
        rho[
            warm_actor,
            warm_t
        ],
        float(
            length[
                warm_actor
            ]
        ),
        float(
            width[
                warm_actor
            ]
        ),
        float(
            pred_heading[
                warm_actor
            ]
        ),
        1.0,
        warm_rng,
    )


    # ========================================================
    # Timings
    #
    # One timing sample = one actor at one future horizon.
    #
    # No disk I/O is included.
    # No model inference is included.
    # No physical actuator latency is claimed.
    # ========================================================

    deterministic_times = []
    occupancy_times = []
    class_policy_times = []
    end_to_end_times = []

    by_class = {
        "VEHICLE": [],
        "PEDESTRIAN": [],
        "CYCLIST": [],
    }

    rng = np.random.default_rng(
        stage6.SEED
    )

    total_states = (
        len(actors)
        *
        mu.shape[1]
    )

    done = 0


    for actor in actors:

        actor = int(
            actor
        )

        cls = str(
            classes[
                actor
            ]
        )

        config = stage6.CLASS_CONFIGS[
            cls
        ]

        for t in range(
            mu.shape[1]
        ):

            # ----------------------------------------------
            # Deterministic future-box -> ADB map
            # ----------------------------------------------

            deterministic_L, dt_det = timed_call(
                stage6.intensity_map,

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
                dt_det
            )


            # ----------------------------------------------
            # Probabilistic occupancy generation
            # ----------------------------------------------

            occupancy, dt_occ = timed_call(
                stage6.probabilistic_occupancy,

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

            occupancy_times.append(
                dt_occ
            )


            # ----------------------------------------------
            # Class-aware policy only
            #
            # Use gamma=0.25 as a representative operating
            # point. This benchmark does NOT select gamma.
            # ----------------------------------------------

            gamma = 0.25

            t0 = time.perf_counter_ns()

            class_mask = (
                occupancy
                >=
                gamma
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

            t1 = time.perf_counter_ns()

            dt_policy = (
                (t1 - t0)
                /
                1_000_000.0
            )

            class_policy_times.append(
                dt_policy
            )


            # ----------------------------------------------
            # Software ADB generation latency
            #
            # Sequential latency of the measured Stage-6
            # components for this state.
            # ----------------------------------------------

            dt_e2e = (
                dt_det
                +
                dt_occ
                +
                dt_policy
            )

            end_to_end_times.append(
                dt_e2e
            )

            by_class[
                cls
            ].append(
                dt_e2e
            )


            done += 1

            if (
                done % 100 == 0
                or
                done == total_states
            ):
                print(
                    f"states={done}/"
                    f"{total_states}"
                )


    # ========================================================
    # Report
    # ========================================================

    report = {
        "benchmark_semantics": (
            "software_computational_latency_only"
        ),

        "physical_actuator_latency_measured":
            False,

        "model_inference_included":
            False,

        "disk_io_included":
            False,

        "timing_unit":
            "milliseconds",

        "timer":
            "time.perf_counter_ns",

        "actors_total":
            int(
                len(classes)
            ),

        "actors_evaluated":
            int(
                len(actors)
            ),

        "states_evaluated":
            int(
                len(
                    end_to_end_times
                )
            ),

        "future_horizons":
            int(
                mu.shape[1]
            ),

        "mc_samples":
            int(
                stage6.MC_SAMPLES
            ),

        "grid_shape": [
            int(
                len(
                    stage6.THETA_RAD
                )
            ),
            int(
                len(
                    stage6.RANGE_M
                )
            ),
        ],

        "representative_gamma":
            0.25,

        "latency": {
            "deterministic_future_box_to_adb":
                summary_ms(
                    deterministic_times
                ),

            "probabilistic_occupancy":
                summary_ms(
                    occupancy_times
                ),

            "class_aware_policy":
                summary_ms(
                    class_policy_times
                ),

            "software_adb_generation":
                summary_ms(
                    end_to_end_times
                ),
        },

        "software_adb_generation_by_class": {
            cls:
                summary_ms(
                    values
                )
            for cls, values
            in by_class.items()
            if values
        },

        "interpretation": (
            "Measured latency covers only the Stage-6 "
            "software ADB-map generation path from already "
            "available probabilistic predictions. It does "
            "not measure prediction-model inference, sensor "
            "latency, communication latency, headlamp ECU "
            "latency, LED/optical response, or any physical "
            "actuator latency."
        ),
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
    print("LATENCY RESULTS")
    print("=" * 78)

    for name, result in report[
        "latency"
    ].items():

        print()
        print(name)

        print(
            " mean   =",
            result[
                "mean_ms"
            ],
            "ms"
        )

        print(
            " median =",
            result[
                "median_ms"
            ],
            "ms"
        )

        print(
            " p95    =",
            result[
                "p95_ms"
            ],
            "ms"
        )

        print(
            " p99    =",
            result[
                "p99_ms"
            ],
            "ms"
        )


    print()
    print(
        "IMPORTANT: these are computational "
        "software latencies only."
    )

    print(
        "Physical ADB actuator latency "
        "was NOT measured."
    )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
