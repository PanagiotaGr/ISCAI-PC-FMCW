from pathlib import Path
import importlib.util
import json
import time

import numpy as np


# ============================================================
# Paths / configuration
# ============================================================

ROLLING = Path(
    "stage6_real/data/"
    "real_stage6_predictions_rolling.npz"
)

EVALUATOR = Path(
    "stage6_real/scripts/"
    "evaluate_stage6_streaming.py"
)

REPORT = Path(
    "stage6_real/reports/"
    "stage6_temporal_controller_evaluation.json"
)

# First future prediction = one WOMD step = 0.1 s ahead.
CONTROL_HORIZON = 0

GAMMAS = [
    0.10,
    0.25,
    0.50,
]

# Sensitivity sweep only.
# These are NOT claimed as hardware-calibrated constants.
ALPHAS = [
    0.25,
    0.50,
    0.75,
    1.00,
]

MAX_DELTAS = [
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
]

SEED = 42


# ============================================================
# Load canonical Stage-6 implementation
# ============================================================

spec = importlib.util.spec_from_file_location(
    "stage6_eval",
    EVALUATOR,
)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# ============================================================
# Controller
# ============================================================

def apply_controller(
    target,
    previous,
    alpha,
    max_delta,
):
    """
    Stateful temporal ADB controller.

    Step 1:
        exponential temporal smoothing.

    Step 2:
        cellwise maximum actuation change per controller step.

    The first frame initializes directly from the target, avoiding
    an artificial all-on/all-off startup transient.
    """

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    if previous is None:

        return (
            target.copy(),
            0.0,
        )

    smoothed = (
        float(alpha)
        *
        target
        +
        (
            1.0
            -
            float(alpha)
        )
        *
        previous
    )

    raw_delta = (
        smoothed
        -
        previous
    )

    limited_delta = np.clip(
        raw_delta,
        -float(max_delta),
        float(max_delta),
    )

    command = (
        previous
        +
        limited_delta
    )

    limiter_active = float(
        np.mean(
            np.abs(raw_delta)
            >
            float(max_delta)
        )
    )

    return (
        command.astype(
            np.float32,
            copy=False,
        ),
        limiter_active,
    )


# ============================================================
# Target map construction
# ============================================================

def class_aware_target(
    data,
    row,
    gamma,
    rng,
):

    cls = str(
        data[
            "actor_class"
        ][row]
    )

    config = m.CLASS_CONFIGS[
        cls
    ]

    mu = data[
        "future_mean_H_m"
    ][row]

    std = data[
        "future_std_xy_H_m"
    ][row]

    rho = data[
        "future_rho_xy_H"
    ][row]

    length = float(
        data[
            "length_m"
        ][row]
    )

    width = float(
        data[
            "width_m"
        ][row]
    )

    heading = float(
        data[
            "current_heading_H_rad"
        ][row]
    )

    t = CONTROL_HORIZON

    # --------------------------------------------------------
    # Deterministic predictive base
    # --------------------------------------------------------

    deterministic_L = m.intensity_map(
        mu[
            t,
            :2
        ],
        length,
        width,
        heading,
    )

    deterministic_D = (
        1.0
        -
        deterministic_L
    )

    # --------------------------------------------------------
    # Class-dependent dynamic margin
    # --------------------------------------------------------

    class_margin_scale = float(
        config[
            "lateral_margin_scale"
        ]
    )

    if cls == "VEHICLE":

        x = m.vehicle_dynamic_margin_scale(
            mu[
                :,
                :2
            ],
            std,
            rho,
            t,
            class_margin_scale,
            dt_s=0.1,
        )

        class_margin_scale = float(
            x[
                "scale"
            ]
        )

    elif cls == "CYCLIST":

        x = m.cyclist_dynamic_margin_scale(
            mu[
                :,
                :2
            ],
            heading,
            t,
            class_margin_scale,
            dt_s=0.1,
        )

        class_margin_scale = float(
            x[
                "scale"
            ]
        )

    # --------------------------------------------------------
    # Same class-aware probabilistic occupancy used by the
    # canonical Stage-6 evaluator.
    # --------------------------------------------------------

    occupancy = m.probabilistic_occupancy(
        mu[
            t,
            :2
        ],
        std[
            t
        ],
        rho[
            t
        ],
        length,
        width,
        heading,
        class_margin_scale,
        rng,
    )

    class_mask = (
        occupancy
        >=
        float(gamma)
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

    target = (
        deterministic_D.copy()
    )

    target[
        class_mask
    ] = np.maximum(
        target[
            class_mask
        ],
        max_dimming,
    )

    target = np.minimum(
        target,
        max_dimming,
    )

    return target.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Metrics
# ============================================================

def new_accumulator():

    return {
        "frames":
            0,

        "temporal_pairs":
            0,

        "target_iou":
            [],

        "target_region_miss":
            [],

        "overmask_vs_target":
            [],

        "road_retention":
            [],

        "target_tracking_mae":
            [],

        "change_rate":
            [],

        "temporal_smoothness":
            [],

        "flicker_0_10":
            [],

        "flicker_0_25":
            [],

        "flicker_0_50":
            [],

        "limiter_active_fraction":
            [],
    }


def append_metrics(
    acc,
    command,
    target,
    previous,
    limiter_active,
):

    acc[
        "frames"
    ] += 1

    acc[
        "target_iou"
    ].append(
        m.iou(
            command,
            target,
        )
    )

    # Same mathematical form as shadow-violation, but target map
    # is the reference. Name it explicitly as target-region miss.
    acc[
        "target_region_miss"
    ].append(
        m.shadow_violation(
            command,
            target,
        )
    )

    acc[
        "overmask_vs_target"
    ].append(
        m.overmask(
            command,
            target,
        )
    )

    acc[
        "road_retention"
    ].append(
        m.road_retention(
            command
        )
    )

    acc[
        "target_tracking_mae"
    ].append(
        float(
            np.mean(
                np.abs(
                    command
                    -
                    target
                )
            )
        )
    )

    acc[
        "limiter_active_fraction"
    ].append(
        float(
            limiter_active
        )
    )

    if previous is None:
        return

    acc[
        "temporal_pairs"
    ] += 1

    acc[
        "change_rate"
    ].append(
        m.change_rate(
            command,
            previous,
        )
    )

    acc[
        "temporal_smoothness"
    ].append(
        m.temporal_smoothness(
            command,
            previous,
        )
    )

    acc[
        "flicker_0_10"
    ].append(
        m.flicker_rate(
            command,
            previous,
            0.10,
        )
    )

    acc[
        "flicker_0_25"
    ].append(
        m.flicker_rate(
            command,
            previous,
            0.25,
        )
    )

    acc[
        "flicker_0_50"
    ].append(
        m.flicker_rate(
            command,
            previous,
            0.50,
        )
    )


def summarize(acc):

    out = {
        "frames":
            int(
                acc[
                    "frames"
                ]
            ),

        "temporal_pairs":
            int(
                acc[
                    "temporal_pairs"
                ]
            ),
    }

    for key, values in acc.items():

        if key in {
            "frames",
            "temporal_pairs",
        }:
            continue

        if values:

            out[
                "mean_"
                +
                key
            ] = float(
                np.mean(
                    values
                )
            )

        else:

            out[
                "mean_"
                +
                key
            ] = None

    return out


# ============================================================
# Main
# ============================================================

def main():

    t0 = time.perf_counter()

    data = np.load(
        ROLLING
    )

    seq = data[
        "sequence_index"
    ]

    pos = data[
        "anchor_position"
    ]

    classes = data[
        "actor_class"
    ]

    unique_sequences = np.unique(
        seq
    )

    print("=" * 90)
    print(
        "STAGE 6 ROLLING TEMPORAL / ACTUATION CONTROLLER EVALUATION"
    )
    print("=" * 90)

    print(
        "controller frames =",
        len(seq)
    )

    print(
        "sequences =",
        len(unique_sequences)
    )

    print(
        "control horizon =",
        CONTROL_HORIZON
    )

    print(
        "controller dt = 0.1 s"
    )

    print(
        "gammas =",
        GAMMAS
    )

    print(
        "alphas =",
        ALPHAS
    )

    print(
        "max deltas =",
        MAX_DELTAS
    )

    # --------------------------------------------------------
    # Operating points
    # --------------------------------------------------------

    operating_points = []

    # Explicit no-controller reference.
    operating_points.append(
        (
            "target_reference",
            1.0,
            1.0,
            True,
        )
    )

    for alpha in ALPHAS:

        for delta in MAX_DELTAS:

            name = (
                f"alpha_{alpha:g}"
                f"_delta_{delta:g}"
            )

            operating_points.append(
                (
                    name,
                    float(alpha),
                    float(delta),
                    False,
                )
            )

    # gamma -> method -> accumulator
    results = {
        str(gamma): {
            name:
                new_accumulator()
            for (
                name,
                _,
                _,
                _,
            ) in operating_points
        }
        for gamma in GAMMAS
    }

    results_by_class = {
        str(gamma): {
            cls: {
                name:
                    new_accumulator()
                for (
                    name,
                    _,
                    _,
                    _,
                ) in operating_points
            }
            for cls in m.CLASS_CONFIGS
        }
        for gamma in GAMMAS
    }

    # --------------------------------------------------------
    # Sequence-wise evaluation
    # --------------------------------------------------------

    for seq_number, sequence_id in enumerate(
        unique_sequences,
        start=1,
    ):

        indices = np.flatnonzero(
            seq
            ==
            sequence_id
        )

        indices = indices[
            np.argsort(
                pos[
                    indices
                ]
            )
        ]

        cls_values = np.unique(
            classes[
                indices
            ]
        )

        if len(cls_values) != 1:
            raise RuntimeError(
                "Mixed-class sequence"
            )

        cls = str(
            cls_values[
                0
            ]
        )

        # Previous command is independent for each
        # gamma / controller operating point.
        previous = {
            str(gamma): {
                name:
                    None
                for (
                    name,
                    _,
                    _,
                    _,
                ) in operating_points
            }
            for gamma in GAMMAS
        }

        for row in indices:

            row = int(
                row
            )

            # Use stable row-specific random sampling so every
            # controller operating point sees exactly the same
            # probabilistic target map.
            for gamma_index, gamma in enumerate(
                GAMMAS
            ):

                rng = np.random.default_rng(
                    SEED
                    +
                    100000
                    *
                    int(
                        sequence_id
                    )
                    +
                    100
                    *
                    int(
                        pos[
                            row
                        ]
                    )
                    +
                    gamma_index
                )

                target = class_aware_target(
                    data,
                    row,
                    gamma,
                    rng,
                )

                gamma_key = str(
                    gamma
                )

                for (
                    name,
                    alpha,
                    delta,
                    reference,
                ) in operating_points:

                    prev = previous[
                        gamma_key
                    ][
                        name
                    ]

                    if reference:

                        command = (
                            target.copy()
                        )

                        limiter_active = 0.0

                    else:

                        (
                            command,
                            limiter_active,
                        ) = apply_controller(
                            target,
                            prev,
                            alpha,
                            delta,
                        )

                    append_metrics(
                        results[
                            gamma_key
                        ][
                            name
                        ],
                        command,
                        target,
                        prev,
                        limiter_active,
                    )

                    append_metrics(
                        results_by_class[
                            gamma_key
                        ][
                            cls
                        ][
                            name
                        ],
                        command,
                        target,
                        prev,
                        limiter_active,
                    )

                    previous[
                        gamma_key
                    ][
                        name
                    ] = (
                        command
                    )

        print(
            f"sequence="
            f"{seq_number}/"
            f"{len(unique_sequences)}",
            flush=True,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        gamma: {
            name:
                summarize(
                    acc
                )
            for name, acc in methods.items()
        }
        for gamma, methods in results.items()
    }

    summary_by_class = {
        gamma: {
            cls: {
                name:
                    summarize(
                        acc
                    )
                for name, acc in methods.items()
            }
            for cls, methods in classes_dict.items()
        }
        for gamma, classes_dict
        in results_by_class.items()
    }

    # --------------------------------------------------------
    # Rank operating points.
    #
    # No single "best" hardware controller is claimed.
    # Produce a simple normalized engineering score only as
    # an exploratory sensitivity ranking.
    # --------------------------------------------------------

    rankings = {}

    for gamma_key, methods in summary.items():

        reference = methods[
            "target_reference"
        ]

        candidates = []

        for name, x in methods.items():

            if name == "target_reference":
                continue

            candidates.append(
                {
                    "method":
                        name,

                    "target_tracking_mae":
                        x[
                            "mean_target_tracking_mae"
                        ],

                    "change_rate":
                        x[
                            "mean_change_rate"
                        ],

                    "flicker_0_25":
                        x[
                            "mean_flicker_0_25"
                        ],

                    "target_region_miss":
                        x[
                            "mean_target_region_miss"
                        ],

                    "road_retention":
                        x[
                            "mean_road_retention"
                        ],

                    "limiter_active_fraction":
                        x[
                            "mean_limiter_active_fraction"
                        ],
                }
            )

        rankings[
            gamma_key
        ] = candidates

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    report = {
        "status":
            "PASS",

        "data_source":
            str(
                ROLLING
            ),

        "causal_controller_stream":
            True,

        "future_used_as_input":
            False,

        "controller_frame_interval_s":
            0.1,

        "control_horizon_index":
            CONTROL_HORIZON,

        "control_horizon_semantics":
            (
                "First predictive horizon, corresponding to "
                "one 100-ms WOMD prediction step ahead."
            ),

        "temporal_controller":
            {
                "smoothing":
                    (
                        "S_k = alpha * target_k "
                        "+ (1-alpha) * command_(k-1)"
                    ),

                "rate_limit":
                    (
                        "command_k = command_(k-1) + "
                        "clip(S_k-command_(k-1), "
                        "-delta_max, +delta_max)"
                    ),

                "parameter_semantics":
                    (
                        "alpha and delta_max are sensitivity-sweep "
                        "modeling parameters. They are not measured "
                        "or hardware-calibrated headlamp constants."
                    ),

                "alphas":
                    ALPHAS,

                "max_delta_per_100ms":
                    MAX_DELTAS,
            },

        "gammas":
            GAMMAS,

        "results":
            summary,

        "results_by_class":
            summary_by_class,

        "sensitivity_rankings":
            rankings,

        "scientific_scope":
            (
                "This experiment evaluates temporal command "
                "smoothing and per-frame actuation-rate limiting "
                "on real consecutive WOMD anchor frames. "
                "Target-region miss is measured relative to the "
                "unsmoothed class-aware predictive target, not "
                "relative to physical headlamp hardware or an "
                "oracle actuator measurement."
            ),

        "runtime_s":
            float(
                elapsed
            ),
    }

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("TEMPORAL CONTROLLER SWEEP COMPLETE")
    print("=" * 90)

    print(
        "runtime =",
        elapsed,
        "sec"
    )

    for gamma in GAMMAS:

        gamma_key = str(
            gamma
        )

        print()
        print(
            f"===== gamma={gamma} ====="
        )

        for name, x in summary[
            gamma_key
        ].items():

            print(
                f"{name:28s} "
                f"trackMAE="
                f"{x['mean_target_tracking_mae']:.6f} "
                f"miss="
                f"{100*x['mean_target_region_miss']:.3f}% "
                f"change="
                f"{x['mean_change_rate']:.6f} "
                f"smooth="
                f"{x['mean_temporal_smoothness']:.6f} "
                f"flicker25="
                f"{100*x['mean_flicker_0_25']:.4f}% "
                f"limiter="
                f"{100*x['mean_limiter_active_fraction']:.3f}%"
            )

    print()
    print(
        "Saved:",
        REPORT
    )


if __name__ == "__main__":
    main()
