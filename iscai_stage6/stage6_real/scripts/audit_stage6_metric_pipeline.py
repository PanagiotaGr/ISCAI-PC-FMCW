import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(
    "stage6_real/scripts/"
    "evaluate_stage6_streaming.py"
)

spec = importlib.util.spec_from_file_location(
    "stage6_eval",
    SCRIPT,
)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


pred = np.load(m.PRED_FILE)
oracle = np.load(m.ORACLE_FILE)

classes = pred["actor_class"]

mu = pred["future_mean_H_m"]
std = pred["future_std_xy_H_m"]
rho = pred["future_rho_xy_H"]

length = pred["length_m"]
width = pred["width_m"]
heading = pred["current_heading_H_rad"]

current = oracle["current_center_H_m"]

gt_center = oracle["future_center_H_m"]
gt_heading = oracle["future_heading_H_rad"]
gt_lwh = oracle["future_lwh_m"]


# ============================================================
# SAME CAUSAL COHORT AS FINAL EVALUATOR
# ============================================================

current_angle = np.degrees(
    np.arctan2(
        current[:,1],
        current[:,0],
    )
)

current_range = np.hypot(
    current[:,0],
    current[:,1],
)

current_fov = (
    (current[:,0] > 0)
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

future_fov = (
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
        future_fov,
        axis=1,
    )
)

indices = np.where(
    current_fov
    |
    predicted_entry
)[0]


rng = np.random.default_rng(
    m.SEED
)


pairs = [
    (0.10, 0.25),
    (0.25, 0.50),
    (0.10, 0.50),
]


stats = {
    pair: {
        "binary_candidate_changed_cells": 0,
        "binary_candidate_total_cells": 0,

        "iou_changed_states": 0,
        "violation_changed_states": 0,
        "overmask_changed_states": 0,

        "sum_abs_iou_difference": 0.0,
        "sum_abs_violation_difference": 0.0,
        "sum_abs_overmask_difference": 0.0,

        "states": 0,
    }
    for pair in pairs
}


print("=" * 78)
print("STAGE 6 METRIC PIPELINE AUDIT")
print("=" * 78)

print(
    "relevant actors =",
    len(indices)
)


for actor in indices:

    cls = str(
        classes[actor]
    )

    config = m.CLASS_CONFIGS[
        cls
    ]

    max_dimming = (
        1.0
        -
        float(
            config["floor"]
        )
    )


    for t in range(10):

        # ----------------------------------------------------
        # Oracle - exactly as final evaluator
        # ----------------------------------------------------

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

        oracle_D = (
            1.0
            -
            oracle_L
        )


        # ----------------------------------------------------
        # Deterministic predictive
        # ----------------------------------------------------

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
                heading[
                    actor
                ]
            ),
        )

        deterministic_D = (
            1.0
            -
            deterministic_L
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # reproduce the SAME RNG sequence as full evaluator.
        #
        # First call = class-agnostic occupancy.
        # Its result is intentionally discarded here.
        # ----------------------------------------------------

        _ = m.probabilistic_occupancy(
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
                heading[
                    actor
                ]
            ),
            1.0,
            rng,
        )


        # Second call = class-aware occupancy,
        # exactly like the full evaluator.
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
                heading[
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


        candidates = {}

        for gamma in m.GAMMAS:

            mask = (
                occupancy
                >=
                gamma
            )

            D = deterministic_D.copy()

            D[
                mask
            ] = np.maximum(
                D[
                    mask
                ],
                max_dimming,
            )

            D = np.minimum(
                D,
                max_dimming,
            )

            candidates[
                gamma
            ] = D


        for pair in pairs:

            a, b = pair

            A = candidates[a]
            B = candidates[b]

            binary_A = (
                A >= 0.5
            )

            binary_B = (
                B >= 0.5
            )

            changed = np.count_nonzero(
                binary_A
                !=
                binary_B
            )

            row = stats[pair]

            row[
                "binary_candidate_changed_cells"
            ] += int(
                changed
            )

            row[
                "binary_candidate_total_cells"
            ] += int(
                binary_A.size
            )

            row[
                "states"
            ] += 1


            # ----------------------------------------------
            # EXACT FINAL METRIC FUNCTIONS
            # ----------------------------------------------

            iou_A = m.iou(
                A,
                oracle_D,
            )

            iou_B = m.iou(
                B,
                oracle_D,
            )

            viol_A = m.shadow_violation(
                A,
                oracle_D,
            )

            viol_B = m.shadow_violation(
                B,
                oracle_D,
            )

            over_A = m.overmask(
                A,
                oracle_D,
            )

            over_B = m.overmask(
                B,
                oracle_D,
            )


            diou = abs(
                iou_A
                -
                iou_B
            )

            dviol = abs(
                viol_A
                -
                viol_B
            )

            dover = abs(
                over_A
                -
                over_B
            )


            row[
                "sum_abs_iou_difference"
            ] += diou

            row[
                "sum_abs_violation_difference"
            ] += dviol

            row[
                "sum_abs_overmask_difference"
            ] += dover


            if diou > 0:
                row[
                    "iou_changed_states"
                ] += 1

            if dviol > 0:
                row[
                    "violation_changed_states"
                ] += 1

            if dover > 0:
                row[
                    "overmask_changed_states"
                ] += 1


print()
print("=" * 78)
print("RESULTS")
print("=" * 78)


for pair, row in stats.items():

    states = row[
        "states"
    ]

    binary_fraction = (
        row[
            "binary_candidate_changed_cells"
        ]
        /
        row[
            "binary_candidate_total_cells"
        ]
    )

    print()
    print(
        f"gamma {pair[0]} vs {pair[1]}"
    )

    print(
        "binary changed fraction =",
        binary_fraction
    )

    print(
        "states with IoU change =",
        row[
            "iou_changed_states"
        ],
        "/",
        states
    )

    print(
        "states with violation change =",
        row[
            "violation_changed_states"
        ],
        "/",
        states
    )

    print(
        "states with overmask change =",
        row[
            "overmask_changed_states"
        ],
        "/",
        states
    )

    print(
        "mean |ΔIoU| =",
        row[
            "sum_abs_iou_difference"
        ]
        /
        states
    )

    print(
        "mean |Δviolation| =",
        row[
            "sum_abs_violation_difference"
        ]
        /
        states
    )

    print(
        "mean |Δovermask| =",
        row[
            "sum_abs_overmask_difference"
        ]
        /
        states
    )


print()
print("=" * 78)
print("AUTOMATIC VERDICT")
print("=" * 78)


for pair, row in stats.items():

    binary_changed = (
        row[
            "binary_candidate_changed_cells"
        ]
        >
        0
    )

    metric_changed = (
        row[
            "iou_changed_states"
        ]
        >
        0
        or
        row[
            "violation_changed_states"
        ]
        >
        0
        or
        row[
            "overmask_changed_states"
        ]
        >
        0
    )

    if (
        binary_changed
        and
        not metric_changed
    ):

        print(
            pair,
            "=> ERROR: candidate binary masks change "
            "but all oracle metrics remain invariant."
        )

    elif (
        binary_changed
        and
        metric_changed
    ):

        print(
            pair,
            "=> METRIC PIPELINE RESPONDS CORRECTLY."
        )

    else:

        print(
            pair,
            "=> no binary gamma effect."
        )

print("=" * 78)
