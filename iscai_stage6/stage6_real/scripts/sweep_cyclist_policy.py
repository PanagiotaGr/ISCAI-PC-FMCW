import importlib.util
from pathlib import Path
import numpy as np


SCRIPT = Path(
    "stage6_real/scripts/evaluate_stage6_streaming.py"
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
# Same causal evaluation cohort
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
    (current[:,0] > 0.0)
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
    (mu[...,0] > 0.0)
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


cyclist_indices = np.where(
    (classes == "CYCLIST")
    &
    (
        current_fov
        |
        predicted_entry
    )
)[0]


print("=" * 88)
print("STAGE 6 CYCLIST POLICY VALIDATION SWEEP")
print("=" * 88)

print(
    "relevant cyclists =",
    len(cyclist_indices)
)


FLOORS = [
    0.25,
    0.30,
    0.35,
    0.40,
]

MARGINS = [
    1.25,
    1.50,
]

GAMMAS = [
    0.10,
    0.25,
    0.50,
]


results = {}


for floor in FLOORS:

    for margin in MARGINS:

        for gamma in GAMMAS:

            key = (
                floor,
                margin,
                gamma,
            )

            results[key] = {
                "visibility": [],
                "violation": [],
                "false_dimming": [],
                "overmask": [],
                "retention": [],
            }


rng = np.random.default_rng(
    m.SEED
)


for pos, actor in enumerate(
    cyclist_indices,
    start=1,
):

    for t in range(10):

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


        # Generate one occupancy per lateral margin.
        occupancy_by_margin = {}

        for margin in MARGINS:

            occupancy_by_margin[
                margin
            ] = m.probabilistic_occupancy(
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
                margin,
                rng,
            )


        for floor in FLOORS:

            max_dimming = (
                1.0
                -
                floor
            )

            for margin in MARGINS:

                occupancy = (
                    occupancy_by_margin[
                        margin
                    ]
                )

                for gamma in GAMMAS:

                    mask = (
                        occupancy
                        >=
                        gamma
                    )

                    D = (
                        deterministic_D.copy()
                    )

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


                    row = results[
                        (
                            floor,
                            margin,
                            gamma,
                        )
                    ]

                    row[
                        "visibility"
                    ].append(
                        m.oracle_region_visibility(
                            D,
                            oracle_D,
                        )
                    )

                    row[
                        "violation"
                    ].append(
                        m.shadow_violation(
                            D,
                            oracle_D,
                        )
                    )

                    row[
                        "false_dimming"
                    ].append(
                        m.false_dimming(
                            D,
                            oracle_D,
                        )
                    )

                    row[
                        "overmask"
                    ].append(
                        m.overmask(
                            D,
                            oracle_D,
                        )
                    )

                    row[
                        "retention"
                    ].append(
                        m.road_retention(
                            D
                        )
                    )


    if (
        pos % 10 == 0
        or
        pos == len(
            cyclist_indices
        )
    ):

        print(
            f"cyclists="
            f"{pos}/"
            f"{len(cyclist_indices)}",
            flush=True,
        )


print()
print("=" * 88)
print("RESULTS")
print("=" * 88)

summary = []

for key, row in results.items():

    floor, margin, gamma = key

    visibility = float(
        np.mean(
            row["visibility"]
        )
    )

    violation = float(
        np.mean(
            row["violation"]
        )
    )

    false_dimming = float(
        np.mean(
            row["false_dimming"]
        )
    )

    overmask = float(
        np.mean(
            row["overmask"]
        )
    )

    retention = float(
        np.mean(
            row["retention"]
        )
    )

    feasible = (
        visibility
        >=
        0.310615
    )

    summary.append(
        (
            feasible,
            violation,
            false_dimming,
            -visibility,
            floor,
            margin,
            gamma,
            visibility,
            overmask,
            retention,
        )
    )

    print(
        f"floor={floor:.2f} "
        f"margin={margin:.2f} "
        f"gamma={gamma:.2f} "
        f"visibility={visibility:.6f} "
        f"violation={100*violation:.3f}% "
        f"false_dim={100*false_dimming:.3f}% "
        f"overmask={100*overmask:.3f}% "
        f"retention={retention:.6f} "
        f"feasible={feasible}"
    )


feasible_rows = [
    x
    for x in summary
    if x[0]
]

print()
print("=" * 88)
print("FEASIBLE OPERATING POINTS")
print("=" * 88)

if not feasible_rows:

    print(
        "NONE: no tested configuration "
        "reaches reactive cyclist visibility."
    )

else:

    feasible_rows.sort(
        key=lambda x: (
            x[1],
            x[2],
            x[3],
        )
    )

    for row in feasible_rows[:10]:

        (
            _,
            violation,
            false_dimming,
            neg_visibility,
            floor,
            margin,
            gamma,
            visibility,
            overmask,
            retention,
        ) = row

        print(
            f"floor={floor:.2f} "
            f"margin={margin:.2f} "
            f"gamma={gamma:.2f} "
            f"visibility={visibility:.6f} "
            f"violation={100*violation:.3f}% "
            f"false_dim={100*false_dimming:.3f}% "
            f"overmask={100*overmask:.3f}% "
            f"retention={retention:.6f}"
        )
