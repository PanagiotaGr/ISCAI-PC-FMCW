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

m = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    m
)


pred = np.load(
    m.PRED_FILE
)

oracle = np.load(
    m.ORACLE_FILE
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

heading = pred[
    "current_heading_H_rad"
]


current = oracle[
    "current_center_H_m"
]


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
    (np.abs(current_angle) <= 25)
    &
    (current_range <= 180)
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
    (np.abs(pred_angle) <= 25)
    &
    (pred_range <= 180)
)

entry = (
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
    entry
)[0]


rng = np.random.default_rng(
    12345
)


all_nonzero = []

threshold_counts = {
    g: []
    for g in m.GAMMAS
}

candidate_differences = {
    (0.10, 0.25): [],
    (0.25, 0.50): [],
    (0.10, 0.50): [],
}


print("=" * 78)
print("STAGE 6 GAMMA / OCCUPANCY DIAGNOSTIC")
print("=" * 78)

print(
    "actors checked =",
    len(indices)
)


# First 30 relevant actors × first 3 horizons
for actor in indices[:30]:

    cls = str(
        classes[
            actor
        ]
    )

    config = m.CLASS_CONFIGS[
        cls
    ]

    for t in range(3):

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
        )


        nz = occupancy[
            occupancy > 0
        ]

        if len(nz):

            all_nonzero.extend(
                nz.tolist()
            )


        deterministic_L = (
            m.intensity_map(
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
        )

        deterministic_D = (
            1.0
            -
            deterministic_L
        )


        candidates = {}

        max_dimming = (
            1.0
            -
            float(
                config[
                    "floor"
                ]
            )
        )


        for gamma in m.GAMMAS:

            mask = (
                occupancy
                >=
                gamma
            )

            threshold_counts[
                gamma
            ].append(
                int(
                    mask.sum()
                )
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
                max_dimming
                *
                occupancy[
                    mask
                ],
            )

            D = np.minimum(
                D,
                max_dimming,
            )

            candidates[
                gamma
            ] = D


        for pair in candidate_differences:

            a, b = pair

            candidate_differences[
                pair
            ].append(
                float(
                    np.mean(
                        np.abs(
                            candidates[a]
                            -
                            candidates[b]
                        )
                    )
                )
            )


x = np.asarray(
    all_nonzero,
    dtype=np.float64,
)


print()
print("===== OCCUPANCY DISTRIBUTION =====")

print(
    "non-zero cells =",
    len(x)
)

if len(x):

    for q in [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]:

        print(
            f"p{q:3d} =",
            float(
                np.percentile(
                    x,
                    q
                )
            )
        )


print()
print("===== CELLS ABOVE GAMMA =====")

for gamma in m.GAMMAS:

    v = np.asarray(
        threshold_counts[
            gamma
        ]
    )

    print(
        f"gamma={gamma}",
        "mean cells =",
        float(
            v.mean()
        ),
        "median =",
        float(
            np.median(v)
        ),
        "min/max =",
        (
            int(v.min()),
            int(v.max()),
        )
    )


print()
print("===== CANDIDATE MAP DIFFERENCES =====")

for pair, values in candidate_differences.items():

    v = np.asarray(
        values
    )

    print(
        f"{pair[0]} vs {pair[1]}:",
        "mean abs difference =",
        float(
            v.mean()
        ),
        "max =",
        float(
            v.max()
        )
    )


print()
print("=" * 78)

if len(x):

    middle_probability = np.mean(
        (x > 0.10)
        &
        (x < 0.50)
    )

    print(
        "fraction of non-zero occupancy "
        "strictly between 0.10 and 0.50 =",
        float(
            middle_probability
        )
    )

print("=" * 78)
