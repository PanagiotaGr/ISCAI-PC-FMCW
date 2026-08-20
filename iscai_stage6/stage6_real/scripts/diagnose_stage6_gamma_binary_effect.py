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


# ============================================================
# Same causal cohort as final evaluator
# ============================================================

current_angle = np.degrees(
    np.arctan2(
        current[:, 1],
        current[:, 0],
    )
)

current_range = np.hypot(
    current[:, 0],
    current[:, 1],
)

current_fov = (
    (current[:, 0] > 0.0)
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

future_fov = (
    (mu[..., 0] > 0.0)
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


rng = np.random.default_rng(2026)


gamma_pairs = [
    (0.10, 0.25),
    (0.25, 0.50),
    (0.10, 0.50),
]


stats = {
    pair: {
        "occupancy_mask_diff": [],
        "candidate_continuous_diff": [],
        "candidate_binary_diff": [],
        "extra_cells_already_deterministic": [],
    }
    for pair in gamma_pairs
}


print("=" * 78)
print("STAGE 6 GAMMA BINARY-EFFECT DIAGNOSTIC")
print("=" * 78)

print("relevant actors =", len(indices))


# Enough states for diagnosis without another full 77 s evaluation.
for actor in indices[:50]:

    cls = str(classes[actor])

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

    for t in range(5):

        deterministic_L = m.intensity_map(
            mu[actor, t, :2],
            float(length[actor]),
            float(width[actor]),
            float(heading[actor]),
        )

        deterministic_D = (
            1.0
            -
            deterministic_L
        )

        deterministic_binary = (
            deterministic_D
            >= 0.5
        )


        occupancy = m.probabilistic_occupancy(
            mu[actor, t, :2],
            std[actor, t],
            rho[actor, t],
            float(length[actor]),
            float(width[actor]),
            float(heading[actor]),
            float(
                config[
                    "lateral_margin_scale"
                ]
            ),
            rng,
        )


        candidates = {}
        masks = {}

        for gamma in m.GAMMAS:

            mask = (
                occupancy
                >=
                gamma
            )

            D = deterministic_D.copy()

            D[mask] = np.maximum(
                D[mask],
                max_dimming,
            )

            D = np.minimum(
                D,
                max_dimming,
            )

            masks[gamma] = mask
            candidates[gamma] = D


        for pair in gamma_pairs:

            a, b = pair

            ma = masks[a]
            mb = masks[b]

            Da = candidates[a]
            Db = candidates[b]

            binary_a = (
                Da >= 0.5
            )

            binary_b = (
                Db >= 0.5
            )

            mask_diff = (
                ma != mb
            )

            binary_diff = (
                binary_a != binary_b
            )

            # Cells included by lower gamma but not by higher gamma.
            lower_only = (
                ma
                &
                (~mb)
            )

            if np.any(lower_only):

                already = np.mean(
                    deterministic_binary[
                        lower_only
                    ]
                )

            else:
                already = 0.0


            stats[pair][
                "occupancy_mask_diff"
            ].append(
                float(
                    mask_diff.mean()
                )
            )

            stats[pair][
                "candidate_continuous_diff"
            ].append(
                float(
                    np.mean(
                        np.abs(
                            Da
                            -
                            Db
                        )
                    )
                )
            )

            stats[pair][
                "candidate_binary_diff"
            ].append(
                float(
                    binary_diff.mean()
                )
            )

            stats[pair][
                "extra_cells_already_deterministic"
            ].append(
                float(already)
            )


print()
print("===== RESULTS =====")

for pair in gamma_pairs:

    print()
    print(
        f"gamma {pair[0]} vs {pair[1]}"
    )

    for key, values in stats[pair].items():

        v = np.asarray(
            values,
            dtype=np.float64,
        )

        print(
            f"  {key:36s}"
            f" mean={v.mean():.8f}"
            f" max={v.max():.8f}"
        )


print()
print("=" * 78)
print("INTERPRETATION")
print("=" * 78)

for pair in gamma_pairs:

    binary = np.asarray(
        stats[pair][
            "candidate_binary_diff"
        ]
    )

    mask = np.asarray(
        stats[pair][
            "occupancy_mask_diff"
        ]
    )

    print()

    if (
        mask.mean() > 0
        and
        binary.mean() == 0
    ):
        print(
            pair,
            "=> gamma changes probabilistic membership, "
            "but NOT the >=0.5 binary ADB mask."
        )

    elif binary.mean() > 0:
        print(
            pair,
            "=> gamma also changes the binary ADB mask."
        )

    else:
        print(
            pair,
            "=> no measurable gamma effect in sampled states."
        )

print("=" * 78)
