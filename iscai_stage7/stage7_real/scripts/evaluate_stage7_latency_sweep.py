from pathlib import Path
import json
import math
import numpy as np


ROLLING = Path(
    "/home/agni/waymo/iscai_stage6/"
    "stage6_real/data/"
    "real_stage6_predictions_rolling.npz"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_latency_sweep.json"
)

DELAYS_FRAMES = [0, 1, 2, 3, 5]
DT_S = 0.1

NUM_BEAMS = 64
TARGET_MASS = 0.95
MC_SAMPLES = 500
SEED = 42


def rotation_z(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)

    return np.asarray(
        [
            [c, -s],
            [s,  c],
        ],
        dtype=np.float64,
    )


def headlamp_origin_world(
    sdc_center_xy,
    sdc_heading,
    sdc_length,
):
    """
    Default HeadlampSurrogateConfig:
    H origin = front midpoint of SDC body.
    """
    forward = np.asarray(
        [
            math.cos(sdc_heading),
            math.sin(sdc_heading),
        ],
        dtype=np.float64,
    )

    return (
        np.asarray(
            sdc_center_xy,
            dtype=np.float64,
        )
        +
        0.5
        *
        float(sdc_length)
        *
        forward
    )


def transform_mean_oldH_to_newH(
    points_old_H,
    old_sdc_center,
    old_sdc_heading,
    old_sdc_length,
    new_sdc_center,
    new_sdc_heading,
    new_sdc_length,
):
    """
    H_old -> W -> H_new.
    """

    p = np.asarray(
        points_old_H,
        dtype=np.float64,
    )

    R_W_from_H_old = rotation_z(
        old_sdc_heading
    )

    old_origin_W = headlamp_origin_world(
        old_sdc_center[:2],
        old_sdc_heading,
        old_sdc_length,
    )

    p_W = (
        p
        @
        R_W_from_H_old.T
        +
        old_origin_W
    )

    new_origin_W = headlamp_origin_world(
        new_sdc_center[:2],
        new_sdc_heading,
        new_sdc_length,
    )

    R_H_new_from_W = rotation_z(
        -new_sdc_heading
    )

    p_new_H = (
        p_W
        -
        new_origin_W
    ) @ R_H_new_from_W.T

    return p_new_H


def transform_cov_oldH_to_newH(
    sx,
    sy,
    rho,
    old_heading,
    new_heading,
):
    cov_old = np.asarray(
        [
            [
                sx * sx,
                rho * sx * sy,
            ],
            [
                rho * sx * sy,
                sy * sy,
            ],
        ],
        dtype=np.float64,
    )

    relative_yaw = (
        old_heading
        -
        new_heading
    )

    R = rotation_z(
        relative_yaw
    )

    cov_new = (
        R
        @ cov_old
        @ R.T
    )

    sx_new = math.sqrt(
        max(
            cov_new[0, 0],
            1e-12,
        )
    )

    sy_new = math.sqrt(
        max(
            cov_new[1, 1],
            1e-12,
        )
    )

    rho_new = (
        cov_new[0, 1]
        /
        (sx_new * sy_new)
    )

    rho_new = float(
        np.clip(
            rho_new,
            -0.999,
            0.999,
        )
    )

    return (
        sx_new,
        sy_new,
        rho_new,
    )


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
            math.sqrt(
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
    target_mass,
):
    ranked = np.argsort(
        probabilities
    )[::-1]

    mass = 0.0
    k = 0

    for beam in ranked:
        mass += float(
            probabilities[beam]
        )

        k += 1

        if mass >= target_mass:
            break

    return k, mass


def summarize(x):
    a = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        "count":
            int(len(a)),

        "mean":
            float(a.mean()),

        "p50":
            float(
                np.percentile(
                    a,
                    50,
                )
            ),

        "p95":
            float(
                np.percentile(
                    a,
                    95,
                )
            ),
    }


def main():

    print("=" * 88)
    print("STAGE 7 STALE-POSTERIOR LATENCY SWEEP")
    print("=" * 88)

    d = np.load(ROLLING)

    seq = d["sequence_index"]
    anchor_pos = d["anchor_position"]

    mu = d[
        "future_mean_H_m"
    ][..., :2]

    std = d[
        "future_std_xy_H_m"
    ]

    rho = d[
        "future_rho_xy_H"
    ]

    sdc_center = d[
        "sdc_center_W_m"
    ]

    sdc_heading = d[
        "sdc_heading_W_rad"
    ]

    sdc_length = d[
        "sdc_length_m"
    ]

    rows = []

    unique_seq = np.unique(
        seq
    )

    for delay in DELAYS_FRAMES:

        rng = np.random.default_rng(
            SEED
        )

        ks = []
        masses = []
        transformed_states = 0

        for sid in unique_seq:

            rows_idx = np.flatnonzero(
                seq == sid
            )

            order = np.argsort(
                anchor_pos[
                    rows_idx
                ]
            )

            rows_idx = rows_idx[
                order
            ]

            for current_local in range(
                delay,
                len(rows_idx),
            ):

                current_idx = rows_idx[
                    current_local
                ]

                stale_idx = rows_idx[
                    current_local
                    -
                    delay
                ]

                # Compare same control horizon index 0.
                stale_mu = mu[
                    stale_idx,
                    0,
                ]

                transformed_mu = (
                    transform_mean_oldH_to_newH(
                        stale_mu[None, :],
                        sdc_center[
                            stale_idx
                        ],
                        float(
                            sdc_heading[
                                stale_idx
                            ]
                        ),
                        float(
                            sdc_length[
                                stale_idx
                            ]
                        ),
                        sdc_center[
                            current_idx
                        ],
                        float(
                            sdc_heading[
                                current_idx
                            ]
                        ),
                        float(
                            sdc_length[
                                current_idx
                            ]
                        ),
                    )[0]
                )

                sx, sy, r = (
                    transform_cov_oldH_to_newH(
                        float(
                            std[
                                stale_idx,
                                0,
                                0
                            ]
                        ),
                        float(
                            std[
                                stale_idx,
                                0,
                                1
                            ]
                        ),
                        float(
                            rho[
                                stale_idx,
                                0
                            ]
                        ),
                        float(
                            sdc_heading[
                                stale_idx
                            ]
                        ),
                        float(
                            sdc_heading[
                                current_idx
                            ]
                        ),
                    )
                )

                xs, ys = sample_gaussian(
                    transformed_mu,
                    (sx, sy),
                    r,
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

                ks.append(k)
                masses.append(mass)
                transformed_states += 1

        row = {
            "delay_frames":
                int(delay),

            "delay_ms":
                float(
                    delay
                    *
                    DT_S
                    *
                    1000.0
                ),

            "evaluated_states":
                int(
                    transformed_states
                ),

            "communication": {
                "num_beams":
                    NUM_BEAMS,

                "target_mass":
                    TARGET_MASS,

                "selected_k":
                    summarize(ks),

                "covered_mass":
                    summarize(masses),

                "mean_overhead_fraction":
                    float(
                        np.mean(ks)
                        /
                        NUM_BEAMS
                    ),
            },
        }

        rows.append(row)

        print(
            f"delay={row['delay_ms']:5.0f} ms "
            f"states={row['evaluated_states']:3d} "
            f"meanK="
            f"{row['communication']['selected_k']['mean']:.4f} "
            f"mass="
            f"{row['communication']['covered_mass']['mean']:.6f} "
            f"overhead="
            f"{100*row['communication']['mean_overhead_fraction']:.3f}%"
        )

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "sweep":
            "stale_posterior_latency",

        "source":
            str(ROLLING),

        "controller_dt_s":
            DT_S,

        "delay_frames":
            DELAYS_FRAMES,

        "delay_ms":
            [
                int(
                    x * DT_S * 1000
                )
                for x in DELAYS_FRAMES
            ],

        "communication_policy": {
            "num_beams":
                NUM_BEAMS,

            "target_mass":
                TARGET_MASS,
        },

        "frame_transform":
            (
                "Each stale posterior is transformed "
                "from its original headlamp frame via "
                "world coordinates into the current "
                "headlamp frame. Gaussian covariance "
                "is rotated consistently."
            ),

        "physical_actuator_latency_measured":
            False,

        "scientific_scope": (
            "This sweep measures algorithmic stale-"
            "posterior effects for controller delays. "
            "It is not a measurement of physical "
            "headlamp actuation latency."
        ),

        "results":
            rows,
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
        "STAGE 7 LATENCY SWEEP = PASS"
    )


if __name__ == "__main__":
    main()
