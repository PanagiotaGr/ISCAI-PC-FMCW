from pathlib import Path
import math
import numpy as np


PRED = Path(
    "stage6_real/data/"
    "real_stage6_predictions_balanced.npz"
)

ORACLE = Path(
    "stage6_real/data/"
    "real_stage6_oracle_future_boxes.npz"
)


# ============================================================
# EXACT PART-A NUMERICAL GRID
# ============================================================

THETA_DEG = np.linspace(
    -25.0,
    25.0,
    501,
)

RANGE_M = np.linspace(
    0.0,
    180.0,
    361,
)

RADIAL_SAFETY_MARGIN_M = 4.0
LATERAL_SAFETY_MARGIN_M = 1.0
TRANSITION_LENGTH_M = 20.0


def box_corners(
    center_xy,
    length,
    width,
    heading,
):

    local = np.asarray(
        [
            [+length/2, +width/2],
            [+length/2, -width/2],
            [-length/2, +width/2],
            [-length/2, -width/2],
        ],
        dtype=np.float64,
    )

    c = math.cos(heading)
    s = math.sin(heading)

    R = np.asarray(
        [
            [c, -s],
            [s,  c],
        ]
    )

    corners = local @ R.T

    corners[:, 0] += center_xy[0]
    corners[:, 1] += center_xy[1]

    return corners


def intensity_map(
    center_xy,
    length,
    width,
    heading,
):

    corners = box_corners(
        center_xy,
        length,
        width,
        heading,
    )

    angles = np.unwrap(
        np.arctan2(
            corners[:, 1],
            corners[:, 0],
        )
    )

    center_range = max(
        float(
            np.hypot(
                center_xy[0],
                center_xy[1],
            )
        ),
        1e-6,
    )

    epsilon = math.atan(
        LATERAL_SAFETY_MARGIN_M
        /
        center_range
    )

    theta_min = (
        float(angles.min())
        -
        epsilon
    )

    theta_max = (
        float(angles.max())
        +
        epsilon
    )

    corner_ranges = np.hypot(
        corners[:, 0],
        corners[:, 1],
    )

    target_range = float(
        corner_ranges.max()
    )

    r_min = (
        target_range
        +
        RADIAL_SAFETY_MARGIN_M
    )

    r_max = (
        r_min
        +
        TRANSITION_LENGTH_M
    )

    theta_center = (
        0.5
        *
        (
            theta_min
            +
            theta_max
        )
    )

    theta_half = (
        0.5
        *
        (
            theta_max
            -
            theta_min
        )
    )

    T, R = np.meshgrid(
        np.deg2rad(
            THETA_DEG
        ),
        RANGE_M,
        indexing="ij",
    )

    angle_difference = np.angle(
        np.exp(
            1j
            *
            (
                T
                -
                theta_center
            )
        )
    )

    inside_angle = (
        np.abs(
            angle_difference
        )
        <=
        theta_half
    )

    L = np.ones(
        T.shape,
        dtype=np.float32,
    )

    full_shadow = (
        inside_angle
        &
        (R <= r_min)
    )

    transition = (
        inside_angle
        &
        (R > r_min)
        &
        (R < r_max)
    )

    L[
        full_shadow
    ] = 0.0

    if np.any(
        transition
    ):

        u = (
            R[transition]
            -
            r_min
        ) / (
            r_max
            -
            r_min
        )

        L[
            transition
        ] = (
            0.5
            *
            (
                1.0
                -
                np.cos(
                    np.pi
                    *
                    u
                )
            )
        )

    return (
        L,
        {
            "theta_min_deg":
                math.degrees(
                    theta_min
                ),

            "theta_max_deg":
                math.degrees(
                    theta_max
                ),

            "r_min_m":
                r_min,

            "r_max_m":
                r_max,
        },
    )


def main():

    print("=" * 76)
    print(
        "STAGE 6 PART-A ADB GRID SMOKE TEST"
    )
    print("=" * 76)

    pred = np.load(
        PRED
    )

    oracle = np.load(
        ORACLE
    )

    actor = 0
    horizon = 0

    print(
        "actor class =",
        pred[
            "actor_class"
        ][actor]
    )

    print(
        "theta samples =",
        len(
            THETA_DEG
        )
    )

    print(
        "theta range =",
        (
            THETA_DEG[0],
            THETA_DEG[-1],
        )
    )

    print(
        "theta step =",
        THETA_DEG[1]
        -
        THETA_DEG[0],
        "deg"
    )

    print(
        "range samples =",
        len(
            RANGE_M
        )
    )

    print(
        "range range =",
        (
            RANGE_M[0],
            RANGE_M[-1],
        )
    )

    print(
        "range step =",
        RANGE_M[1]
        -
        RANGE_M[0],
        "m"
    )


    # ========================================================
    # Deterministic Stage-4 prediction
    # ========================================================

    pred_L, pred_info = intensity_map(
        center_xy=
            pred[
                "future_mean_H_m"
            ][
                actor,
                horizon,
                :2
            ],

        length=float(
            pred[
                "length_m"
            ][actor]
        ),

        width=float(
            pred[
                "width_m"
            ][actor]
        ),

        heading=float(
            pred[
                "current_heading_H_rad"
            ][actor]
        ),
    )


    # ========================================================
    # Oracle real future WOMD box
    # ========================================================

    oracle_L, oracle_info = intensity_map(
        center_xy=
            oracle[
                "future_center_H_m"
            ][
                actor,
                horizon,
                :2
            ],

        length=float(
            oracle[
                "future_lwh_m"
            ][
                actor,
                horizon,
                0
            ]
        ),

        width=float(
            oracle[
                "future_lwh_m"
            ][
                actor,
                horizon,
                1
            ]
        ),

        heading=float(
            oracle[
                "future_heading_H_rad"
            ][
                actor,
                horizon
            ]
        ),
    )


    assert (
        pred_L.shape
        ==
        (501, 361)
    )

    assert (
        oracle_L.shape
        ==
        (501, 361)
    )

    assert np.isfinite(
        pred_L
    ).all()

    assert np.isfinite(
        oracle_L
    ).all()

    assert (
        pred_L.min() >= 0
        and
        pred_L.max() <= 1
    )

    assert (
        oracle_L.min() >= 0
        and
        oracle_L.max() <= 1
    )


    pred_D = (
        1.0
        -
        pred_L
    )

    oracle_D = (
        1.0
        -
        oracle_L
    )


    print()
    print(
        "deterministic map shape =",
        pred_L.shape
    )

    print(
        "oracle map shape =",
        oracle_L.shape
    )

    print()
    print(
        "deterministic geometry =",
        pred_info
    )

    print(
        "oracle geometry =",
        oracle_info
    )

    print()
    print(
        "deterministic mean dimming =",
        float(
            pred_D.mean()
        )
    )

    print(
        "oracle mean dimming =",
        float(
            oracle_D.mean()
        )
    )

    print(
        "deterministic max dimming =",
        float(
            pred_D.max()
        )
    )

    print(
        "oracle max dimming =",
        float(
            oracle_D.max()
        )
    )

    print()
    print("=" * 76)
    print(
        "PART-A GRID SMOKE TEST PASSED"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
