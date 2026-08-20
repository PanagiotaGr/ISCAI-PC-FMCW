from pathlib import Path
import json
import math
import time

import numpy as np


PRED_FILE = Path(
    "stage6_real/data/"
    "real_stage6_predictions_balanced.npz"
)

ORACLE_FILE = Path(
    "stage6_real/data/"
    "real_stage6_oracle_future_boxes.npz"
)

OCC_FILE = Path(
    "stage6_real/data/"
    "real_stage6_probabilistic_occupancy.npz"
)

OUTPUT = Path(
    "stage6_real/data/"
    "real_stage6_adb_baseline_masks.npz"
)

REPORT = Path(
    "stage6_real/reports/"
    "real_stage6_adb_baseline_masks_summary.json"
)


# Part-A physical/evaluation parameters
RADIAL_MARGIN_M = 4.0
TRANSITION_LENGTH_M = 20.0
LATERAL_MARGIN_M = 1.0


def oriented_corners(
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
        ],
        dtype=np.float64,
    )

    corners = local @ R.T

    corners[:, 0] += center_xy[0]
    corners[:, 1] += center_xy[1]

    return corners


def angular_support(corners):

    angles = np.unwrap(
        np.arctan2(
            corners[:, 1],
            corners[:, 0],
        )
    )

    ranges = np.hypot(
        corners[:, 0],
        corners[:, 1],
    )

    return (
        float(angles.min()),
        float(angles.max()),
        float(ranges.max()),
    )


def rasterize_intensity(
    center_xy,
    length,
    width,
    heading,
    az_centers,
    range_centers,
):
    """
    Part-A style:
      full shadow to target range + 4m
      raised-cosine over next 20m
      full illumination outside angular window

    Lateral margin is converted into an angular margin
    using atan(margin/range), as in Part-A.
    """

    corners = oriented_corners(
        center_xy,
        length,
        width,
        heading,
    )

    a0, a1, target_range = angular_support(
        corners
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
        LATERAL_MARGIN_M
        /
        center_range
    )

    a0 -= epsilon
    a1 += epsilon

    r_min = (
        target_range
        +
        RADIAL_MARGIN_M
    )

    r_max = (
        r_min
        +
        TRANSITION_LENGTH_M
    )

    A, R = np.meshgrid(
        az_centers,
        range_centers,
        indexing="ij",
    )

    # handle local wrap around +/-pi
    da = np.angle(
        np.exp(
            1j * (
                A
                -
                0.5 * (a0+a1)
            )
        )
    )

    half_width = (
        0.5
        *
        (a1-a0)
    )

    inside_angle = (
        np.abs(da)
        <=
        half_width
    )

    L = np.ones(
        A.shape,
        dtype=np.float32,
    )

    full = (
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

    L[full] = 0.0

    if np.any(transition):

        u = (
            R[transition]
            -
            r_min
        ) / (
            r_max
            -
            r_min
        )

        L[transition] = (
            0.5
            *
            (
                1.0
                -
                np.cos(
                    np.pi * u
                )
            )
        )

    return L


def main():

    print("=" * 78)
    print(
        "STAGE 6 REAL ADB BASELINE MASKS"
    )
    print("=" * 78)

    pred = np.load(
        PRED_FILE
    )

    oracle = np.load(
        ORACLE_FILE
    )

    occ = np.load(
        OCC_FILE
    )

    classes = pred[
        "actor_class"
    ]

    pred_mean = pred[
        "future_mean_H_m"
    ]

    pred_heading = pred[
        "current_heading_H_rad"
    ]

    pred_length = pred[
        "length_m"
    ]

    pred_width = pred[
        "width_m"
    ]

    current_center = oracle[
        "current_center_H_m"
    ]

    current_heading = oracle[
        "current_heading_H_rad"
    ]

    current_lwh = oracle[
        "current_lwh_m"
    ]

    future_center = oracle[
        "future_center_H_m"
    ]

    future_heading = oracle[
        "future_heading_H_rad"
    ]

    future_lwh = oracle[
        "future_lwh_m"
    ]

    az_edges = occ[
        "azimuth_edges_rad"
    ]

    range_edges = occ[
        "range_edges_m"
    ]

    az_centers = (
        0.5
        *
        (
            az_edges[:-1]
            +
            az_edges[1:]
        )
    )

    range_centers = (
        0.5
        *
        (
            range_edges[:-1]
            +
            range_edges[1:]
        )
    )

    actors = len(classes)
    horizons = pred_mean.shape[1]

    shape = (
        actors,
        horizons,
        len(az_centers),
        len(range_centers),
    )

    print(
        "actors =",
        actors
    )

    print(
        "mask shape =",
        shape
    )

    reactive = np.empty(
        shape,
        dtype=np.float32,
    )

    deterministic = np.empty(
        shape,
        dtype=np.float32,
    )

    oracle_mask = np.empty(
        shape,
        dtype=np.float32,
    )

    t0 = time.perf_counter()

    for i in range(actors):

        # ----------------------------------------------------
        # Reactive ADB:
        # current real box held fixed across all future horizons.
        # ----------------------------------------------------

        reactive_single = rasterize_intensity(
            center_xy=
                current_center[i, :2],

            length=float(
                current_lwh[i, 0]
            ),

            width=float(
                current_lwh[i, 1]
            ),

            heading=float(
                current_heading[i]
            ),

            az_centers=
                az_centers,

            range_centers=
                range_centers,
        )

        for t in range(horizons):

            reactive[
                i,
                t
            ] = reactive_single


            # ------------------------------------------------
            # Deterministic predictive:
            # Stage-4 Gaussian mean only; no uncertainty.
            # Causal current dimensions/heading.
            # ------------------------------------------------

            deterministic[
                i,
                t
            ] = rasterize_intensity(
                center_xy=
                    pred_mean[
                        i,
                        t,
                        :2
                    ],

                length=float(
                    pred_length[i]
                ),

                width=float(
                    pred_width[i]
                ),

                heading=float(
                    pred_heading[i]
                ),

                az_centers=
                    az_centers,

                range_centers=
                    range_centers,
            )


            # ------------------------------------------------
            # Oracle future:
            # real future WOMD box.
            # Evaluation only.
            # ------------------------------------------------

            oracle_mask[
                i,
                t
            ] = rasterize_intensity(
                center_xy=
                    future_center[
                        i,
                        t,
                        :2
                    ],

                length=float(
                    future_lwh[
                        i,
                        t,
                        0
                    ]
                ),

                width=float(
                    future_lwh[
                        i,
                        t,
                        1
                    ]
                ),

                heading=float(
                    future_heading[
                        i,
                        t
                    ]
                ),

                az_centers=
                    az_centers,

                range_centers=
                    range_centers,
            )


        if (
            (i+1) % 50 == 0
            or
            i+1 == actors
        ):

            print(
                f"actors="
                f"{i+1}/{actors}",
                flush=True,
            )


    elapsed = (
        time.perf_counter()
        -
        t0
    )


    # Convert illumination L to dimming/shadow mask:
    # 0 = full illumination
    # 1 = full dimming
    reactive_dimming = (
        1.0
        -
        reactive
    )

    deterministic_dimming = (
        1.0
        -
        deterministic
    )

    oracle_dimming = (
        1.0
        -
        oracle_mask
    )


    for name, arr in [
        (
            "reactive",
            reactive_dimming
        ),
        (
            "deterministic",
            deterministic_dimming
        ),
        (
            "oracle",
            oracle_dimming
        ),
    ]:

        if not np.isfinite(
            arr
        ).all():
            raise RuntimeError(
                f"Non-finite "
                f"{name} mask"
            )

        if (
            arr.min() < 0.0
            or
            arr.max() > 1.0
        ):
            raise RuntimeError(
                f"{name} outside [0,1]"
            )


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT,

        actor_class=
            classes,

        azimuth_edges_rad=
            az_edges,

        range_edges_m=
            range_edges,

        reactive_dimming=
            reactive_dimming,

        deterministic_predictive_dimming=
            deterministic_dimming,

        oracle_future_dimming=
            oracle_dimming,
    )


    def class_mean(
        array,
        cls,
    ):
        mask = (
            classes == cls
        )

        return float(
            array[
                mask
            ].mean()
        )


    summary = {}

    for cls in [
        "VEHICLE",
        "PEDESTRIAN",
        "CYCLIST",
    ]:

        summary[
            cls
        ] = {
            "reactive_mean_dimming":
                class_mean(
                    reactive_dimming,
                    cls,
                ),

            "deterministic_mean_dimming":
                class_mean(
                    deterministic_dimming,
                    cls,
                ),

            "oracle_mean_dimming":
                class_mean(
                    oracle_dimming,
                    cls,
                ),
        }


    report = {
        "status":
            "PASS",

        "data_source":
            "real_WOMD_balanced",

        "actors":
            int(actors),

        "future_steps":
            int(horizons),

        "future_used_as_input":
            False,

        "part_a_parameters": {
            "radial_safety_margin_m":
                RADIAL_MARGIN_M,

            "transition_length_m":
                TRANSITION_LENGTH_M,

            "lateral_safety_margin_m":
                LATERAL_MARGIN_M,

            "transition":
                "raised_cosine",
        },

        "baselines": {
            "reactive":
                (
                    "real current WOMD box "
                    "held fixed across forecast horizons"
                ),

            "deterministic_predictive":
                (
                    "Stage4 Gaussian mean only; "
                    "causal current dimensions/heading"
                ),

            "oracle_future":
                (
                    "real future WOMD box; "
                    "evaluation only"
                ),
        },

        "class_summary":
            summary,

        "generation_time_seconds":
            float(elapsed),

        "output":
            str(OUTPUT),
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

    print()
    print("=" * 78)
    print(
        "ADB BASELINE MASKS COMPLETE"
    )
    print("=" * 78)

    print(
        "reactive shape =",
        reactive_dimming.shape
    )

    print(
        "deterministic shape =",
        deterministic_dimming.shape
    )

    print(
        "oracle shape =",
        oracle_dimming.shape
    )

    print()
    print(
        "class summary ="
    )

    for cls, row in summary.items():
        print(
            cls,
            row
        )

    print()
    print(
        "time =",
        elapsed,
        "sec"
    )

    print(
        "Saved:",
        OUTPUT
    )

    print(
        "Report:",
        REPORT
    )


if __name__ == "__main__":
    main()
