from pathlib import Path
import json
import math
import time

import numpy as np


SOURCE = Path(
    "stage6_real/data/"
    "real_stage6_predictions_balanced.npz"
)

OUTPUT = Path(
    "stage6_real/data/"
    "real_stage6_probabilistic_occupancy.npz"
)

REPORT = Path(
    "stage6_real/reports/"
    "real_stage6_probabilistic_occupancy_summary.json"
)


# ============================================================
# EXPERIMENTAL DISCRETIZATION
#
# The PDF defines P_occ(theta, r, tau, c), but does not fix
# a particular numerical raster resolution. These are therefore
# explicit Stage-6 evaluation discretization parameters, not
# claimed as paper-provided hardware parameters.
# ============================================================

AZIMUTH_BINS = 64

AZIMUTH_MIN = -math.pi
AZIMUTH_MAX = math.pi

RANGE_BIN_M = 1.0

MC_SAMPLES = 500

SEED = 42


def gaussian_xy_samples(
    mean_xy,
    std_xy,
    rho,
    count,
    rng,
):

    z1 = rng.standard_normal(
        count
    )

    z2 = rng.standard_normal(
        count
    )

    sx = float(
        std_xy[0]
    )

    sy = float(
        std_xy[1]
    )

    r = float(
        np.clip(
            rho,
            -0.999,
            0.999,
        )
    )

    x = (
        float(mean_xy[0])
        +
        sx * z1
    )

    y = (
        float(mean_xy[1])
        +
        sy
        *
        (
            r * z1
            +
            math.sqrt(
                1.0 - r*r
            )
            * z2
        )
    )

    return x, y


def box_corners_xy(
    center_x,
    center_y,
    length,
    width,
    heading,
):

    # Four planar box corners.
    local = np.asarray(
        [
            [
                +length / 2.0,
                +width / 2.0,
            ],
            [
                +length / 2.0,
                -width / 2.0,
            ],
            [
                -length / 2.0,
                +width / 2.0,
            ],
            [
                -length / 2.0,
                -width / 2.0,
            ],
        ],
        dtype=np.float64,
    )

    c = math.cos(
        heading
    )

    s = math.sin(
        heading
    )

    rotation = np.asarray(
        [
            [c, -s],
            [s,  c],
        ],
        dtype=np.float64,
    )

    corners = (
        local
        @
        rotation.T
    )

    corners[:, 0] += (
        center_x
    )

    corners[:, 1] += (
        center_y
    )

    return corners


def angular_interval_from_corners(
    corners,
):

    angles = np.arctan2(
        corners[:, 1],
        corners[:, 0],
    )

    # For ordinary forward-driving boxes the occupied angular
    # support is local. Unwrap to avoid errors near +/-pi.
    unwrapped = np.unwrap(
        angles
    )

    return (
        float(
            unwrapped.min()
        ),
        float(
            unwrapped.max()
        ),
    )


def range_interval_from_corners(
    corners,
):

    ranges = np.sqrt(
        corners[:, 0]**2
        +
        corners[:, 1]**2
    )

    return (
        float(
            ranges.min()
        ),
        float(
            ranges.max()
        ),
    )


def azimuth_to_bin(
    angle,
):

    normalized = (
        angle
        -
        AZIMUTH_MIN
    ) / (
        AZIMUTH_MAX
        -
        AZIMUTH_MIN
    )

    index = int(
        math.floor(
            normalized
            *
            AZIMUTH_BINS
        )
    )

    return int(
        np.clip(
            index,
            0,
            AZIMUTH_BINS - 1,
        )
    )


def main():

    print("=" * 78)
    print(
        "STAGE 6 REAL PROBABILISTIC "
        "ADB OCCUPANCY"
    )
    print("=" * 78)

    data = np.load(
        SOURCE
    )

    classes = data[
        "actor_class"
    ]

    length = data[
        "length_m"
    ]

    width = data[
        "width_m"
    ]

    height = data[
        "height_m"
    ]

    heading = data[
        "current_heading_H_rad"
    ]

    mean = data[
        "future_mean_H_m"
    ]

    std = data[
        "future_std_xy_H_m"
    ]

    rho = data[
        "future_rho_xy_H"
    ]


    actors = mean.shape[0]
    horizons = mean.shape[1]


    if mean.shape != (
        actors,
        horizons,
        3,
    ):
        raise RuntimeError(
            "Unexpected mean shape."
        )


    # ========================================================
    # Range grid derived only from predicted/support geometry.
    #
    # We use a 1-m bin and choose the upper edge from the
    # available Stage-6 evaluation population rather than
    # inventing an optical hardware range.
    # ========================================================

    approximate_max_range = (
        np.sqrt(
            mean[..., 0]**2
            +
            mean[..., 1]**2
        )
        +
        0.5
        *
        np.maximum(
            length[:, None],
            width[:, None],
        )
        +
        4.0
        *
        np.maximum(
            std[..., 0],
            std[..., 1],
        )
    )

    RANGE_MAX_M = float(
        math.ceil(
            float(
                np.max(
                    approximate_max_range
                )
            )
            /
            RANGE_BIN_M
        )
        *
        RANGE_BIN_M
    )

    RANGE_BINS = int(
        math.ceil(
            RANGE_MAX_M
            /
            RANGE_BIN_M
        )
    )


    print(
        "actors =",
        actors
    )

    print(
        "horizons =",
        horizons
    )

    print(
        "MC samples/state =",
        MC_SAMPLES
    )

    print(
        "azimuth bins =",
        AZIMUTH_BINS
    )

    print(
        "range bin =",
        RANGE_BIN_M,
        "m"
    )

    print(
        "range max =",
        RANGE_MAX_M,
        "m"
    )

    print(
        "occupancy shape =",
        (
            actors,
            horizons,
            AZIMUTH_BINS,
            RANGE_BINS,
        )
    )


    # Each cell stores empirical probability that the sampled
    # future actor box overlaps that azimuth/range cell.
    occupancy = np.zeros(
        (
            actors,
            horizons,
            AZIMUTH_BINS,
            RANGE_BINS,
        ),
        dtype=np.float32,
    )

    rng = np.random.default_rng(
        SEED
    )

    start_time = time.perf_counter()


    for actor in range(
        actors
    ):

        L = float(
            length[actor]
        )

        W = float(
            width[actor]
        )

        psi = float(
            heading[actor]
        )


        for t in range(
            horizons
        ):

            xs, ys = gaussian_xy_samples(
                mean_xy=
                    mean[
                        actor,
                        t,
                        :2
                    ],

                std_xy=
                    std[
                        actor,
                        t
                    ],

                rho=
                    rho[
                        actor,
                        t
                    ],

                count=
                    MC_SAMPLES,

                rng=
                    rng,
            )


            # Binary occupancy per MC realization before
            # accumulation. A sampled box contributes at most
            # once to each raster cell.
            state_counts = np.zeros(
                (
                    AZIMUTH_BINS,
                    RANGE_BINS,
                ),
                dtype=np.uint16,
            )


            for sample in range(
                MC_SAMPLES
            ):

                corners = (
                    box_corners_xy(
                        float(
                            xs[sample]
                        ),
                        float(
                            ys[sample]
                        ),
                        L,
                        W,
                        psi,
                    )
                )


                az_min, az_max = (
                    angular_interval_from_corners(
                        corners
                    )
                )

                r_min, r_max = (
                    range_interval_from_corners(
                        corners
                    )
                )


                # If unwrap moved the support outside the canonical
                # interval, shift the local interval back.
                while (
                    az_min
                    <
                    AZIMUTH_MIN
                ):
                    az_min += (
                        2.0 * math.pi
                    )
                    az_max += (
                        2.0 * math.pi
                    )

                while (
                    az_max
                    >
                    AZIMUTH_MAX
                ):
                    az_min -= (
                        2.0 * math.pi
                    )
                    az_max -= (
                        2.0 * math.pi
                    )


                # Rare interval crossing +/-pi:
                # rasterize as two canonical intervals.
                intervals = []

                if (
                    az_min
                    <
                    AZIMUTH_MIN
                ):
                    intervals.append(
                        (
                            AZIMUTH_MIN,
                            az_max,
                        )
                    )

                    intervals.append(
                        (
                            az_min
                            +
                            2.0 * math.pi,
                            AZIMUTH_MAX,
                        )
                    )

                elif (
                    az_max
                    >
                    AZIMUTH_MAX
                ):
                    intervals.append(
                        (
                            az_min,
                            AZIMUTH_MAX,
                        )
                    )

                    intervals.append(
                        (
                            AZIMUTH_MIN,
                            az_max
                            -
                            2.0 * math.pi,
                        )
                    )

                else:
                    intervals.append(
                        (
                            az_min,
                            az_max,
                        )
                    )


                r0 = int(
                    np.clip(
                        math.floor(
                            r_min
                            /
                            RANGE_BIN_M
                        ),
                        0,
                        RANGE_BINS - 1,
                    )
                )

                r1 = int(
                    np.clip(
                        math.floor(
                            r_max
                            /
                            RANGE_BIN_M
                        ),
                        0,
                        RANGE_BINS - 1,
                    )
                )


                for a0, a1 in intervals:

                    b0 = azimuth_to_bin(
                        a0
                    )

                    b1 = azimuth_to_bin(
                        a1
                    )

                    lo = min(
                        b0,
                        b1
                    )

                    hi = max(
                        b0,
                        b1
                    )

                    state_counts[
                        lo:
                        hi + 1,

                        r0:
                        r1 + 1
                    ] += 1


            occupancy[
                actor,
                t
            ] = (
                state_counts.astype(
                    np.float32
                )
                /
                float(
                    MC_SAMPLES
                )
            )


        if (
            (actor + 1) % 25 == 0
            or
            actor + 1 == actors
        ):

            print(
                f"actors="
                f"{actor+1}/"
                f"{actors}",
                flush=True,
            )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    # ========================================================
    # Checks
    # ========================================================

    if not np.isfinite(
        occupancy
    ).all():
        raise RuntimeError(
            "Non-finite occupancy."
        )

    if (
        occupancy.min() < 0.0
        or
        occupancy.max() > 1.0
    ):
        raise RuntimeError(
            "Occupancy probability "
            "outside [0,1]."
        )


    max_occ_per_state = (
        occupancy.max(
            axis=(2, 3)
        )
    )

    occupied_cells_50 = (
        occupancy >= 0.50
    ).sum(
        axis=(2, 3)
    )

    occupied_cells_10 = (
        occupancy >= 0.10
    ).sum(
        axis=(2, 3)
    )


    # ========================================================
    # Save
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT,

        actor_class=
            classes,

        length_m=
            length,

        width_m=
            width,

        height_m=
            height,

        future_mean_H_m=
            mean,

        future_std_xy_H_m=
            std,

        future_rho_xy_H=
            rho,

        occupancy_probability=
            occupancy,

        azimuth_edges_rad=
            np.linspace(
                AZIMUTH_MIN,
                AZIMUTH_MAX,
                AZIMUTH_BINS + 1,
                dtype=np.float32,
            ),

        range_edges_m=
            np.linspace(
                0.0,
                RANGE_MAX_M,
                RANGE_BINS + 1,
                dtype=np.float32,
            ),
    )


    class_summary = {}

    for cls in [
        "VEHICLE",
        "PEDESTRIAN",
        "CYCLIST",
    ]:

        mask = (
            classes == cls
        )

        class_summary[
            cls
        ] = {
            "actors":
                int(
                    mask.sum()
                ),

            "mean_peak_occupancy":
                float(
                    max_occ_per_state[
                        mask
                    ].mean()
                ),

            "mean_cells_p_ge_0_50":
                float(
                    occupied_cells_50[
                        mask
                    ].mean()
                ),

            "mean_cells_p_ge_0_10":
                float(
                    occupied_cells_10[
                        mask
                    ].mean()
                ),
        }


    report = {
        "status":
            "PASS",

        "data_source":
            "real_WOMD_validation",

        "sampling":
            "class_balanced",

        "actors":
            int(
                actors
            ),

        "future_steps":
            int(
                horizons
            ),

        "coordinate_frame":
            "H0_anchor_headlamp_frame",

        "future_used_as_input":
            False,

        "occupancy_definition":
            (
                "Monte-Carlo empirical probability "
                "that the predicted future oriented "
                "actor box overlaps each azimuth-range cell"
            ),

        "monte_carlo_samples_per_state":
            MC_SAMPLES,

        "grid": {
            "azimuth_bins":
                AZIMUTH_BINS,

            "azimuth_min_rad":
                AZIMUTH_MIN,

            "azimuth_max_rad":
                AZIMUTH_MAX,

            "range_bin_m":
                RANGE_BIN_M,

            "range_max_m":
                RANGE_MAX_M,

            "range_bins":
                RANGE_BINS,

            "note":
                (
                    "The proposal defines P_occ(theta,r,tau,c) "
                    "but does not prescribe numerical raster "
                    "resolution. Grid resolution is an explicit "
                    "evaluation discretization."
                ),
        },

        "box_geometry": {
            "centroid_xy":
                "Stage4 Gaussian sampled",

            "dimensions":
                "causal current WOMD box dimensions",

            "heading":
                "causal current relative heading held fixed",

            "height":
                (
                    "retained as actor geometry metadata; "
                    "current occupancy raster is azimuth-range"
                ),
        },

        "class_summary":
            class_summary,

        "generation_time_seconds":
            float(
                elapsed
            ),

        "output":
            str(
                OUTPUT
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


    print()
    print("=" * 78)
    print(
        "PROBABILISTIC ADB OCCUPANCY COMPLETE"
    )
    print("=" * 78)

    print(
        "occupancy shape =",
        occupancy.shape
    )

    print(
        "min probability =",
        float(
            occupancy.min()
        )
    )

    print(
        "max probability =",
        float(
            occupancy.max()
        )
    )

    print()
    print(
        "class summary ="
    )

    for cls, row in class_summary.items():

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
