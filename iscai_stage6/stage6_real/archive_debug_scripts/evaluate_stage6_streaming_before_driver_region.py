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

OUTPUT = Path(
    "stage6_real/reports/"
    "stage6_streaming_evaluation.json"
)


# ============================================================
# Part-A illumination grid / parameters
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

THETA_RAD = np.deg2rad(
    THETA_DEG
)

RADIAL_MARGIN_M = 4.0
TRANSITION_LENGTH_M = 20.0
BASE_LATERAL_MARGIN_M = 1.0


# ============================================================
# Probabilistic evaluation
#
# Explicit experimental parameters.
# Not claimed as Part-A hardware constants.
# ============================================================

MC_SAMPLES = 200
SEED = 42

GAMMAS = [
    0.10,
    0.25,
    0.50,
]

CLASS_CONFIGS = {
    "VEHICLE": {
        "floor": 0.00,
        "lateral_margin_scale": 1.0,
    },

    "PEDESTRIAN": {
        # no complete blackout
        "floor": 0.35,
        "lateral_margin_scale": 1.25,
    },

    "CYCLIST": {
        # retain detection/visibility
        "floor": 0.25,
        "lateral_margin_scale": 1.50,
    },
}


# ============================================================
# Geometry helpers
# ============================================================

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

    c = math.cos(
        heading
    )

    s = math.sin(
        heading
    )

    R = np.asarray(
        [
            [c, -s],
            [s,  c],
        ],
        dtype=np.float64,
    )

    corners = (
        local
        @ R.T
    )

    corners[:,0] += (
        center_xy[0]
    )

    corners[:,1] += (
        center_xy[1]
    )

    return corners


def geometry_parameters(
    center_xy,
    length,
    width,
    heading,
    lateral_margin_scale=1.0,
):

    corners = box_corners(
        center_xy,
        length,
        width,
        heading,
    )

    angles = np.unwrap(
        np.arctan2(
            corners[:,1],
            corners[:,0],
        )
    )

    theta_min = float(
        angles.min()
    )

    theta_max = float(
        angles.max()
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
        (
            BASE_LATERAL_MARGIN_M
            *
            lateral_margin_scale
        )
        /
        center_range
    )

    theta_min -= epsilon
    theta_max += epsilon

    corner_ranges = np.hypot(
        corners[:,0],
        corners[:,1],
    )

    target_range = float(
        corner_ranges.max()
    )

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

    return (
        theta_min,
        theta_max,
        r_min,
        r_max,
    )


def intensity_map(
    center_xy,
    length,
    width,
    heading,
    lateral_margin_scale=1.0,
    intensity_floor=0.0,
):

    (
        theta_min,
        theta_max,
        r_min,
        r_max,
    ) = geometry_parameters(
        center_xy,
        length,
        width,
        heading,
        lateral_margin_scale,
    )

    center_angle = (
        0.5
        *
        (
            theta_min
            +
            theta_max
        )
    )

    half_angle = (
        0.5
        *
        (
            theta_max
            -
            theta_min
        )
    )

    angular_difference = np.angle(
        np.exp(
            1j
            *
            (
                THETA_RAD
                -
                center_angle
            )
        )
    )

    inside_angle = (
        np.abs(
            angular_difference
        )
        <=
        half_angle
    )

    L = np.ones(
        (
            len(THETA_RAD),
            len(RANGE_M),
        ),
        dtype=np.float32,
    )

    if not np.any(
        inside_angle
    ):
        return L

    full_r = (
        RANGE_M
        <=
        r_min
    )

    transition_r = (
        (RANGE_M > r_min)
        &
        (RANGE_M < r_max)
    )

    full = (
        inside_angle[:,None]
        &
        full_r[None,:]
    )

    transition = (
        inside_angle[:,None]
        &
        transition_r[None,:]
    )

    L[
        full
    ] = intensity_floor

    if np.any(
        transition
    ):

        radial_profile = np.ones(
            len(RANGE_M),
            dtype=np.float32,
        )

        u = (
            RANGE_M[
                transition_r
            ]
            -
            r_min
        ) / (
            r_max
            -
            r_min
        )

        base = (
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

        # raised cosine from floor -> 1
        radial_profile[
            transition_r
        ] = (
            intensity_floor
            +
            (
                1.0
                -
                intensity_floor
            )
            *
            base
        )

        L[
            transition
        ] = np.broadcast_to(
            radial_profile[
                None,
                :
            ],
            L.shape,
        )[
            transition
        ]

    return L


# ============================================================
# Monte-Carlo occupancy on Part-A grid
# ============================================================

def sample_gaussian_xy(
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
        sx*z1
    )

    y = (
        float(mean_xy[1])
        +
        sy
        *
        (
            r*z1
            +
            math.sqrt(
                1.0-r*r
            )
            *
            z2
        )
    )

    return (
        x,
        y,
    )


def probabilistic_occupancy(
    mean_xy,
    std_xy,
    rho,
    length,
    width,
    heading,
    lateral_margin_scale,
    rng,
):

    counts = np.zeros(
        (
            len(THETA_RAD),
            len(RANGE_M),
        ),
        dtype=np.uint16,
    )

    xs, ys = sample_gaussian_xy(
        mean_xy,
        std_xy,
        rho,
        MC_SAMPLES,
        rng,
    )

    for i in range(
        MC_SAMPLES
    ):

        (
            a0,
            a1,
            r0,
            _,
        ) = geometry_parameters(
            (
                float(xs[i]),
                float(ys[i]),
            ),
            length,
            width,
            heading,
            lateral_margin_scale,
        )

        center = (
            0.5
            *
            (a0+a1)
        )

        half = (
            0.5
            *
            (a1-a0)
        )

        da = np.angle(
            np.exp(
                1j
                *
                (
                    THETA_RAD
                    -
                    center
                )
            )
        )

        a_mask = (
            np.abs(da)
            <=
            half
        )

        r_mask = (
            RANGE_M
            <=
            r0
        )

        if (
            np.any(a_mask)
            and
            np.any(r_mask)
        ):
            counts[
                np.ix_(
                    a_mask,
                    r_mask,
                )
            ] += 1

    return (
        counts.astype(
            np.float32
        )
        /
        float(
            MC_SAMPLES
        )
    )


# ============================================================
# Metrics
# ============================================================

def binary_mask(
    dimming,
    threshold=0.5,
):
    return (
        dimming
        >=
        threshold
    )


def iou(
    pred,
    oracle,
):

    p = binary_mask(
        pred
    )

    o = binary_mask(
        oracle
    )

    union = np.logical_or(
        p,
        o,
    ).sum()

    if union == 0:
        return 1.0

    inter = np.logical_and(
        p,
        o,
    ).sum()

    return float(
        inter
        /
        union
    )


def shadow_violation(
    pred,
    oracle,
):
    """
    Fraction of oracle-dimmed area for which
    the candidate supplies insufficient dimming.
    """

    o = (
        oracle
        >=
        0.5
    )

    denom = int(
        o.sum()
    )

    if denom == 0:
        return 0.0

    violation = (
        o
        &
        (
            pred
            <
            0.5
        )
    )

    return float(
        violation.sum()
        /
        denom
    )


def overmask(
    pred,
    oracle,
):

    p = (
        pred
        >=
        0.5
    )

    o = (
        oracle
        >=
        0.5
    )

    return float(
        np.logical_and(
            p,
            ~o,
        ).mean()
    )


def road_retention(
    dimming,
):

    return float(
        (
            1.0
            -
            dimming
        ).mean()
    )


def change_rate(
    current,
    previous,
):

    if previous is None:
        return 0.0

    return float(
        np.mean(
            np.abs(
                current
                -
                previous
            )
        )
    )


# ============================================================
# Summary accumulator
# ============================================================


def false_dimming(
    pred,
    oracle,
):
    """
    Fraction of predicted dimmed cells that are not required
    by the constructed oracle.

    This differs from overmask(), whose denominator is the
    complete ADB grid.
    """

    p = (
        pred
        >=
        0.5
    )

    o = (
        oracle
        >=
        0.5
    )

    denom = int(
        p.sum()
    )

    if denom == 0:
        return 0.0

    false = np.logical_and(
        p,
        ~o,
    )

    return float(
        false.sum()
        /
        denom
    )


def oracle_region_visibility(
    dimming,
    oracle,
):
    """
    Mean retained illumination inside the constructed
    oracle future actor region.

    1.0 = fully illuminated.
    0.0 = complete blackout.
    """

    region = (
        oracle
        >=
        0.5
    )

    denom = int(
        region.sum()
    )

    if denom == 0:
        return 1.0

    illumination = (
        1.0
        -
        dimming
    )

    return float(
        illumination[
            region
        ].mean()
    )


def temporal_smoothness(
    current,
    previous,
):
    """
    Complement of mean absolute frame-to-frame dimming
    change. Higher is smoother.
    """

    if previous is None:
        return 1.0

    return float(
        1.0
        -
        np.mean(
            np.abs(
                current
                -
                previous
            )
        )
    )


def flicker_rate(
    current,
    previous,
    threshold,
):
    """
    Fraction of ADB cells whose dimming changes by more
    than the selected threshold between consecutive
    horizons.
    """

    if previous is None:
        return 0.0

    return float(
        np.mean(
            np.abs(
                current
                -
                previous
            )
            >
            threshold
        )
    )


def empty_accumulator():

    return {
        "count": 0,
        "iou": [],
        "shadow_violation": [],
        "overmask": [],
        "road_retention": [],
        "mean_dimming": [],
        "change_rate": [],
        "false_dimming": [],
        "oracle_region_visibility": [],
        "temporal_smoothness": [],
        "flicker_rate_0_10": [],
        "flicker_rate_0_25": [],
        "flicker_rate_0_50": [],
    }


def append_metrics(
    acc,
    candidate,
    oracle,
    previous,
):

    acc[
        "count"
    ] += 1

    acc[
        "iou"
    ].append(
        iou(
            candidate,
            oracle,
        )
    )

    acc[
        "shadow_violation"
    ].append(
        shadow_violation(
            candidate,
            oracle,
        )
    )

    acc[
        "overmask"
    ].append(
        overmask(
            candidate,
            oracle,
        )
    )

    acc[
        "road_retention"
    ].append(
        road_retention(
            candidate
        )
    )

    acc[
        "mean_dimming"
    ].append(
        float(
            candidate.mean()
        )
    )

    acc[
        "change_rate"
    ].append(
        change_rate(
            candidate,
            previous,
        )
    )

    acc[
        "false_dimming"
    ].append(
        false_dimming(
            candidate,
            oracle,
        )
    )

    acc[
        "oracle_region_visibility"
    ].append(
        oracle_region_visibility(
            candidate,
            oracle,
        )
    )

    acc[
        "temporal_smoothness"
    ].append(
        temporal_smoothness(
            candidate,
            previous,
        )
    )

    acc[
        "flicker_rate_0_10"
    ].append(
        flicker_rate(
            candidate,
            previous,
            0.10,
        )
    )

    acc[
        "flicker_rate_0_25"
    ].append(
        flicker_rate(
            candidate,
            previous,
            0.25,
        )
    )

    acc[
        "flicker_rate_0_50"
    ].append(
        flicker_rate(
            candidate,
            previous,
            0.50,
        )
    )


def summarize(
    acc,
):

    result = {
        "states":
            int(
                acc[
                    "count"
                ]
            )
    }

    for key in [
        "iou",
        "shadow_violation",
        "overmask",
        "road_retention",
        "mean_dimming",
        "change_rate",
        "false_dimming",
        "oracle_region_visibility",
        "temporal_smoothness",
        "flicker_rate_0_10",
        "flicker_rate_0_25",
        "flicker_rate_0_50",
    ]:

        x = np.asarray(
            acc[key],
            dtype=np.float64,
        )

        result[
            f"mean_{key}"
        ] = (
            float(
                x.mean()
            )
            if len(x)
            else None
        )

    return result


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 78)
    print(
        "STAGE 6 REAL STREAMING "
        "PREDICTIVE ADB EVALUATION"
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

    current_heading = oracle[
        "current_heading_H_rad"
    ]

    current_lwh = oracle[
        "current_lwh_m"
    ]

    gt_center = oracle[
        "future_center_H_m"
    ]

    gt_heading = oracle[
        "future_heading_H_rad"
    ]

    gt_lwh = oracle[
        "future_lwh_m"
    ]


    # ========================================================
    # Causal cohorts
    # ========================================================

    current_angle = np.degrees(
        np.arctan2(
            current_center[:,1],
            current_center[:,0],
        )
    )

    current_range = np.hypot(
        current_center[:,0],
        current_center[:,1],
    )

    current_fov = (
        (current_center[:,0] > 0)
        &
        (
            np.abs(
                current_angle
            )
            <=
            25.0
        )
        &
        (
            current_range
            <=
            180.0
        )
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

    predicted_future_fov = (
        (mu[...,0] > 0)
        &
        (
            np.abs(
                pred_angle
            )
            <=
            25.0
        )
        &
        (
            pred_range
            <=
            180.0
        )
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


    print(
        "total actors =",
        len(classes)
    )

    print(
        "current-FOV actors =",
        int(
            current_fov.sum()
        )
    )

    print(
        "predicted-entry actors =",
        int(
            predicted_entry.sum()
        )
    )

    print(
        "evaluation union =",
        int(
            evaluation_actor.sum()
        )
    )

    print(
        "MC samples =",
        MC_SAMPLES
    )

    print(
        "grid =",
        (
            len(THETA_DEG),
            len(RANGE_M),
        )
    )


    # ========================================================
    # Accumulators
    # ========================================================

    methods = [
        "reactive",
        "deterministic",
    ]

    for gamma in GAMMAS:

        methods.append(
            f"uncertainty_gamma_{gamma}"
        )

        methods.append(
            f"class_aware_gamma_{gamma}"
        )

    accumulators = {
        "all_relevant": {
            method:
                empty_accumulator()
            for method in methods
        },

        "current_fov": {
            method:
                empty_accumulator()
            for method in methods
        },

        "predicted_entry": {
            method:
                empty_accumulator()
            for method in methods
        },
    }


    class_accumulators = {
        cls: {
            method:
                empty_accumulator()
            for method in methods
        }
        for cls in [
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
        ]
    }


    rng = np.random.default_rng(
        SEED
    )

    selected_indices = np.where(
        evaluation_actor
    )[0]

    start_time = time.perf_counter()


    for pos, actor in enumerate(
        selected_indices,
        start=1,
    ):

        cls = str(
            classes[
                actor
            ]
        )

        config = CLASS_CONFIGS[
            cls
        ]


        # ----------------------------------------------
        # Reactive map is based on current causal box.
        # ----------------------------------------------

        reactive_L = intensity_map(
            current_center[
                actor,
                :2
            ],
            float(
                current_lwh[
                    actor,
                    0
                ]
            ),
            float(
                current_lwh[
                    actor,
                    1
                ]
            ),
            float(
                current_heading[
                    actor
                ]
            ),
        )

        reactive_D = (
            1.0
            -
            reactive_L
        )


        previous = {
            method: None
            for method in methods
        }


        for t in range(10):

            # ------------------------------------------
            # Oracle future
            # ------------------------------------------

            oracle_L = intensity_map(
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


            # ------------------------------------------
            # Deterministic predictive
            # ------------------------------------------

            deterministic_L = intensity_map(
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


            candidate_maps = {
                "reactive":
                    reactive_D,

                "deterministic":
                    deterministic_D,
            }


            # ------------------------------------------
            # Probabilistic occupancy
            # ------------------------------------------

            occupancy_class_agnostic = (
                probabilistic_occupancy(
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
                    1.0,
                    rng,
                )
            )


            occupancy_class_aware = (
                probabilistic_occupancy(
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
            )


            for gamma in GAMMAS:

                # --------------------------------------
                # uncertainty-aware, class agnostic
                # --------------------------------------

                mask = (
                    occupancy_class_agnostic
                    >=
                    gamma
                )

                uncertainty_D = (
                    deterministic_D.copy()
                )

                # Probability controls MASK MEMBERSHIP.
                # Once P_occ >= gamma, this cell belongs to the
                # uncertainty-aware predictive dimming region.
                uncertainty_D[
                    mask
                ] = 1.0

                candidate_maps[
                    f"uncertainty_gamma_{gamma}"
                ] = uncertainty_D


                # --------------------------------------
                # class-aware predictive
                # --------------------------------------

                class_mask = (
                    occupancy_class_aware
                    >=
                    gamma
                )

                # Maximum possible dimming for class.
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

                # Probability controls MASK MEMBERSHIP.
                # Dimming depth is controlled separately by the
                # class-specific illumination floor.
                class_D[
                    class_mask
                ] = np.maximum(
                    class_D[
                        class_mask
                    ],
                    max_dimming,
                )

                # Guarantee the class-specific
                # illumination floor.
                class_D = np.minimum(
                    class_D,
                    max_dimming,
                )

                candidate_maps[
                    f"class_aware_gamma_{gamma}"
                ] = class_D


            # ------------------------------------------
            # Metrics
            # ------------------------------------------

            cohorts = [
                (
                    "all_relevant",
                    True,
                ),
                (
                    "current_fov",
                    bool(
                        current_fov[
                            actor
                        ]
                    ),
                ),
                (
                    "predicted_entry",
                    bool(
                        predicted_entry[
                            actor
                        ]
                    ),
                ),
            ]


            for method, candidate in candidate_maps.items():

                for cohort_name, enabled in cohorts:

                    if enabled:
                        append_metrics(
                            accumulators[
                                cohort_name
                            ][
                                method
                            ],
                            candidate,
                            oracle_D,
                            previous[
                                method
                            ],
                        )

                append_metrics(
                    class_accumulators[
                        cls
                    ][
                        method
                    ],
                    candidate,
                    oracle_D,
                    previous[
                        method
                    ],
                )

                previous[
                    method
                ] = candidate


        if (
            pos % 10 == 0
            or
            pos == len(
                selected_indices
            )
        ):

            print(
                f"actors="
                f"{pos}/"
                f"{len(selected_indices)}",
                flush=True,
            )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    # ========================================================
    # Summaries
    # ========================================================

    results = {
        cohort: {
            method:
                summarize(
                    acc
                )
            for method, acc
            in method_map.items()
        }
        for cohort, method_map
        in accumulators.items()
    }


    by_class = {
        cls: {
            method:
                summarize(
                    acc
                )
            for method, acc
            in method_map.items()
        }
        for cls, method_map
        in class_accumulators.items()
    }


    # ========================================================
    # Gamma operating points
    #
    # No single gamma is declared globally optimal here.
    # The proposal requires a safety / over-masking /
    # visibility trade-off, so all gamma operating points
    # are retained for comparison.
    # ========================================================

    report = {
        "status":
            "PASS",

        "stage":
            6,

        "evaluation":
            (
                "Streaming predictive ADB evaluation "
                "on the Part-A 501x361 illumination grid."
            ),

        "data_source":
            "real_WOMD_validation_class_balanced",

        "future_used_as_input":
            False,

        "part_a": {
            "theta_deg":
                [
                    -25.0,
                    25.0,
                ],

            "theta_samples":
                501,

            "range_m":
                [
                    0.0,
                    180.0,
                ],

            "range_samples":
                361,

            "radial_margin_m":
                RADIAL_MARGIN_M,

            "transition_length_m":
                TRANSITION_LENGTH_M,

            "base_lateral_margin_m":
                BASE_LATERAL_MARGIN_M,

            "transition":
                "raised_cosine",
        },

        "cohorts": {
            "all_balanced_actors":
                int(
                    len(classes)
                ),

            "current_fov_actors":
                int(
                    current_fov.sum()
                ),

            "predicted_entry_actors":
                int(
                    predicted_entry.sum()
                ),

            "causal_evaluation_union":
                int(
                    evaluation_actor.sum()
                ),
        },

        "probabilistic_policy": {
            "mc_samples":
                MC_SAMPLES,

            "gamma_sweep":
                GAMMAS,

            "class_configs":
                CLASS_CONFIGS,

            "note":
                (
                    "Gamma, class floors and class margin "
                    "scales are explicit experimental "
                    "operating parameters because the "
                    "proposal does not provide numerical "
                    "values for them."
                ),
        },

        "results":
            results,

        "results_by_class":
            by_class,

        "selection": {
            "policy":
                (
                    "No single gamma declared optimal. "
                    "All operating points are reported for "
                    "safety/over-masking/visibility comparison."
                )
        },

        "runtime_seconds":
            float(
                elapsed
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


    # ========================================================
    # Console headline
    # ========================================================

    print()
    print("=" * 78)
    print("STAGE 6 STREAMING EVALUATION COMPLETE")
    print("=" * 78)

    print(
        "runtime =",
        elapsed,
        "sec"
    )

    print(
        "gamma operating points =",
        GAMMAS
    )

    print()

    headline_methods = [
        "reactive",
        "deterministic",
    ] + [
        f"class_aware_gamma_{g}"
        for g in GAMMAS
    ]

    for name in headline_methods:

        r = results[
            "all_relevant"
        ][
            name
        ]

        print(
            f"{name:28s} "
            f"IoU={r['mean_iou']:.5f} "
            f"violation={100*r['mean_shadow_violation']:.3f}% "
            f"overmask={100*r['mean_overmask']:.3f}% "
            f"retention={r['mean_road_retention']:.5f} "
            f"change={r['mean_change_rate']:.6f}"
        )

    print()
    print("BY CLASS / CLASS-AWARE GAMMA SWEEP")

    for gamma in GAMMAS:

        name = f"class_aware_gamma_{gamma}"

        print()
        print(
            "gamma =",
            gamma
        )

        for cls in [
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
        ]:

            r = by_class[
                cls
            ][
                name
            ]

            print(
                f"{cls:12s} "
                f"IoU={r['mean_iou']:.5f} "
                f"violation={100*r['mean_shadow_violation']:.3f}% "
                f"overmask={100*r['mean_overmask']:.3f}% "
                f"retention={r['mean_road_retention']:.5f}"
            )

    print()
    print(
        "Saved:",
        OUTPUT
    )


if __name__ == "__main__":
    main()
