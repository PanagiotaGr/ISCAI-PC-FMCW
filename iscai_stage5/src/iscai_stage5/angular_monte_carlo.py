from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np

from iscai_stage5.angular_posterior import (
    circular_angle_difference,
)


HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

FROZEN_VARIANCE_SCALE_ALPHA_H = (
    1.2347064500315355,
    1.3451250295202921,
    1.354822057728461,
    1.2829700217319266,
)

TRAJECTORY_JOINT_SAMPLING_SEMANTICS = (
    "product_of_frozen_calibrated_"
    "per_horizon_Gaussian_marginals"
)

FUTURE_HEADING_SEMANTICS = (
    "samplewise_trajectory_tangent_with_"
    "causal_heading_fallback"
)

RECEIVER_OFFSET_TEMPORAL_SEMANTICS = (
    "one_body_frame_receiver_offset_draw_"
    "shared_across_all_horizons_per_MC_sample"
)

LOW_SPEED_HEADING_THRESHOLD_MPS = (
    0.25
)

LOW_SPEED_HEADING_THRESHOLD_BASIS = (
    "preregistered_numerical_stability_choice_"
    "equivalent_to_2p5cm_displacement_over_0p1s"
)

MONTE_CARLO_BASE_SEED = (
    20260821
)

MONTE_CARLO_CANDIDATE_SAMPLE_COUNTS = (
    2048,
    4096,
    8192,
)

MONTE_CARLO_REFERENCE_SAMPLE_COUNT = (
    32768
)

MONTE_CARLO_MINIMUM_SAMPLE_COUNT = (
    2048
)

MONTE_CARLO_MINIMUM_SAMPLE_COUNT_BASIS = (
    "at_least_2048_samples_so_probability_"
    "mass_resolution_is_below_0p05_percent"
)

RECEIVER_GEOMETRY_MODES = (
    "centroid_baseline",
    "known_receiver_offset",
    "uncertain_receiver_offset",
)


@dataclass(
    frozen=True
)
class ReceiverOffsetGaussianBody:
    mean_body_m: tuple[
        float,
        float,
        float,
    ]

    covariance_body_m2: tuple[
        tuple[
            float,
            float,
            float,
        ],
        tuple[
            float,
            float,
            float,
        ],
        tuple[
            float,
            float,
            float,
        ],
    ]


@dataclass(
    frozen=True
)
class HorizonAngularSummary:
    horizon_s: float

    range_mean_m: float

    range_std_m: float

    azimuth_circular_mean_rad: float

    azimuth_circular_std_rad: float

    elevation_mean_rad: float

    elevation_std_rad: float

    covariance_rae: tuple[
        tuple[
            float,
            float,
            float,
        ],
        tuple[
            float,
            float,
            float,
        ],
        tuple[
            float,
            float,
            float,
        ],
    ]

    heading_fallback_fraction: float


@dataclass(
    frozen=True
)
class ReceiverAngularMonteCarloPosterior:
    """
    Primary Stage5 receiver/angular posterior.

    Arrays have shape [N, H], with H=4.

    The object retains samples because Block5.3 will map
    these posterior samples into beam probability masses.
    """

    range_samples_m: np.ndarray

    azimuth_samples_rad: np.ndarray

    elevation_samples_rad: np.ndarray

    heading_samples_rad: np.ndarray

    receiver_position_samples_h0_m: np.ndarray

    summaries: tuple[
        HorizonAngularSummary,
        ...
    ]

    sample_count: int

    seed: int

    receiver_geometry_mode: str

    sample_sha256: str

    trajectory_joint_sampling_semantics: str = (
        TRAJECTORY_JOINT_SAMPLING_SEMANTICS
    )

    receiver_offset_temporal_semantics: str = (
        RECEIVER_OFFSET_TEMPORAL_SEMANTICS
    )


def _finite_vector3(
    values,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.shape != (
        3,
    ):
        raise ValueError(
            f"{name} must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return array


def _finite_horizon_positions(
    values,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.shape != (
        4,
        3,
    ):
        raise ValueError(
            f"{name} must have shape (4,3)."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return array


def _psd_covariance3(
    values,
    *,
    name: str,
) -> np.ndarray:
    covariance = np.asarray(
        values,
        dtype=np.float64,
    )

    if covariance.shape != (
        3,
        3,
    ):
        raise ValueError(
            f"{name} must have shape (3,3)."
        )

    if not np.all(
        np.isfinite(
            covariance
        )
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if not np.allclose(
        covariance,
        covariance.T,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name} must be symmetric."
        )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    if float(
        np.min(
            eigenvalues
        )
    ) < -1e-10:
        raise ValueError(
            f"{name} must be PSD."
        )

    return (
        0.5
        *
        (
            covariance
            +
            covariance.T
        )
    )


def _horizon_covariances(
    values,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.shape != (
        4,
        3,
        3,
    ):
        raise ValueError(
            "trajectory_covariance_h0_m2 "
            "must have shape (4,3,3)."
        )

    result = np.zeros_like(
        array
    )

    for horizon_index in range(
        4
    ):
        result[
            horizon_index
        ] = _psd_covariance3(
            array[
                horizon_index
            ],
            name=(
                "trajectory_covariance_h0_m2"
                f"[{horizon_index}]"
            ),
        )

    return result


def calibrate_stage4_horizon_covariances(
    raw_covariance_h0_m2,
    variance_scale_alpha_h=(
        FROZEN_VARIANCE_SCALE_ALPHA_H
    ),
) -> np.ndarray:
    """
    Apply the frozen Block4.5 covariance calibration:

        Sigma_cal,h = alpha_h * Sigma_raw,h

    Stage4 predictive mean remains unchanged.
    Stage2 measurement covariance R_t is not touched.
    """

    raw = _horizon_covariances(
        raw_covariance_h0_m2
    )

    alpha = np.asarray(
        variance_scale_alpha_h,
        dtype=np.float64,
    )

    if alpha.shape != (
        4,
    ):
        raise ValueError(
            "variance_scale_alpha_h "
            "must have shape (4,)."
        )

    if (
        not np.all(
            np.isfinite(
                alpha
            )
        )
        or
        np.any(
            alpha
            <=
            0.0
        )
    ):
        raise ValueError(
            "variance_scale_alpha_h "
            "must be finite and positive."
        )

    calibrated = (
        raw
        *
        alpha[
            :,
            None,
            None,
        ]
    )

    return calibrated


def _psd_factor(
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Deterministic PSD square-root.

    Cholesky is used for strictly positive-definite
    covariance.  Eigen fallback is used only for valid
    semidefinite cases such as zero placement covariance.
    """

    covariance = _psd_covariance3(
        covariance,
        name="covariance",
    )

    if np.allclose(
        covariance,
        0.0,
        atol=0.0,
        rtol=0.0,
    ):
        return np.zeros(
            (
                3,
                3,
            ),
            dtype=np.float64,
        )

    try:
        return np.linalg.cholesky(
            covariance
        )

    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = (
            np.linalg.eigh(
                covariance
            )
        )

        if float(
            np.min(
                eigenvalues
            )
        ) < -1e-10:
            raise ValueError(
                "Covariance is not PSD."
            )

        eigenvalues = np.maximum(
            eigenvalues,
            0.0,
        )

        #
        # Canonicalize eigenvector signs for deterministic
        # hashing within the frozen numerical environment.
        #
        eigenvectors = eigenvectors.copy()

        for column in range(
            eigenvectors.shape[
                1
            ]
        ):
            vector = eigenvectors[
                :,
                column
            ]

            pivot = int(
                np.argmax(
                    np.abs(
                        vector
                    )
                )
            )

            if vector[
                pivot
            ] < 0.0:
                eigenvectors[
                    :,
                    column
                ] *= -1.0

        return (
            eigenvectors
            @
            np.diag(
                np.sqrt(
                    eigenvalues
                )
            )
        )


def _sample_gaussian3(
    *,
    mean,
    covariance,
    standard_normal_samples,
) -> np.ndarray:
    mean_array = _finite_vector3(
        mean,
        name="mean",
    )

    covariance_array = _psd_covariance3(
        covariance,
        name="covariance",
    )

    z = np.asarray(
        standard_normal_samples,
        dtype=np.float64,
    )

    if (
        z.ndim != 2
        or
        z.shape[
            1
        ] != 3
    ):
        raise ValueError(
            "standard_normal_samples "
            "must have shape (N,3)."
        )

    factor = _psd_factor(
        covariance_array
    )

    return (
        mean_array[
            None,
            :
        ]
        +
        z
        @
        factor.T
    )


def sample_product_of_horizon_gaussians(
    *,
    trajectory_mean_h0_m,
    calibrated_covariance_h0_m2,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """
    Sample the only joint trajectory distribution directly
    implied by the frozen Gaussian output:

        prod_h N(mu_h, Sigma_cal,h)

    No cross-horizon covariance is invented.
    """

    mean = _finite_horizon_positions(
        trajectory_mean_h0_m,
        name="trajectory_mean_h0_m",
    )

    covariance = _horizon_covariances(
        calibrated_covariance_h0_m2
    )

    sample_count = int(
        sample_count
    )

    if sample_count <= 0:
        raise ValueError(
            "sample_count must be positive."
        )

    generator = np.random.Generator(
        np.random.PCG64(
            int(
                seed
            )
        )
    )

    z = generator.standard_normal(
        (
            sample_count,
            4,
            3,
        ),
        dtype=np.float64,
    )

    result = np.empty(
        (
            sample_count,
            4,
            3,
        ),
        dtype=np.float64,
    )

    for horizon_index in range(
        4
    ):
        result[
            :,
            horizon_index,
            :
        ] = _sample_gaussian3(
            mean=(
                mean[
                    horizon_index
                ]
            ),

            covariance=(
                covariance[
                    horizon_index
                ]
            ),

            standard_normal_samples=(
                z[
                    :,
                    horizon_index,
                    :
                ]
            ),
        )

    return result


def resolve_samplewise_headings(
    *,
    trajectory_samples_h0_m,
    current_position_h0_m,
    current_heading_h0_rad: float,
    low_speed_threshold_mps: float = (
        LOW_SPEED_HEADING_THRESHOLD_MPS
    ),
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Resolve future body heading from sampled trajectory
    tangents.

    At low planar speed, heading is carried forward from the
    previous causal/predicted heading rather than using an
    unstable atan2 of an almost-zero displacement.

    Returns:
        headings_rad         [N,4]
        fallback_mask        [N,4]
    """

    samples = np.asarray(
        trajectory_samples_h0_m,
        dtype=np.float64,
    )

    if (
        samples.ndim != 3
        or
        samples.shape[
            1:
        ] != (
            4,
            3,
        )
    ):
        raise ValueError(
            "trajectory_samples_h0_m "
            "must have shape (N,4,3)."
        )

    if not np.all(
        np.isfinite(
            samples
        )
    ):
        raise ValueError(
            "trajectory samples must be finite."
        )

    current_position = _finite_vector3(
        current_position_h0_m,
        name="current_position_h0_m",
    )

    current_heading = float(
        current_heading_h0_rad
    )

    if not math.isfinite(
        current_heading
    ):
        raise ValueError(
            "current_heading_h0_rad "
            "must be finite."
        )

    threshold = float(
        low_speed_threshold_mps
    )

    if (
        not math.isfinite(
            threshold
        )
        or
        threshold < 0.0
    ):
        raise ValueError(
            "low_speed_threshold_mps "
            "must be finite and non-negative."
        )

    count = samples.shape[
        0
    ]

    headings = np.empty(
        (
            count,
            4,
        ),
        dtype=np.float64,
    )

    fallback = np.zeros(
        (
            count,
            4,
        ),
        dtype=bool,
    )

    previous_position = np.repeat(
        current_position[
            None,
            :
        ],
        count,
        axis=0,
    )

    previous_heading = np.full(
        (
            count,
        ),
        current_heading,
        dtype=np.float64,
    )

    previous_horizon = 0.0

    for horizon_index, horizon in enumerate(
        HORIZONS_S
    ):
        dt = (
            float(
                horizon
            )
            -
            previous_horizon
        )

        if dt <= 0.0:
            raise ValueError(
                "Horizons must be strictly increasing."
            )

        current = samples[
            :,
            horizon_index,
            :
        ]

        displacement = (
            current
            -
            previous_position
        )

        planar_speed = (
            np.hypot(
                displacement[
                    :,
                    0
                ],
                displacement[
                    :,
                    1
                ],
            )
            /
            dt
        )

        moving = (
            planar_speed
            >
            threshold
        )

        candidate_heading = np.arctan2(
            displacement[
                :,
                1
            ],
            displacement[
                :,
                0
            ],
        )

        resolved = np.where(
            moving,
            candidate_heading,
            previous_heading,
        )

        headings[
            :,
            horizon_index
        ] = resolved

        fallback[
            :,
            horizon_index
        ] = ~moving

        previous_position = current

        previous_heading = resolved

        previous_horizon = float(
            horizon
        )

    return (
        headings,
        fallback,
    )


def _yaw_rotate_samples(
    body_vectors: np.ndarray,
    headings_rad: np.ndarray,
) -> np.ndarray:
    """
    Rotate one body-frame vector per MC sample into H0 at
    every horizon.

    body_vectors shape: [N,3]
    headings shape:     [N,4]

    output shape:       [N,4,3]
    """

    vectors = np.asarray(
        body_vectors,
        dtype=np.float64,
    )

    headings = np.asarray(
        headings_rad,
        dtype=np.float64,
    )

    if (
        vectors.ndim != 2
        or
        vectors.shape[
            1
        ] != 3
    ):
        raise ValueError(
            "body_vectors must have shape (N,3)."
        )

    if headings.shape != (
        vectors.shape[
            0
        ],
        4,
    ):
        raise ValueError(
            "headings_rad must have shape (N,4)."
        )

    cosine = np.cos(
        headings
    )

    sine = np.sin(
        headings
    )

    x_body = vectors[
        :,
        0
    ][
        :,
        None
    ]

    y_body = vectors[
        :,
        1
    ][
        :,
        None
    ]

    z_body = vectors[
        :,
        2
    ][
        :,
        None
    ]

    result = np.empty(
        (
            vectors.shape[
                0
            ],
            4,
            3,
        ),
        dtype=np.float64,
    )

    result[
        :,
        :,
        0
    ] = (
        cosine
        *
        x_body
        -
        sine
        *
        y_body
    )

    result[
        :,
        :,
        1
    ] = (
        sine
        *
        x_body
        +
        cosine
        *
        y_body
    )

    result[
        :,
        :,
        2
    ] = np.broadcast_to(
        z_body,
        (
            vectors.shape[
                0
            ],
            4,
        ),
    )

    return result


def _circular_mean(
    angles_rad: np.ndarray,
) -> float:
    sine_mean = float(
        np.mean(
            np.sin(
                angles_rad
            )
        )
    )

    cosine_mean = float(
        np.mean(
            np.cos(
                angles_rad
            )
        )
    )

    return float(
        math.atan2(
            sine_mean,
            cosine_mean,
        )
    )


def _circular_std(
    angles_rad: np.ndarray,
) -> float:
    sine_mean = float(
        np.mean(
            np.sin(
                angles_rad
            )
        )
    )

    cosine_mean = float(
        np.mean(
            np.cos(
                angles_rad
            )
        )
    )

    resultant = math.hypot(
        sine_mean,
        cosine_mean,
    )

    resultant = min(
        1.0,
        max(
            resultant,
            1e-15,
        ),
    )

    return float(
        math.sqrt(
            max(
                0.0,
                -2.0
                *
                math.log(
                    resultant
                ),
            )
        )
    )


def _angular_samples_from_positions(
    receiver_position_samples_h0_m: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    positions = np.asarray(
        receiver_position_samples_h0_m,
        dtype=np.float64,
    )

    if (
        positions.ndim != 3
        or
        positions.shape[
            1:
        ] != (
            4,
            3,
        )
    ):
        raise ValueError(
            "receiver positions must "
            "have shape (N,4,3)."
        )

    x = positions[
        :,
        :,
        0
    ]

    y = positions[
        :,
        :,
        1
    ]

    z = positions[
        :,
        :,
        2
    ]

    horizontal = np.hypot(
        x,
        y,
    )

    if np.any(
        horizontal
        <=
        1e-9
    ):
        raise ValueError(
            "At least one MC receiver sample "
            "has undefined azimuth."
        )

    range_m = np.sqrt(
        x * x
        +
        y * y
        +
        z * z
    )

    azimuth = np.arctan2(
        y,
        x,
    )

    elevation = np.arctan2(
        z,
        horizontal,
    )

    return (
        range_m,
        azimuth,
        elevation,
    )


def _sample_hash(
    *arrays: np.ndarray,
) -> str:
    digest = sha256()

    for array in arrays:
        contiguous = np.ascontiguousarray(
            array,
            dtype=np.float64,
        )

        digest.update(
            str(
                contiguous.shape
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            contiguous.tobytes(
                order="C"
            )
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def deterministic_receiver_angular_posterior(
    *,
    trajectory_mean_h0_m,
    raw_trajectory_covariance_h0_m2,
    current_position_h0_m,
    current_heading_h0_rad: float,
    receiver_geometry_mode: str,
    receiver_offset: ReceiverOffsetGaussianBody | None,
    sample_count: int,
    seed: int = MONTE_CARLO_BASE_SEED,
    low_speed_threshold_mps: float = (
        LOW_SPEED_HEADING_THRESHOLD_MPS
    ),
) -> ReceiverAngularMonteCarloPosterior:
    """
    Primary Stage5 receiver-aware angular posterior.

    Pipeline:

      frozen Gaussian mean/cov
      → frozen covariance calibration
      → product-of-horizon Gaussian sampling
      → samplewise predicted trajectory heading
      → shared receiver placement draw
      → receiver position in H0
      → range/azimuth/elevation samples
    """

    if (
        receiver_geometry_mode
        not in
        RECEIVER_GEOMETRY_MODES
    ):
        raise ValueError(
            "Unsupported receiver geometry mode."
        )

    mean = _finite_horizon_positions(
        trajectory_mean_h0_m,
        name="trajectory_mean_h0_m",
    )

    calibrated_covariance = (
        calibrate_stage4_horizon_covariances(
            raw_trajectory_covariance_h0_m2
        )
    )

    sample_count = int(
        sample_count
    )

    if (
        sample_count
        <
        MONTE_CARLO_MINIMUM_SAMPLE_COUNT
    ):
        raise ValueError(
            "sample_count is below the frozen "
            "Stage5 minimum probability-resolution "
            "requirement."
        )

    trajectory_samples = (
        sample_product_of_horizon_gaussians(
            trajectory_mean_h0_m=mean,

            calibrated_covariance_h0_m2=(
                calibrated_covariance
            ),

            sample_count=sample_count,

            seed=int(
                seed
            ),
        )
    )

    (
        heading_samples,
        fallback_mask,
    ) = resolve_samplewise_headings(
        trajectory_samples_h0_m=(
            trajectory_samples
        ),

        current_position_h0_m=(
            current_position_h0_m
        ),

        current_heading_h0_rad=(
            current_heading_h0_rad
        ),

        low_speed_threshold_mps=(
            low_speed_threshold_mps
        ),
    )

    if (
        receiver_geometry_mode
        ==
        "centroid_baseline"
    ):
        receiver_positions = (
            trajectory_samples.copy()
        )

    else:
        if receiver_offset is None:
            raise ValueError(
                "known/uncertain receiver mode "
                "requires receiver_offset."
            )

        offset_mean = _finite_vector3(
            receiver_offset.mean_body_m,
            name="receiver_offset.mean_body_m",
        )

        offset_covariance = (
            _psd_covariance3(
                receiver_offset.covariance_body_m2,
                name=(
                    "receiver_offset."
                    "covariance_body_m2"
                ),
            )
        )

        if (
            receiver_geometry_mode
            ==
            "known_receiver_offset"
        ):
            body_offsets = np.repeat(
                offset_mean[
                    None,
                    :
                ],
                sample_count,
                axis=0,
            )

        else:
            #
            # Separate RNG stream from trajectory sampling.
            #
            # One physical receiver-placement draw is shared
            # across all future horizons of the same MC sample.
            #
            generator = np.random.Generator(
                np.random.PCG64(
                    int(
                        seed
                    )
                    +
                    1_000_003
                )
            )

            z = generator.standard_normal(
                (
                    sample_count,
                    3,
                ),
                dtype=np.float64,
            )

            body_offsets = (
                _sample_gaussian3(
                    mean=offset_mean,

                    covariance=(
                        offset_covariance
                    ),

                    standard_normal_samples=z,
                )
            )

        offset_h0 = _yaw_rotate_samples(
            body_offsets,
            heading_samples,
        )

        receiver_positions = (
            trajectory_samples
            +
            offset_h0
        )

    (
        range_samples,
        azimuth_samples,
        elevation_samples,
    ) = _angular_samples_from_positions(
        receiver_positions
    )

    summaries = []

    for horizon_index, horizon in enumerate(
        HORIZONS_S
    ):
        ranges = range_samples[
            :,
            horizon_index
        ]

        azimuths = azimuth_samples[
            :,
            horizon_index
        ]

        elevations = elevation_samples[
            :,
            horizon_index
        ]

        range_mean = float(
            np.mean(
                ranges
            )
        )

        azimuth_mean = (
            _circular_mean(
                azimuths
            )
        )

        elevation_mean = float(
            np.mean(
                elevations
            )
        )

        residuals = np.empty(
            (
                sample_count,
                3,
            ),
            dtype=np.float64,
        )

        residuals[
            :,
            0
        ] = (
            ranges
            -
            range_mean
        )

        residuals[
            :,
            1
        ] = np.asarray(
            [
                circular_angle_difference(
                    float(
                        angle
                    ),
                    azimuth_mean,
                )
                for angle in azimuths
            ],
            dtype=np.float64,
        )

        residuals[
            :,
            2
        ] = (
            elevations
            -
            elevation_mean
        )

        covariance_rae = (
            residuals.T
            @
            residuals
            /
            float(
                sample_count
                -
                1
            )
        )

        covariance_rae = (
            0.5
            *
            (
                covariance_rae
                +
                covariance_rae.T
            )
        )

        summaries.append(
            HorizonAngularSummary(
                horizon_s=(
                    float(
                        horizon
                    )
                ),

                range_mean_m=(
                    range_mean
                ),

                range_std_m=(
                    float(
                        np.std(
                            ranges,
                            ddof=1,
                        )
                    )
                ),

                azimuth_circular_mean_rad=(
                    azimuth_mean
                ),

                azimuth_circular_std_rad=(
                    _circular_std(
                        azimuths
                    )
                ),

                elevation_mean_rad=(
                    elevation_mean
                ),

                elevation_std_rad=(
                    float(
                        np.std(
                            elevations,
                            ddof=1,
                        )
                    )
                ),

                covariance_rae=tuple(
                    tuple(
                        float(
                            value
                        )
                        for value in row
                    )
                    for row in covariance_rae
                ),

                heading_fallback_fraction=(
                    float(
                        np.mean(
                            fallback_mask[
                                :,
                                horizon_index
                            ]
                        )
                    )
                ),
            )
        )

    sample_sha = _sample_hash(
        range_samples,
        azimuth_samples,
        elevation_samples,
        heading_samples,
        receiver_positions,
    )

    return ReceiverAngularMonteCarloPosterior(
        range_samples_m=(
            range_samples
        ),

        azimuth_samples_rad=(
            azimuth_samples
        ),

        elevation_samples_rad=(
            elevation_samples
        ),

        heading_samples_rad=(
            heading_samples
        ),

        receiver_position_samples_h0_m=(
            receiver_positions
        ),

        summaries=tuple(
            summaries
        ),

        sample_count=(
            sample_count
        ),

        seed=int(
            seed
        ),

        receiver_geometry_mode=(
            receiver_geometry_mode
        ),

        sample_sha256=(
            sample_sha
        ),
    )


def single_horizon_gaussian_angular_summary(
    *,
    mean_h0_m,
    covariance_h0_m2,
    sample_count: int,
    seed: int,
) -> dict:
    """
    Small numerical-convergence helper used only on frozen
    non-formal Stage4 development predictions.

    This does not perform receiver selection, beam control,
    formal evaluation or model inference.
    """

    mean = _finite_vector3(
        mean_h0_m,
        name="mean_h0_m",
    )

    covariance = _psd_covariance3(
        covariance_h0_m2,
        name="covariance_h0_m2",
    )

    generator = np.random.Generator(
        np.random.PCG64(
            int(
                seed
            )
        )
    )

    z = generator.standard_normal(
        (
            int(
                sample_count
            ),
            3,
        ),
        dtype=np.float64,
    )

    positions = _sample_gaussian3(
        mean=mean,

        covariance=covariance,

        standard_normal_samples=z,
    )

    x = positions[
        :,
        0
    ]

    y = positions[
        :,
        1
    ]

    z_coord = positions[
        :,
        2
    ]

    horizontal = np.hypot(
        x,
        y,
    )

    if np.any(
        horizontal <= 1e-9
    ):
        raise ValueError(
            "Undefined azimuth sample."
        )

    ranges = np.sqrt(
        x * x
        +
        y * y
        +
        z_coord * z_coord
    )

    azimuth = np.arctan2(
        y,
        x,
    )

    elevation = np.arctan2(
        z_coord,
        horizontal,
    )

    return {
        "range_mean_m":
            float(
                np.mean(
                    ranges
                )
            ),

        "range_std_m":
            float(
                np.std(
                    ranges,
                    ddof=1,
                )
            ),

        "azimuth_mean_rad":
            _circular_mean(
                azimuth
            ),

        "azimuth_std_rad":
            _circular_std(
                azimuth
            ),

        "elevation_mean_rad":
            float(
                np.mean(
                    elevation
                )
            ),

        "elevation_std_rad":
            float(
                np.std(
                    elevation,
                    ddof=1,
                )
            ),
    }
