from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


ANGULAR_COORDINATE_ORDER = (
    "range_m",
    "azimuth_rad",
    "elevation_rad",
)

ANGULAR_FRAME = (
    "headlamp_frame_H0"
)

PRIMARY_STAGE52_METHOD = (
    "deterministic_monte_carlo"
)

ANALYTIC_CROSSCHECK_METHOD = (
    "first_order_Jacobian_Gaussian"
)

ANALYTIC_CROSSCHECK_SCOPE = (
    "fixed_heading_known_receiver_geometry_or_"
    "already_resolved_receiver_position_Gaussian"
)

TRAJECTORY_AND_PLACEMENT_COVARIANCE_SEMANTICS = (
    "distinct_then_combined_only_at_receiver_position_level"
)


@dataclass(
    frozen=True
)
class GaussianPositionH0:
    """
    Gaussian receiver-position approximation in H0.

    covariance_h0_m2 may represent:

      1. Stage4 predictive trajectory covariance only, or
      2. trajectory covariance + receiver-placement
         covariance after an explicitly valid independent
         position-level combination.

    It must never be interpreted as Stage2 measurement R_t.
    """

    mean_h0_m: tuple[
        float,
        float,
        float,
    ]

    covariance_h0_m2: tuple[
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

    covariance_semantics: str = (
        "predictive_receiver_position_uncertainty"
    )


@dataclass(
    frozen=True
)
class RangeAzimuthElevationPoint:
    range_m: float

    azimuth_rad: float

    elevation_rad: float

    frame: str = (
        ANGULAR_FRAME
    )


@dataclass(
    frozen=True
)
class LinearizedAngularGaussian:
    """
    First-order Gaussian approximation in:

        [range, azimuth, elevation]

    This object is an analytic numerical cross-check.

    The primary Stage5 posterior will be deterministic
    Monte Carlo in Block5.2 Part2.
    """

    mean: tuple[
        float,
        float,
        float,
    ]

    covariance: tuple[
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

    method: str = (
        ANALYTIC_CROSSCHECK_METHOD
    )

    coordinate_order: tuple[
        str,
        str,
        str,
    ] = (
        ANGULAR_COORDINATE_ORDER
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
    ) < -1e-12:
        raise ValueError(
            f"{name} must be positive semidefinite."
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


def circular_angle_difference(
    first_rad: float,
    second_rad: float,
) -> float:
    """
    Wrapped signed difference:

        first - second

    in [-pi, pi).
    """

    first = float(
        first_rad
    )

    second = float(
        second_rad
    )

    if not (
        math.isfinite(
            first
        )
        and
        math.isfinite(
            second
        )
    ):
        raise ValueError(
            "Angles must be finite."
        )

    return (
        (
            first
            -
            second
            +
            math.pi
        )
        %
        (
            2.0
            *
            math.pi
        )
        -
        math.pi
    )


def position_h0_to_range_azimuth_elevation(
    position_h0_m,
    *,
    minimum_horizontal_range_m: float = 1e-9,
) -> RangeAzimuthElevationPoint:
    """
    Convert receiver position in H0 into:

        range
        azimuth
        elevation

    Definitions:

        r = sqrt(x^2 + y^2 + z^2)

        az = atan2(y, x)

        el = atan2(z, sqrt(x^2 + y^2))

    A near-zero horizontal range is rejected because the
    azimuth becomes numerically/physically undefined.
    """

    position = _finite_vector3(
        position_h0_m,
        name="position_h0_m",
    )

    x, y, z = (
        float(
            value
        )
        for value in position
    )

    horizontal = math.hypot(
        x,
        y,
    )

    minimum_horizontal = float(
        minimum_horizontal_range_m
    )

    if (
        not math.isfinite(
            minimum_horizontal
        )
        or
        minimum_horizontal <= 0.0
    ):
        raise ValueError(
            "minimum_horizontal_range_m must "
            "be positive and finite."
        )

    if horizontal <= minimum_horizontal:
        raise ValueError(
            "Receiver horizontal range is too "
            "small for a well-defined azimuth."
        )

    distance = math.sqrt(
        x * x
        +
        y * y
        +
        z * z
    )

    azimuth = math.atan2(
        y,
        x,
    )

    elevation = math.atan2(
        z,
        horizontal,
    )

    return RangeAzimuthElevationPoint(
        range_m=(
            float(
                distance
            )
        ),

        azimuth_rad=(
            float(
                azimuth
            )
        ),

        elevation_rad=(
            float(
                elevation
            )
        ),
    )


def range_azimuth_elevation_jacobian(
    position_h0_m,
    *,
    minimum_horizontal_range_m: float = 1e-9,
) -> np.ndarray:
    """
    Jacobian:

        d [r, az, el]
        --------------
          d [x, y, z]

    evaluated in H0.
    """

    position = _finite_vector3(
        position_h0_m,
        name="position_h0_m",
    )

    x, y, z = (
        float(
            value
        )
        for value in position
    )

    horizontal_squared = (
        x * x
        +
        y * y
    )

    horizontal = math.sqrt(
        horizontal_squared
    )

    minimum_horizontal = float(
        minimum_horizontal_range_m
    )

    if (
        not math.isfinite(
            minimum_horizontal
        )
        or
        minimum_horizontal <= 0.0
    ):
        raise ValueError(
            "minimum_horizontal_range_m must "
            "be positive and finite."
        )

    if horizontal <= minimum_horizontal:
        raise ValueError(
            "Receiver horizontal range is too "
            "small for a stable angular Jacobian."
        )

    range_squared = (
        horizontal_squared
        +
        z * z
    )

    distance = math.sqrt(
        range_squared
    )

    if distance <= 0.0:
        raise ValueError(
            "Receiver range must be positive."
        )

    jacobian = np.asarray(
        [
            [
                x
                /
                distance,

                y
                /
                distance,

                z
                /
                distance,
            ],

            [
                -y
                /
                horizontal_squared,

                x
                /
                horizontal_squared,

                0.0,
            ],

            [
                -z
                *
                x
                /
                (
                    horizontal
                    *
                    range_squared
                ),

                -z
                *
                y
                /
                (
                    horizontal
                    *
                    range_squared
                ),

                horizontal
                /
                range_squared,
            ],
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            jacobian
        )
    ):
        raise ValueError(
            "Angular Jacobian is non-finite."
        )

    return jacobian


def linearized_position_gaussian_to_angular(
    gaussian_position: GaussianPositionH0,
) -> LinearizedAngularGaussian:
    """
    First-order covariance propagation:

        Sigma_rae = J Sigma_xyz J^T

    This is a cross-check path only.

    It does not replace the primary deterministic Monte
    Carlo posterior that will be implemented/frozen in
    Block5.2 Part2.
    """

    mean = _finite_vector3(
        gaussian_position.mean_h0_m,
        name="gaussian_position.mean_h0_m",
    )

    covariance = _psd_covariance3(
        gaussian_position.covariance_h0_m2,
        name=(
            "gaussian_position."
            "covariance_h0_m2"
        ),
    )

    angular_mean = (
        position_h0_to_range_azimuth_elevation(
            mean
        )
    )

    jacobian = (
        range_azimuth_elevation_jacobian(
            mean
        )
    )

    angular_covariance = (
        jacobian
        @
        covariance
        @
        jacobian.T
    )

    angular_covariance = (
        0.5
        *
        (
            angular_covariance
            +
            angular_covariance.T
        )
    )

    eigenvalues = np.linalg.eigvalsh(
        angular_covariance
    )

    if float(
        np.min(
            eigenvalues
        )
    ) < -1e-10:
        raise ValueError(
            "Propagated angular covariance "
            "is not positive semidefinite."
        )

    return LinearizedAngularGaussian(
        mean=(
            float(
                angular_mean.range_m
            ),
            float(
                angular_mean.azimuth_rad
            ),
            float(
                angular_mean.elevation_rad
            ),
        ),

        covariance=tuple(
            tuple(
                float(
                    value
                )
                for value in row
            )
            for row in angular_covariance
        ),
    )


def combine_independent_position_uncertainties(
    *,
    trajectory_mean_h0_m,
    trajectory_covariance_h0_m2,
    receiver_offset_mean_h0_m,
    receiver_placement_covariance_h0_m2,
) -> GaussianPositionH0:
    """
    Position-level analytic combination:

        mu_RX = mu_traj + mu_offset

        Sigma_RX =
            Sigma_predictive
            +
            Sigma_receiver_placement

    Assumption:
        conditional independence of Stage4 predictive
        position uncertainty and receiver-placement
        uncertainty at this fixed-orientation analytic
        cross-check.

    This helper does NOT model uncertainty of future
    receiver heading.  Sample-dependent heading rotation
    belongs to the primary Monte Carlo path in Part2.
    """

    trajectory_mean = _finite_vector3(
        trajectory_mean_h0_m,
        name="trajectory_mean_h0_m",
    )

    trajectory_covariance = (
        _psd_covariance3(
            trajectory_covariance_h0_m2,
            name=(
                "trajectory_covariance_h0_m2"
            ),
        )
    )

    offset_mean = _finite_vector3(
        receiver_offset_mean_h0_m,
        name="receiver_offset_mean_h0_m",
    )

    placement_covariance = (
        _psd_covariance3(
            receiver_placement_covariance_h0_m2,
            name=(
                "receiver_placement_"
                "covariance_h0_m2"
            ),
        )
    )

    receiver_mean = (
        trajectory_mean
        +
        offset_mean
    )

    receiver_covariance = (
        trajectory_covariance
        +
        placement_covariance
    )

    return GaussianPositionH0(
        mean_h0_m=tuple(
            float(
                value
            )
            for value in receiver_mean
        ),

        covariance_h0_m2=tuple(
            tuple(
                float(
                    value
                )
                for value in row
            )
            for row in receiver_covariance
        ),

        covariance_semantics=(
            "Stage4_predictive_plus_receiver_"
            "placement_fixed_orientation_analytic"
        ),
    )
