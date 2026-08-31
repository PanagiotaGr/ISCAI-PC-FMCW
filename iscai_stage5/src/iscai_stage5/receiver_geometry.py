from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


RECEIVER_GEOMETRY_MODES = (
    "centroid_baseline",
    "known_receiver_offset",
    "uncertain_receiver_offset",
)


@dataclass(
    frozen=True
)
class ReceiverOffsetDistribution:
    """
    Receiver placement distribution in the receiver
    vehicle body frame.

    This covariance is receiver-placement uncertainty only.

    It is NOT:
      - Stage2 measurement covariance R_t,
      - Stage4 predictive trajectory covariance.

    Joint trajectory + receiver-placement propagation is
    performed later in the Stage5 angular-posterior layer.
    """

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
class ReceiverGeometryDistributionH0:
    mode: str

    receiver_mean_h0_m: tuple[
        float,
        float,
        float,
    ]

    placement_covariance_h0_m2: tuple[
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
        "receiver_placement_uncertainty_only"
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


def _valid_covariance3(
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
        rtol=0.0,
        atol=1e-12,
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

    #
    # Remove only floating-point anti-symmetry.
    # No negative-eigenvalue repair is performed.
    #
    return (
        0.5
        *
        (
            covariance
            +
            covariance.T
        )
    )


def yaw_rotation_h0_from_body(
    heading_h0_rad: float,
) -> np.ndarray:
    heading = float(
        heading_h0_rad
    )

    if not math.isfinite(
        heading
    ):
        raise ValueError(
            "heading_h0_rad must be finite."
        )

    cosine = math.cos(
        heading
    )

    sine = math.sin(
        heading
    )

    return np.asarray(
        [
            [
                cosine,
                -sine,
                0.0,
            ],
            [
                sine,
                cosine,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )


def receiver_geometry_distribution_h0(
    *,
    actor_center_h0_m,
    heading_h0_rad: float,
    mode: str,
    receiver_offset: (
        ReceiverOffsetDistribution
        |
        None
    ) = None,
) -> ReceiverGeometryDistributionH0:
    """
    Map receiver-body-frame placement into H0.

    The returned covariance represents receiver placement
    uncertainty only.  Stage4 predictive covariance remains
    separate and is combined only by the downstream
    trajectory→receiver/angular posterior propagator.
    """

    if mode not in RECEIVER_GEOMETRY_MODES:
        raise ValueError(
            f"Unsupported receiver geometry mode: {mode!r}"
        )

    center = _finite_vector3(
        actor_center_h0_m,
        name="actor_center_h0_m",
    )

    rotation = yaw_rotation_h0_from_body(
        heading_h0_rad
    )

    zero_covariance = np.zeros(
        (
            3,
            3,
        ),
        dtype=np.float64,
    )

    if mode == "centroid_baseline":
        mean = center.copy()

        covariance = zero_covariance

    else:
        if receiver_offset is None:
            raise ValueError(
                "known/uncertain receiver geometry "
                "requires receiver_offset."
            )

        offset_mean = _finite_vector3(
            receiver_offset.mean_body_m,
            name="receiver_offset.mean_body_m",
        )

        offset_covariance = (
            _valid_covariance3(
                receiver_offset.covariance_body_m2,
                name=(
                    "receiver_offset."
                    "covariance_body_m2"
                ),
            )
        )

        mean = (
            center
            +
            rotation
            @
            offset_mean
        )

        if mode == "known_receiver_offset":
            #
            # The receiver location is treated as known.
            # Offset covariance therefore does not enter
            # the controller posterior for this baseline.
            #
            covariance = zero_covariance

        elif mode == "uncertain_receiver_offset":
            covariance = (
                rotation
                @
                offset_covariance
                @
                rotation.T
            )

            covariance = (
                0.5
                *
                (
                    covariance
                    +
                    covariance.T
                )
            )

        else:
            raise AssertionError(
                "Unreachable receiver geometry mode."
            )

    return ReceiverGeometryDistributionH0(
        mode=mode,

        receiver_mean_h0_m=tuple(
            float(
                value
            )
            for value in mean
        ),

        placement_covariance_h0_m2=tuple(
            tuple(
                float(
                    value
                )
                for value in row
            )
            for row in covariance
        ),
    )
