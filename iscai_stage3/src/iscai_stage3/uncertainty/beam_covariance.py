from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BeamCovariance:
    """
    Covariance in spherical beam coordinates.

    order:
        [range, azimuth, elevation]
    """

    range_variance: float
    azimuth_variance: float
    elevation_variance: float

    covariance_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


def _validate_covariance(
    covariance,
) -> None:

    if len(covariance) != 3:
        raise ValueError(
            "Covariance must be 3x3."
        )

    for row in covariance:
        if len(row) != 3:
            raise ValueError(
                "Covariance must be 3x3."
            )

    for i in range(3):
        if covariance[i][i] < 0:
            raise ValueError(
                "Negative variance."
            )

        for j in range(3):
            if (
                abs(
                    covariance[i][j]
                    -
                    covariance[j][i]
                )
                > 1e-12
            ):
                raise ValueError(
                    "Covariance must be symmetric."
                )


def beam_jacobian(
    position_xyz: tuple[
        float,
        float,
        float,
    ],
):
    """
    Jacobian:

        [r, azimuth, elevation]
              |
              v
        [x,y,z]

    """

    x, y, z = position_xyz

    if not all(
        math.isfinite(v)
        for v in position_xyz
    ):
        raise ValueError(
            "Position must be finite."
        )


    rho2 = x*x + y*y

    r2 = rho2 + z*z

    if r2 <= 0.0:
        raise ValueError(
            "Undefined beam Jacobian."
        )


    r = math.sqrt(r2)

    rho = math.sqrt(rho2)


    if rho == 0.0:
        raise ValueError(
            "Azimuth undefined."
        )


    # dr/dx,y,z
    dr = (
        x/r,
        y/r,
        z/r,
    )


    # daz/dx,y,z
    daz = (
        -y/rho2,
        x/rho2,
        0.0,
    )


    # del/dx,y,z
    den = r2 * rho

    dele = (
        -x*z/den,
        -y*z/den,
        rho/ r2,
    )


    return (
        dr,
        daz,
        dele,
    )


def propagate_cartesian_covariance_to_beam(
    *,
    position_xyz: tuple[
        float,
        float,
        float,
    ],
    covariance_xyz,
) -> BeamCovariance:
    """
    First-order uncertainty propagation:

        Σbeam = J Σxyz Jᵀ
    """

    _validate_covariance(
        covariance_xyz
    )


    J = beam_jacobian(
        position_xyz
    )


    result = []

    for i in range(3):

        row = []

        for j in range(3):

            value = 0.0

            for a in range(3):
                for b in range(3):

                    value += (
                        J[i][a]
                        *
                        covariance_xyz[a][b]
                        *
                        J[j][b]
                    )

            row.append(
                value
            )

        result.append(
            tuple(row)
        )


    matrix = tuple(result)


    return BeamCovariance(
        range_variance=matrix[0][0],
        azimuth_variance=matrix[1][1],
        elevation_variance=matrix[2][2],
        covariance_matrix=matrix,
    )
