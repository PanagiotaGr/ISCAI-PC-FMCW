from __future__ import annotations

from typing import Tuple


Matrix3 = Tuple[
    Tuple[float,float,float],
    Tuple[float,float,float],
    Tuple[float,float,float],
]


Matrix4 = Tuple[
    Tuple[float,...],
    Tuple[float,...],
    Tuple[float,...],
    Tuple[float,...],
]


def mat_mul(
    A,
    B,
):
    rows = len(A)
    cols = len(B[0])

    out = []

    for i in range(rows):
        row = []

        for j in range(cols):
            value = sum(
                A[i][k] * B[k][j]
                for k in range(len(B))
            )

            row.append(value)

        out.append(tuple(row))

    return tuple(out)



def transpose(A):

    return tuple(
        tuple(
            A[j][i]
            for j in range(len(A))
        )
        for i in range(len(A[0]))
    )



def validate_covariance(
    covariance,
    tolerance=1e-10,
):
    """
    Minimal covariance contract.

    Checks:
    - square
    - symmetric
    - diagonal non-negative
    """

    n = len(covariance)

    if any(len(row)!=n for row in covariance):
        raise ValueError(
            "Covariance is not square."
        )


    for i in range(n):

        if covariance[i][i] < -tolerance:
            raise ValueError(
                "Negative variance."
            )

        for j in range(n):

            if abs(
                covariance[i][j]
                -
                covariance[j][i]
            ) > tolerance:

                raise ValueError(
                    "Covariance not symmetric."
                )



def propagate_position_covariance(
    jacobian,
    measurement_covariance,
):
    """
    Sigma_x = J Sigma_z J^T
    """

    validate_covariance(
        measurement_covariance
    )

    JT = transpose(jacobian)

    return mat_mul(
        mat_mul(
            jacobian,
            measurement_covariance,
        ),
        JT,
    )
