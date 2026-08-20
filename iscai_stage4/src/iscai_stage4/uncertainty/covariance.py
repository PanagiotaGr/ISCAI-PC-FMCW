from __future__ import annotations


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def validate_covariance(
    covariance: Matrix3,
) -> None:
    """
    Validate 3D covariance matrix.

    Conditions:
    - symmetric
    - non-negative diagonal
    """


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


    for i in range(3):

        for j in range(3):

            if abs(
                covariance[i][j]
                -
                covariance[j][i]
            ) > 1e-9:

                raise ValueError(
                    "Covariance not symmetric."
                )



def covariance_trace(
    covariance: Matrix3,
) -> float:
    """
    Total uncertainty magnitude.
    """

    return (
        covariance[0][0]
        +
        covariance[1][1]
        +
        covariance[2][2]
    )
