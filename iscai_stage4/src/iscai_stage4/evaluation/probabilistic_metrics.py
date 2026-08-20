from __future__ import annotations

import math


def gaussian_nll(
    *,
    error_vector,
    covariance,
):
    """
    Negative log likelihood
    for 3D Gaussian.
    """

    x, y, z = error_vector

    var_x = covariance[0][0]
    var_y = covariance[1][1]
    var_z = covariance[2][2]


    if (
        var_x <= 0
        or var_y <= 0
        or var_z <= 0
    ):
        raise ValueError(
            "Invalid covariance."
        )


    mahalanobis = (
        x*x / var_x
        +
        y*y / var_y
        +
        z*z / var_z
    )


    log_det = math.log(
        var_x * var_y * var_z
    )


    return 0.5 * (
        mahalanobis
        +
        log_det
        +
        3 * math.log(
            2 * math.pi
        )
    )



def mahalanobis_distance(
    *,
    error_vector,
    covariance,
):

    x, y, z = error_vector


    return math.sqrt(
        x*x / covariance[0][0]
        +
        y*y / covariance[1][1]
        +
        z*z / covariance[2][2]
    )



def empirical_coverage(
    *,
    mahalanobis_values,
    threshold,
):

    if len(
        mahalanobis_values
    ) == 0:
        raise ValueError(
            "Empty samples."
        )


    inside = sum(
        1
        for value in mahalanobis_values
        if value <= threshold
    )


    return (
        inside
        /
        len(mahalanobis_values)
    )
