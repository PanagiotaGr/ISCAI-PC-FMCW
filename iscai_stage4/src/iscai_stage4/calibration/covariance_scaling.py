from __future__ import annotations


def scale_covariance(
    covariance,
    alpha: float,
):

    if alpha <= 0:
        raise ValueError(
            "Alpha must be positive."
        )


    return tuple(
        tuple(
            value * alpha
            for value in row
        )
        for row in covariance
    )
