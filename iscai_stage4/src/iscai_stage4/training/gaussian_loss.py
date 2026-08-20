import math

import torch


def gaussian_nll(
    pred,
    target,
    eps=1e-6,
):
    """
    Negative log-likelihood of a correlated
    bivariate Gaussian trajectory prediction.

    pred["mu"]:
        [..., 2]

    pred["std"]:
        [..., 2]

    pred["rho"]:
        [..., 1]

    target:
        [..., 2]
    """

    mu = pred["mu"]

    std = pred["std"].clamp_min(
        eps
    )

    rho = (
        pred["rho"]
        .squeeze(-1)
        .clamp(
            min=-1.0 + eps,
            max=1.0 - eps,
        )
    )

    dx = (
        target[..., 0]
        -
        mu[..., 0]
    )

    dy = (
        target[..., 1]
        -
        mu[..., 1]
    )

    sx = std[..., 0]
    sy = std[..., 1]

    nx = dx / sx
    ny = dy / sy

    one_minus_rho2 = (
        1.0
        -
        rho * rho
    ).clamp_min(
        eps
    )

    mahalanobis = (
        nx * nx
        +
        ny * ny
        -
        2.0
        *
        rho
        *
        nx
        *
        ny
    ) / one_minus_rho2

    log_det_term = (
        2.0 * torch.log(sx)
        +
        2.0 * torch.log(sy)
        +
        torch.log(one_minus_rho2)
    )

    nll = (
        0.5
        *
        (
            mahalanobis
            +
            log_det_term
            +
            2.0
            *
            math.log(
                2.0 * math.pi
            )
        )
    )

    return nll.mean()
