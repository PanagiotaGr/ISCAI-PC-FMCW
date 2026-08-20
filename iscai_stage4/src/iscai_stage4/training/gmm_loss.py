import math

import torch


def gmm_nll(
    pred,
    target,
    eps=1e-6,
):
    """
    Negative log-likelihood for a mixture of
    correlated bivariate Gaussian trajectories.

    pred["pi"]:
        [B, K]

    pred["mu"]:
        [B, K, T, 2]

    pred["std"]:
        [B, K, T, 2]

    pred["rho"]:
        [B, K, T, 1]

    target:
        [B, T, 2]

    Each mixture component represents a complete
    future trajectory.
    """

    pi = pred["pi"].clamp_min(eps)

    mu = pred["mu"]

    std = pred["std"].clamp_min(eps)

    rho = (
        pred["rho"]
        .squeeze(-1)
        .clamp(
            min=-1.0 + eps,
            max=1.0 - eps,
        )
    )

    # [B,1,T,2]
    target = target.unsqueeze(1)

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
    ).clamp_min(eps)

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

    # Per timestep log-density.
    log_prob_t = (
        -0.5
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

    # A component represents the whole trajectory,
    # therefore joint trajectory log likelihood is
    # the sum across future timesteps.
    log_prob_trajectory = (
        log_prob_t.sum(dim=-1)
    )

    log_mixture = (
        torch.log(pi)
        +
        log_prob_trajectory
    )

    log_likelihood = torch.logsumexp(
        log_mixture,
        dim=1,
    )

    return -log_likelihood.mean()
