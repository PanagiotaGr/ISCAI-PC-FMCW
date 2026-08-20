from __future__ import annotations

import math

import numpy as np


def gaussian_log_likelihood(
    *,
    innovation,
    innovation_covariance,
) -> float:
    """
    Multivariate Gaussian log-likelihood:

        log Lambda =
            -1/2 [
                n log(2*pi)
                + log(det(S))
                + nu^T S^-1 nu
            ]

    No truth information.
    """

    nu = np.asarray(
        innovation,
        dtype=np.float64,
    )

    S = np.asarray(
        innovation_covariance,
        dtype=np.float64,
    )

    if nu.ndim != 1:
        raise ValueError(
            "Innovation must be a vector."
        )

    n = nu.shape[0]

    if S.shape != (n, n):
        raise ValueError(
            "Innovation covariance shape mismatch."
        )

    if not (
        np.all(np.isfinite(nu))
        and np.all(np.isfinite(S))
    ):
        raise ValueError(
            "Innovation inputs must be finite."
        )

    if not np.allclose(
        S,
        S.T,
        atol=1e-12,
    ):
        raise ValueError(
            "Innovation covariance must be symmetric."
        )

    sign, logdet = np.linalg.slogdet(
        S
    )

    if sign <= 0:
        raise ValueError(
            "Innovation covariance must be "
            "positive definite."
        )

    try:
        solved = np.linalg.solve(
            S,
            nu,
        )

    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Innovation covariance is singular."
        ) from exc

    mahalanobis2 = float(
        nu @ solved
    )

    return float(
        -0.5
        * (
            n * math.log(
                2.0 * math.pi
            )
            + logdet
            + mahalanobis2
        )
    )


def normalized_likelihoods_from_logs(
    log_likelihoods: tuple[
        float,
        float,
        float,
    ],
) -> tuple[
    float,
    float,
    float,
]:
    """
    Stable conversion of CV/CA/CTRV log-likelihoods
    into positive relative likelihoods.

    Frozen order:

        [CV, CA, CTRV]
    """

    if not all(
        math.isfinite(value)
        for value in log_likelihoods
    ):
        raise ValueError(
            "Log-likelihoods must be finite."
        )

    maximum = max(
        log_likelihoods
    )

    weights = tuple(
        math.exp(
            value - maximum
        )
        for value
        in log_likelihoods
    )

    total = sum(weights)

    if total <= 0.0:
        raise ValueError(
            "Likelihood normalization failed."
        )

    return tuple(
        value / total
        for value
        in weights
    )
