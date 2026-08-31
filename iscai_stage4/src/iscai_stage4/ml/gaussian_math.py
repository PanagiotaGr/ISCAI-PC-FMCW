from __future__ import annotations

import math

import torch


POSITION_DIM = 3

CONFIDENCE_LEVELS = (
    0.50,
    0.80,
    0.90,
    0.95,
    0.99,
)

# Chi-square quantiles for df=3.
# These are squared Mahalanobis-radius
# thresholds of a 3-D Gaussian confidence
# ellipsoid.
CHI_SQUARE_3_THRESHOLDS = {
    0.50:
        2.3659738843753377,

    0.80:
        4.64162767608745,

    0.90:
        6.251388631170325,

    0.95:
        7.814727903251179,

    0.99:
        11.344866730144373,
}


def covariance_from_scale_tril(
    scale_tril: torch.Tensor,
) -> torch.Tensor:
    if scale_tril.shape[-2:] != (
        POSITION_DIM,
        POSITION_DIM,
    ):
        raise ValueError(
            "scale_tril must end "
            "in [3,3]."
        )

    return (
        scale_tril
        @
        scale_tril.transpose(
            -1,
            -2,
        )
    )


def gaussian_nll_per_horizon(
    mean: torch.Tensor,
    scale_tril: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Full 3-D Gaussian negative log likelihood.

    Returns:
        [B,H]
    """

    if mean.shape != target.shape:
        raise ValueError(
            "Gaussian mean/target "
            "shape mismatch."
        )

    if mean.ndim != 3:
        raise ValueError(
            "mean and target must "
            "be [B,H,3]."
        )

    if mean.shape[-1] != (
        POSITION_DIM
    ):
        raise ValueError(
            "Expected 3-D trajectory."
        )

    if scale_tril.shape != (
        mean.shape[0],
        mean.shape[1],
        POSITION_DIM,
        POSITION_DIM,
    ):
        raise ValueError(
            "scale_tril shape mismatch."
        )

    diagonal = torch.diagonal(
        scale_tril,
        dim1=-2,
        dim2=-1,
    )

    if bool(
        torch.any(
            diagonal
            <=
            0.0
        )
    ):
        raise ValueError(
            "Cholesky diagonal must "
            "be strictly positive."
        )

    residual = (
        target
        -
        mean
    )

    solved = (
        torch.linalg
        .solve_triangular(
            scale_tril,
            residual.unsqueeze(
                -1
            ),
            upper=False,
        )
        .squeeze(
            -1
        )
    )

    mahalanobis = (
        solved
        .square()
        .sum(
            dim=-1
        )
    )

    log_determinant = (
        2.0
        *
        torch.log(
            diagonal
        ).sum(
            dim=-1
        )
    )

    constant = (
        POSITION_DIM
        *
        math.log(
            2.0
            *
            math.pi
        )
    )

    return (
        0.5
        *
        (
            constant
            +
            log_determinant
            +
            mahalanobis
        )
    )


def masked_gaussian_nll(
    mean: torch.Tensor,
    scale_tril: torch.Tensor,
    target: torch.Tensor,
    validity_mask: torch.Tensor,
) -> torch.Tensor:
    nll = (
        gaussian_nll_per_horizon(
            mean,
            scale_tril,
            target,
        )
    )

    if validity_mask.shape != (
        nll.shape
    ):
        raise ValueError(
            "Gaussian validity-mask "
            "shape mismatch."
        )

    mask = validity_mask.to(
        dtype=nll.dtype
    )

    denominator = (
        mask.sum()
    )

    if denominator.item() == 0:
        return (
            mean.sum()
            *
            0.0
            +
            scale_tril.sum()
            *
            0.0
        )

    return (
        nll
        *
        mask
    ).sum() / denominator


def mahalanobis_squared(
    mean: torch.Tensor,
    scale_tril: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != target.shape:
        raise ValueError(
            "Mean/target shape mismatch."
        )

    if scale_tril.shape != (
        mean.shape[0],
        mean.shape[1],
        POSITION_DIM,
        POSITION_DIM,
    ):
        raise ValueError(
            "scale_tril shape mismatch."
        )

    residual = (
        target
        -
        mean
    )

    solved = (
        torch.linalg
        .solve_triangular(
            scale_tril,
            residual.unsqueeze(
                -1
            ),
            upper=False,
        )
        .squeeze(
            -1
        )
    )

    return (
        solved
        .square()
        .sum(
            dim=-1
        )
    )


def denormalize_gaussian(
    mean_normalized: torch.Tensor,
    scale_tril_normalized:
        torch.Tensor,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
):
    """
    Transform normalized Gaussian output
    into metric H0-displacement coordinates.

    If:
        y_metric = D y_norm + b

    then:
        mu_metric = D mu_norm + b
        L_metric  = D L_norm
        Sigma_metric =
            D Sigma_norm D^T
    """

    if label_mean.shape != (
        mean_normalized.shape[1],
        POSITION_DIM,
    ):
        raise ValueError(
            "label_mean shape mismatch."
        )

    if label_std.shape != (
        mean_normalized.shape[1],
        POSITION_DIM,
    ):
        raise ValueError(
            "label_std shape mismatch."
        )

    if bool(
        torch.any(
            label_std
            <=
            0.0
        )
    ):
        raise ValueError(
            "label_std must be positive."
        )

    mean_metric = (
        mean_normalized
        *
        label_std.unsqueeze(
            0
        )
        +
        label_mean.unsqueeze(
            0
        )
    )

    scale_metric = (
        scale_tril_normalized
        *
        label_std[
            None,
            :,
            :,
            None,
        ]
    )

    return (
        mean_metric,
        scale_metric,
    )


def empirical_coverage(
    mean: torch.Tensor,
    scale_tril: torch.Tensor,
    target: torch.Tensor,
    validity_mask: torch.Tensor,
):
    squared_distance = (
        mahalanobis_squared(
            mean,
            scale_tril,
            target,
        )
    )

    if validity_mask.shape != (
        squared_distance.shape
    ):
        raise ValueError(
            "Coverage validity-mask "
            "shape mismatch."
        )

    valid = (
        validity_mask
        >
        0.5
    )

    valid_count = int(
        valid.sum().item()
    )

    if valid_count <= 0:
        raise ValueError(
            "Cannot compute coverage "
            "with zero valid targets."
        )

    result = {}

    for level in (
        CONFIDENCE_LEVELS
    ):
        threshold = (
            CHI_SQUARE_3_THRESHOLDS[
                level
            ]
        )

        inside = (
            squared_distance
            <=
            threshold
        )

        count = int(
            (
                inside
                &
                valid
            ).sum().item()
        )

        result[
            f"{level:.2f}"
        ] = {
            "nominal":
                float(
                    level
                ),

            "empirical":
                float(
                    count
                    /
                    valid_count
                ),

            "inside":
                count,

            "valid":
                valid_count,

            "chi_square_3_threshold":
                float(
                    threshold
                ),
        }

    return result
