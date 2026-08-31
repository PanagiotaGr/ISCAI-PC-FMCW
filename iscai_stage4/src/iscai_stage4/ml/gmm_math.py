from __future__ import annotations

import torch

from .gaussian_math import (
    gaussian_nll_per_horizon,
)


def component_gaussian_nll_per_horizon(
    means: torch.Tensor,
    scale_tril: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Returns component Gaussian NLL:

        [B,K,H]

    target:
        [B,H,3]
    """

    if means.ndim != 4:
        raise ValueError(
            "means must be [B,K,H,3]."
        )

    if target.ndim != 3:
        raise ValueError(
            "target must be [B,H,3]."
        )

    batch, components, horizons, dim = (
        means.shape
    )

    if target.shape != (
        batch,
        horizons,
        dim,
    ):
        raise ValueError(
            "GMM target shape mismatch."
        )

    if scale_tril.shape != (
        batch,
        components,
        horizons,
        dim,
        dim,
    ):
        raise ValueError(
            "GMM scale_tril shape mismatch."
        )

    expanded_target = (
        target[
            :,
            None,
            :,
            :
        ]
        .expand(
            batch,
            components,
            horizons,
            dim,
        )
        .reshape(
            batch * components,
            horizons,
            dim,
        )
    )

    flattened_means = means.reshape(
        batch * components,
        horizons,
        dim,
    )

    flattened_scale = (
        scale_tril.reshape(
            batch * components,
            horizons,
            dim,
            dim,
        )
    )

    nll = gaussian_nll_per_horizon(
        flattened_means,
        flattened_scale,
        expanded_target,
    )

    return nll.reshape(
        batch,
        components,
        horizons,
    )


def gmm_joint_nll_per_sample(
    mixture_logits: torch.Tensor,
    means: torch.Tensor,
    scale_tril: torch.Tensor,
    target: torch.Tensor,
    validity_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Joint trajectory GMM NLL.

    A single mixture mode applies to the complete
    trajectory. Component likelihood is therefore
    the product of valid horizon likelihoods:

      log p_k(Y) =
          Σ_h valid_h log N(y_h | μ_kh, Σ_kh)

    Mixture:
      p(Y) = Σ_k π_k p_k(Y)

    Returns:
      [B]

    Samples with zero valid horizons return zero
    here and are removed by masked_gmm_joint_nll().
    """

    if validity_mask.shape != (
        target.shape[0],
        target.shape[1],
    ):
        raise ValueError(
            "GMM validity-mask shape mismatch."
        )

    valid = (
        validity_mask
        >
        0.5
    )

    safe_target = torch.where(
        valid.unsqueeze(-1),
        target,
        torch.zeros_like(
            target
        ),
    )

    component_nll = (
        component_gaussian_nll_per_horizon(
            means,
            scale_tril,
            safe_target,
        )
    )

    mask = (
        valid
        .to(
            dtype=component_nll.dtype
        )
        .unsqueeze(1)
    )

    component_log_likelihood = (
        -(
            component_nll
            *
            mask
        ).sum(
            dim=-1
        )
    )

    if mixture_logits.shape != (
        means.shape[0],
        means.shape[1],
    ):
        raise ValueError(
            "Mixture-logit shape mismatch."
        )

    log_weights = torch.log_softmax(
        mixture_logits,
        dim=-1,
    )

    mixture_log_likelihood = (
        torch.logsumexp(
            log_weights
            +
            component_log_likelihood,
            dim=-1,
        )
    )

    sample_nll = (
        -mixture_log_likelihood
    )

    active = valid.any(
        dim=-1
    )

    return torch.where(
        active,
        sample_nll,
        torch.zeros_like(
            sample_nll
        ),
    )


def masked_gmm_joint_nll(
    mixture_logits: torch.Tensor,
    means: torch.Tensor,
    scale_tril: torch.Tensor,
    target: torch.Tensor,
    validity_mask: torch.Tensor,
) -> torch.Tensor:
    sample_nll = (
        gmm_joint_nll_per_sample(
            mixture_logits,
            means,
            scale_tril,
            target,
            validity_mask,
        )
    )

    active = (
        validity_mask
        >
        0.5
    ).any(
        dim=-1
    )

    count = int(
        active.sum().item()
    )

    if count == 0:
        return (
            mixture_logits.sum()
            *
            0.0
            +
            means.sum()
            *
            0.0
            +
            scale_tril.sum()
            *
            0.0
        )

    return sample_nll[
        active
    ].mean()


def mixture_probabilities(
    mixture_logits: torch.Tensor,
) -> torch.Tensor:
    if mixture_logits.ndim != 2:
        raise ValueError(
            "Mixture logits must be [B,K]."
        )

    return torch.softmax(
        mixture_logits,
        dim=-1,
    )


def mixture_mean(
    mixture_logits: torch.Tensor,
    means: torch.Tensor,
) -> torch.Tensor:
    """
    Probability-weighted trajectory mean.

    Returns:
        [B,H,3]
    """

    if means.ndim != 4:
        raise ValueError(
            "means must be [B,K,H,3]."
        )

    probabilities = mixture_probabilities(
        mixture_logits
    )

    if probabilities.shape != (
        means.shape[0],
        means.shape[1],
    ):
        raise ValueError(
            "Mixture probability shape mismatch."
        )

    return (
        probabilities[
            :,
            :,
            None,
            None
        ]
        *
        means
    ).sum(
        dim=1
    )


def map_component_index(
    mixture_logits: torch.Tensor,
) -> torch.Tensor:
    return torch.argmax(
        mixture_logits,
        dim=-1,
    )


def map_component_mean(
    mixture_logits: torch.Tensor,
    means: torch.Tensor,
) -> torch.Tensor:
    if means.ndim != 4:
        raise ValueError(
            "means must be [B,K,H,3]."
        )

    index = map_component_index(
        mixture_logits
    )

    batch = torch.arange(
        means.shape[0],
        device=means.device,
    )

    return means[
        batch,
        index,
        :,
        :
    ]
