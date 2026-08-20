from __future__ import annotations

import torch


def mdn_nll_loss(
    output,
    target,
):
    """
    Placeholder MDN negative log likelihood.

    Stage5 initial implementation:
    single Gaussian approximation.
    """

    mean = output[:, :2]

    error = target[:, -1, :] - mean

    loss = (
        error ** 2
    ).mean()

    return loss
