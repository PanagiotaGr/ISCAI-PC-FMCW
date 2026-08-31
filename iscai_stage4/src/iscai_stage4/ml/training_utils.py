from __future__ import annotations

import os
import random

import numpy as np
import torch


CLASS_VEHICLE = 0
CLASS_PEDESTRIAN = 1
CLASS_CYCLIST = 2
CLASS_OTHER = 3

CLASS_ID_TO_NAME = {
    CLASS_VEHICLE:
        "TYPE_VEHICLE",

    CLASS_PEDESTRIAN:
        "TYPE_PEDESTRIAN",

    CLASS_CYCLIST:
        "TYPE_CYCLIST",

    CLASS_OTHER:
        "OTHER",
}


def actor_class_id(
    actor_class: str,
) -> int:
    if actor_class == (
        "TYPE_VEHICLE"
    ):
        return CLASS_VEHICLE

    if actor_class == (
        "TYPE_PEDESTRIAN"
    ):
        return CLASS_PEDESTRIAN

    if actor_class == (
        "TYPE_CYCLIST"
    ):
        return CLASS_CYCLIST

    return CLASS_OTHER


def set_global_determinism(
    seed: int,
) -> None:
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.use_deterministic_algorithms(
        True
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def masked_mse_loss(
    prediction,
    target,
    validity_mask,
):
    if prediction.shape != (
        target.shape
    ):
        raise ValueError(
            "prediction/target "
            "shape mismatch."
        )

    if validity_mask.shape != (
        prediction.shape[0],
        prediction.shape[1],
    ):
        raise ValueError(
            "validity-mask shape "
            "mismatch."
        )

    mask = (
        validity_mask
        .to(
            dtype=prediction.dtype
        )
        .unsqueeze(-1)
        .expand_as(
            prediction
        )
    )

    denominator = mask.sum()

    if denominator.item() == 0:
        return (
            prediction.sum()
            *
            0.0
        )

    squared = (
        prediction
        -
        target
    ).square()

    return (
        squared
        *
        mask
    ).sum() / denominator


def balanced_class_weights(
    class_ids,
):
    """
    Fit-sampler weights only.

    Primary vehicle/pedestrian/cyclist
    classes receive equal total sampling
    mass.

    OTHER remains represented but with
    one quarter of a primary class's
    total mass.
    """

    values = torch.as_tensor(
        class_ids,
        dtype=torch.long,
        device="cpu",
    )

    weights = torch.zeros(
        len(values),
        dtype=torch.float64,
    )

    for class_id in (
        CLASS_VEHICLE,
        CLASS_PEDESTRIAN,
        CLASS_CYCLIST,
    ):
        mask = (
            values
            ==
            class_id
        )

        count = int(
            mask.sum().item()
        )

        if count <= 0:
            raise ValueError(
                "Class-aware fit sampler "
                f"is missing class "
                f"{CLASS_ID_TO_NAME[class_id]}."
            )

        weights[
            mask
        ] = (
            1.0
            /
            count
        )

    other_mask = (
        values
        ==
        CLASS_OTHER
    )

    other_count = int(
        other_mask
        .sum()
        .item()
    )

    if other_count > 0:
        weights[
            other_mask
        ] = (
            0.25
            /
            other_count
        )

    if not torch.isfinite(
        weights
    ).all():
        raise RuntimeError(
            "Non-finite sampling weights."
        )

    if weights.sum() <= 0:
        raise RuntimeError(
            "Zero fit-sampling mass."
        )

    return weights
