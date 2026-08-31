from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from iscai_stage4.data.neural_inputs import (
    FEATURE_DIM,
    MAP_CONTEXT_DIM,
    MAX_NEIGHBORS,
)


PREDICTION_HORIZONS = 4
POSITION_DIM = 3

CHOLESKY_PARAMETERS_PER_HORIZON = 6

MIN_PREDICTIVE_STD_NORMALIZED = 1e-3

DEFAULT_INITIAL_STD_NORMALIZED = 0.5


@dataclass(frozen=True)
class GaussianTrajectoryOutput:
    """
    Predictive distribution in normalized
    future-displacement coordinates.

    This predictive covariance is NOT
    Stage2 measurement covariance R_t.

    Stage2 R_t remains part of the model
    input history. scale_tril below is an
    output of the learned predictor.
    """

    mean: torch.Tensor

    scale_tril: torch.Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 3:
            raise ValueError(
                "Gaussian mean must be "
                "[B,H,3]."
            )

        if self.mean.shape[1:] != (
            PREDICTION_HORIZONS,
            POSITION_DIM,
        ):
            raise ValueError(
                "Gaussian mean shape changed."
            )

        if self.scale_tril.ndim != 4:
            raise ValueError(
                "scale_tril must be "
                "[B,H,3,3]."
            )

        if self.scale_tril.shape != (
            self.mean.shape[0],
            PREDICTION_HORIZONS,
            POSITION_DIM,
            POSITION_DIM,
        ):
            raise ValueError(
                "Gaussian scale_tril "
                "shape changed."
            )

    @property
    def covariance(
        self,
    ) -> torch.Tensor:
        return (
            self.scale_tril
            @
            self.scale_tril.transpose(
                -1,
                -2,
            )
        )


def _positive_diagonal(
    raw: torch.Tensor,
) -> torch.Tensor:
    return (
        F.softplus(
            raw
        )
        +
        MIN_PREDICTIVE_STD_NORMALIZED
    )


def scale_tril_from_raw(
    raw: torch.Tensor,
) -> torch.Tensor:
    """
    raw:
        [..., 6]

    Parameter order:
        L11,
        L21,
        L22,
        L31,
        L32,
        L33

    Positive diagonal is guaranteed by:
        softplus(raw_diag) + floor
    """

    if raw.shape[-1] != (
        CHOLESKY_PARAMETERS_PER_HORIZON
    ):
        raise ValueError(
            "Expected six Cholesky "
            "parameters per horizon."
        )

    l11 = _positive_diagonal(
        raw[..., 0]
    )

    l21 = raw[..., 1]

    l22 = _positive_diagonal(
        raw[..., 2]
    )

    l31 = raw[..., 3]

    l32 = raw[..., 4]

    l33 = _positive_diagonal(
        raw[..., 5]
    )

    zero = torch.zeros_like(
        l11
    )

    row0 = torch.stack(
        (
            l11,
            zero,
            zero,
        ),
        dim=-1,
    )

    row1 = torch.stack(
        (
            l21,
            l22,
            zero,
        ),
        dim=-1,
    )

    row2 = torch.stack(
        (
            l31,
            l32,
            l33,
        ),
        dim=-1,
    )

    return torch.stack(
        (
            row0,
            row1,
            row2,
        ),
        dim=-2,
    )


class GaussianTrajectoryGRU(
    nn.Module
):
    """
    Probabilistic extension of the frozen
    Block4.3 deterministic architecture.

    Input:
        same causal target history,
        same Stage2 R_t conditioning,
        same neighbours,
        same map context.

    Output per horizon:
        predictive mean in R^3
        full 3x3 SPD predictive covariance.
    """

    def __init__(
        self,
        *,
        target_hidden_dim: int = 64,
        neighbor_hidden_dim: int = 32,
        map_hidden_dim: int = 32,
        fusion_hidden_dim: int = 128,
        use_neighbors: bool = True,
        use_map: bool = True,
    ):
        super().__init__()

        if min(
            target_hidden_dim,
            neighbor_hidden_dim,
            map_hidden_dim,
            fusion_hidden_dim,
        ) <= 0:
            raise ValueError(
                "Hidden dimensions "
                "must be positive."
            )

        self.use_neighbors = bool(
            use_neighbors
        )

        self.use_map = bool(
            use_map
        )

        self.target_gru = nn.GRU(
            input_size=FEATURE_DIM,
            hidden_size=(
                target_hidden_dim
            ),
            num_layers=1,
            batch_first=True,
        )

        self.neighbor_gru = nn.GRU(
            input_size=FEATURE_DIM,
            hidden_size=(
                neighbor_hidden_dim
            ),
            num_layers=1,
            batch_first=True,
        )

        self.map_encoder = (
            nn.Sequential(
                nn.Linear(
                    MAP_CONTEXT_DIM,
                    map_hidden_dim,
                ),
                nn.ReLU(),

                nn.Linear(
                    map_hidden_dim,
                    map_hidden_dim,
                ),
                nn.ReLU(),
            )
        )

        fusion_input_dim = (
            target_hidden_dim
            +
            neighbor_hidden_dim
            +
            map_hidden_dim
        )

        self.fusion = (
            nn.Sequential(
                nn.Linear(
                    fusion_input_dim,
                    fusion_hidden_dim,
                ),
                nn.ReLU(),

                nn.Linear(
                    fusion_hidden_dim,
                    64,
                ),
                nn.ReLU(),
            )
        )

        self.mean_head = nn.Linear(
            64,
            PREDICTION_HORIZONS
            *
            POSITION_DIM,
        )

        self.cholesky_head = nn.Linear(
            64,
            PREDICTION_HORIZONS
            *
            CHOLESKY_PARAMETERS_PER_HORIZON,
        )

    def _encode(
        self,
        target_history,
        neighbor_histories,
        neighbor_mask,
        map_context,
    ):
        if target_history.ndim != 3:
            raise ValueError(
                "target_history must be "
                "[B,T,F]."
            )

        if neighbor_histories.ndim != 4:
            raise ValueError(
                "neighbor_histories must "
                "be [B,N,T,F]."
            )

        batch = int(
            target_history.shape[0]
        )

        if target_history.shape[-1] != (
            FEATURE_DIM
        ):
            raise ValueError(
                "Target feature dimension "
                "changed."
            )

        if (
            neighbor_histories.shape[0]
            !=
            batch
        ):
            raise ValueError(
                "Batch-size mismatch."
            )

        if neighbor_histories.shape[1] != (
            MAX_NEIGHBORS
        ):
            raise ValueError(
                "Neighbour slot count changed."
            )

        if neighbor_histories.shape[-1] != (
            FEATURE_DIM
        ):
            raise ValueError(
                "Neighbour feature dimension "
                "changed."
            )

        if neighbor_mask.shape != (
            batch,
            MAX_NEIGHBORS,
        ):
            raise ValueError(
                "neighbor_mask shape changed."
            )

        if map_context.shape != (
            batch,
            MAP_CONTEXT_DIM,
        ):
            raise ValueError(
                "map_context shape changed."
            )

        _, target_hidden = (
            self.target_gru(
                target_history
            )
        )

        target_embedding = (
            target_hidden[-1]
        )

        if self.use_neighbors:
            neighbor_flat = (
                neighbor_histories
                .reshape(
                    batch
                    *
                    MAX_NEIGHBORS,
                    neighbor_histories
                    .shape[2],
                    neighbor_histories
                    .shape[3],
                )
            )

            _, neighbor_hidden = (
                self.neighbor_gru(
                    neighbor_flat
                )
            )

            neighbor_embedding = (
                neighbor_hidden[-1]
                .reshape(
                    batch,
                    MAX_NEIGHBORS,
                    -1,
                )
            )

            mask = (
                neighbor_mask
                .to(
                    dtype=(
                        neighbor_embedding
                        .dtype
                    )
                )
                .unsqueeze(-1)
            )

            neighbor_embedding = (
                neighbor_embedding
                *
                mask
            )

            denominator = (
                mask.sum(
                    dim=1
                )
                .clamp_min(
                    1.0
                )
            )

            neighbor_embedding = (
                neighbor_embedding
                .sum(
                    dim=1
                )
                /
                denominator
            )

        else:
            neighbor_embedding = (
                torch.zeros(
                    (
                        batch,
                        self.neighbor_gru
                        .hidden_size,
                    ),
                    dtype=(
                        target_history.dtype
                    ),
                    device=(
                        target_history.device
                    ),
                )
            )

        if self.use_map:
            map_embedding = (
                self.map_encoder(
                    map_context
                )
            )

        else:
            map_embedding = (
                torch.zeros(
                    (
                        batch,
                        self.map_encoder[0]
                        .out_features,
                    ),
                    dtype=(
                        target_history.dtype
                    ),
                    device=(
                        target_history.device
                    ),
                )
            )

        fused = torch.cat(
            (
                target_embedding,
                neighbor_embedding,
                map_embedding,
            ),
            dim=-1,
        )

        return self.fusion(
            fused
        )

    def forward(
        self,
        target_history,
        neighbor_histories,
        neighbor_mask,
        map_context,
    ) -> GaussianTrajectoryOutput:

        embedding = self._encode(
            target_history,
            neighbor_histories,
            neighbor_mask,
            map_context,
        )

        batch = int(
            embedding.shape[0]
        )

        mean = (
            self.mean_head(
                embedding
            )
            .reshape(
                batch,
                PREDICTION_HORIZONS,
                POSITION_DIM,
            )
        )

        raw_cholesky = (
            self.cholesky_head(
                embedding
            )
            .reshape(
                batch,
                PREDICTION_HORIZONS,
                CHOLESKY_PARAMETERS_PER_HORIZON,
            )
        )

        scale_tril = (
            scale_tril_from_raw(
                raw_cholesky
            )
        )

        return GaussianTrajectoryOutput(
            mean=mean,
            scale_tril=scale_tril,
        )


def _inverse_softplus(
    value: float,
) -> float:
    if value <= 0.0:
        raise ValueError(
            "inverse-softplus input "
            "must be positive."
        )

    return math.log(
        math.expm1(
            value
        )
    )


def initialize_from_deterministic(
    gaussian_model:
        GaussianTrajectoryGRU,
    deterministic_state_dict,
    *,
    initial_std_normalized:
        float
        =
        DEFAULT_INITIAL_STD_NORMALIZED,
) -> None:
    """
    Exact mean/backbone initialization from
    frozen deterministic Block4.3 model.

    Covariance head starts input-independent:
      off-diagonals = 0
      diagonal std = requested value.
    """

    if initial_std_normalized <= (
        MIN_PREDICTIVE_STD_NORMALIZED
    ):
        raise ValueError(
            "initial predictive std "
            "must exceed covariance floor."
        )

    required_prefixes = (
        "target_gru.",
        "neighbor_gru.",
        "map_encoder.",
    )

    own = (
        gaussian_model
        .state_dict()
    )

    copied = {}

    for key in own:
        if key.startswith(
            required_prefixes
        ):
            if key not in (
                deterministic_state_dict
            ):
                raise KeyError(
                    "Frozen deterministic "
                    f"checkpoint lacks {key}."
                )

            copied[key] = (
                deterministic_state_dict[
                    key
                ]
            )

    deterministic_to_gaussian = {
        "head.0.weight":
            "fusion.0.weight",

        "head.0.bias":
            "fusion.0.bias",

        "head.2.weight":
            "fusion.2.weight",

        "head.2.bias":
            "fusion.2.bias",

        "head.4.weight":
            "mean_head.weight",

        "head.4.bias":
            "mean_head.bias",
    }

    for old_key, new_key in (
        deterministic_to_gaussian.items()
    ):
        if old_key not in (
            deterministic_state_dict
        ):
            raise KeyError(
                "Frozen deterministic "
                f"checkpoint lacks {old_key}."
            )

        copied[new_key] = (
            deterministic_state_dict[
                old_key
            ]
        )

    missing_copy = [
        key
        for key in (
            own.keys()
        )
        if (
            not key.startswith(
                "cholesky_head."
            )
            and
            key not in copied
        )
    ]

    if missing_copy:
        raise RuntimeError(
            "Incomplete deterministic "
            "initialization mapping: "
            f"{missing_copy}"
        )

    with torch.no_grad():
        for key, value in (
            copied.items()
        ):
            own[
                key
            ].copy_(
                value
            )

        gaussian_model.cholesky_head.weight.zero_()

        gaussian_model.cholesky_head.bias.zero_()

        target_softplus = (
            initial_std_normalized
            -
            MIN_PREDICTIVE_STD_NORMALIZED
        )

        raw_diagonal = (
            _inverse_softplus(
                target_softplus
            )
        )

        bias = (
            gaussian_model
            .cholesky_head
            .bias
            .view(
                PREDICTION_HORIZONS,
                CHOLESKY_PARAMETERS_PER_HORIZON,
            )
        )

        bias[:, 0] = (
            raw_diagonal
        )

        bias[:, 2] = (
            raw_diagonal
        )

        bias[:, 5] = (
            raw_diagonal
        )
