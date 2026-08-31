from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from iscai_stage4.data.neural_inputs import (
    FEATURE_DIM,
    MAP_CONTEXT_DIM,
    MAX_NEIGHBORS,
)

from .gaussian_gru import (
    CHOLESKY_PARAMETERS_PER_HORIZON,
    PREDICTION_HORIZONS,
    POSITION_DIM,
    scale_tril_from_raw,
)


GMM_COMPONENTS = 3

CENTRAL_COMPONENT_INDEX = 1

INITIAL_SECOND_COORDINATE_OFFSETS = (
    -0.05,
    0.0,
    +0.05,
)


@dataclass(frozen=True)
class GMMTrajectoryOutput:
    """
    One trajectory-level mixture weight per mode.

    means:
        [B,K,H,3]

    scale_tril:
        [B,K,H,3,3]

    mixture_logits:
        [B,K]

    A mode therefore remains coherent across
    all prediction horizons.
    """

    means: torch.Tensor
    scale_tril: torch.Tensor
    mixture_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.means.ndim != 4:
            raise ValueError(
                "GMM means must be [B,K,H,3]."
            )

        batch = int(
            self.means.shape[0]
        )

        if self.means.shape[1:] != (
            GMM_COMPONENTS,
            PREDICTION_HORIZONS,
            POSITION_DIM,
        ):
            raise ValueError(
                "GMM mean shape changed."
            )

        if self.scale_tril.shape != (
            batch,
            GMM_COMPONENTS,
            PREDICTION_HORIZONS,
            POSITION_DIM,
            POSITION_DIM,
        ):
            raise ValueError(
                "GMM scale_tril shape changed."
            )

        if self.mixture_logits.shape != (
            batch,
            GMM_COMPONENTS,
        ):
            raise ValueError(
                "GMM mixture logits must "
                "be trajectory-level [B,K]."
            )

    @property
    def mixture_probabilities(
        self,
    ) -> torch.Tensor:
        return torch.softmax(
            self.mixture_logits,
            dim=-1,
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


class GMMTrajectoryGRU(nn.Module):
    """
    Multi-modal extension of the frozen
    Gaussian GRU architecture.

    Inputs remain exactly the same causal
    Stage4 inputs.

    No maneuver label, actor ID, future state,
    tracks_to_predict or oracle mode label is
    supplied to the network.
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
                "Hidden dimensions must be positive."
            )

        self.use_neighbors = bool(
            use_neighbors
        )

        self.use_map = bool(
            use_map
        )

        self.target_gru = nn.GRU(
            input_size=FEATURE_DIM,
            hidden_size=target_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.neighbor_gru = nn.GRU(
            input_size=FEATURE_DIM,
            hidden_size=neighbor_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.map_encoder = nn.Sequential(
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

        fusion_input_dim = (
            target_hidden_dim
            +
            neighbor_hidden_dim
            +
            map_hidden_dim
        )

        self.fusion = nn.Sequential(
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

        self.mean_head = nn.Linear(
            64,
            GMM_COMPONENTS
            *
            PREDICTION_HORIZONS
            *
            POSITION_DIM,
        )

        self.cholesky_head = nn.Linear(
            64,
            GMM_COMPONENTS
            *
            PREDICTION_HORIZONS
            *
            CHOLESKY_PARAMETERS_PER_HORIZON,
        )

        self.mixture_logits_head = nn.Linear(
            64,
            GMM_COMPONENTS,
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
                "target_history must be [B,T,F]."
            )

        if neighbor_histories.ndim != 4:
            raise ValueError(
                "neighbor_histories must "
                "be [B,N,T,F]."
            )

        batch = int(
            target_history.shape[0]
        )

        if target_history.shape[-1] != FEATURE_DIM:
            raise ValueError(
                "Target feature dimension changed."
            )

        if neighbor_histories.shape[0] != batch:
            raise ValueError(
                "Batch-size mismatch."
            )

        if neighbor_histories.shape[1] != MAX_NEIGHBORS:
            raise ValueError(
                "Neighbour slot count changed."
            )

        if neighbor_histories.shape[-1] != FEATURE_DIM:
            raise ValueError(
                "Neighbour feature dimension changed."
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

        _, target_hidden = self.target_gru(
            target_history
        )

        target_embedding = target_hidden[-1]

        if self.use_neighbors:
            neighbor_flat = (
                neighbor_histories.reshape(
                    batch * MAX_NEIGHBORS,
                    neighbor_histories.shape[2],
                    neighbor_histories.shape[3],
                )
            )

            _, neighbor_hidden = self.neighbor_gru(
                neighbor_flat
            )

            neighbor_embedding = (
                neighbor_hidden[-1].reshape(
                    batch,
                    MAX_NEIGHBORS,
                    -1,
                )
            )

            mask = (
                neighbor_mask
                .to(
                    dtype=neighbor_embedding.dtype
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
                .clamp_min(1.0)
            )

            neighbor_embedding = (
                neighbor_embedding.sum(
                    dim=1
                )
                /
                denominator
            )

        else:
            neighbor_embedding = torch.zeros(
                (
                    batch,
                    self.neighbor_gru.hidden_size,
                ),
                dtype=target_history.dtype,
                device=target_history.device,
            )

        if self.use_map:
            map_embedding = self.map_encoder(
                map_context
            )

        else:
            map_embedding = torch.zeros(
                (
                    batch,
                    self.map_encoder[0].out_features,
                ),
                dtype=target_history.dtype,
                device=target_history.device,
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
    ) -> GMMTrajectoryOutput:

        embedding = self._encode(
            target_history,
            neighbor_histories,
            neighbor_mask,
            map_context,
        )

        batch = int(
            embedding.shape[0]
        )

        means = self.mean_head(
            embedding
        ).reshape(
            batch,
            GMM_COMPONENTS,
            PREDICTION_HORIZONS,
            POSITION_DIM,
        )

        raw_cholesky = self.cholesky_head(
            embedding
        ).reshape(
            batch,
            GMM_COMPONENTS,
            PREDICTION_HORIZONS,
            CHOLESKY_PARAMETERS_PER_HORIZON,
        )

        scale_tril = scale_tril_from_raw(
            raw_cholesky
        )

        mixture_logits = (
            self.mixture_logits_head(
                embedding
            )
        )

        return GMMTrajectoryOutput(
            means=means,
            scale_tril=scale_tril,
            mixture_logits=mixture_logits,
        )


def initialize_gmm_from_gaussian(
    gmm_model: GMMTrajectoryGRU,
    gaussian_state_dict,
    *,
    second_coordinate_offsets=(
        INITIAL_SECOND_COORDINATE_OFFSETS
    ),
) -> None:
    """
    Initialize the GMM from the frozen Gaussian.

    Shared encoder/fusion:
        exact Gaussian copy.

    Every component covariance:
        exact Gaussian covariance head copy.

    Every component mean:
        exact Gaussian mean head copy,
        plus a deterministic small offset in
        coordinate index 1.

    Central component index 1:
        exact Gaussian mean.

    Mixture logits:
        uniform initially.

    The offsets only break permutation symmetry.
    They are not maneuver labels.
    """

    offsets = tuple(
        float(value)
        for value in (
            second_coordinate_offsets
        )
    )

    if len(offsets) != GMM_COMPONENTS:
        raise ValueError(
            "Expected one initialization "
            "offset per GMM component."
        )

    if offsets[
        CENTRAL_COMPONENT_INDEX
    ] != 0.0:
        raise ValueError(
            "Central GMM component must "
            "retain exact Gaussian mean."
        )

    own = gmm_model.state_dict()

    shared_prefixes = (
        "target_gru.",
        "neighbor_gru.",
        "map_encoder.",
        "fusion.",
    )

    with torch.no_grad():
        for key in own:
            if key.startswith(
                shared_prefixes
            ):
                if key not in gaussian_state_dict:
                    raise KeyError(
                        f"Gaussian state lacks {key}."
                    )

                own[key].copy_(
                    gaussian_state_dict[key]
                )

        gaussian_mean_weight = (
            gaussian_state_dict[
                "mean_head.weight"
            ]
        )

        gaussian_mean_bias = (
            gaussian_state_dict[
                "mean_head.bias"
            ]
        )

        gaussian_cholesky_weight = (
            gaussian_state_dict[
                "cholesky_head.weight"
            ]
        )

        gaussian_cholesky_bias = (
            gaussian_state_dict[
                "cholesky_head.bias"
            ]
        )

        mean_weight = (
            gmm_model
            .mean_head
            .weight
            .view(
                GMM_COMPONENTS,
                PREDICTION_HORIZONS
                *
                POSITION_DIM,
                -1,
            )
        )

        mean_bias = (
            gmm_model
            .mean_head
            .bias
            .view(
                GMM_COMPONENTS,
                PREDICTION_HORIZONS,
                POSITION_DIM,
            )
        )

        chol_weight = (
            gmm_model
            .cholesky_head
            .weight
            .view(
                GMM_COMPONENTS,
                PREDICTION_HORIZONS
                *
                CHOLESKY_PARAMETERS_PER_HORIZON,
                -1,
            )
        )

        chol_bias = (
            gmm_model
            .cholesky_head
            .bias
            .view(
                GMM_COMPONENTS,
                PREDICTION_HORIZONS
                *
                CHOLESKY_PARAMETERS_PER_HORIZON,
            )
        )

        gaussian_mean_bias_reshaped = (
            gaussian_mean_bias.view(
                PREDICTION_HORIZONS,
                POSITION_DIM,
            )
        )

        for component in range(
            GMM_COMPONENTS
        ):
            mean_weight[
                component
            ].copy_(
                gaussian_mean_weight
            )

            mean_bias[
                component
            ].copy_(
                gaussian_mean_bias_reshaped
            )

            mean_bias[
                component,
                :,
                1,
            ].add_(
                offsets[
                    component
                ]
            )

            chol_weight[
                component
            ].copy_(
                gaussian_cholesky_weight
            )

            chol_bias[
                component
            ].copy_(
                gaussian_cholesky_bias
            )

        gmm_model.mixture_logits_head.weight.zero_()
        gmm_model.mixture_logits_head.bias.zero_()
