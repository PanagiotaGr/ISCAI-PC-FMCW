from __future__ import annotations

import torch
from torch import nn

from iscai_stage4.data.neural_inputs import (
    FEATURE_DIM,
    MAP_CONTEXT_DIM,
    MAX_NEIGHBORS,
)


PREDICTION_HORIZONS = 4


class DeterministicTrajectoryGRU(
    nn.Module
):
    """
    Lightweight multi-agent deterministic
    Stage4 baseline.

    Numeric input only:
      target causal history,
      neighbour causal histories,
      neighbour mask,
      static map context.

    No actor IDs/classes/truth metadata.
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

        self.head = nn.Sequential(
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

            nn.Linear(
                64,
                PREDICTION_HORIZONS
                *
                3,
            ),
        )

    def forward(
        self,
        target_history,
        neighbor_histories,
        neighbor_mask,
        map_context,
    ):
        if (
            target_history.ndim
            !=
            3
        ):
            raise ValueError(
                "target_history must "
                "be [B,T,F]."
            )

        if (
            neighbor_histories.ndim
            !=
            4
        ):
            raise ValueError(
                "neighbor_histories must "
                "be [B,N,T,F]."
            )

        batch = (
            target_history.shape[0]
        )

        if (
            neighbor_histories
            .shape[0]
            !=
            batch
        ):
            raise ValueError(
                "Batch-size mismatch."
            )

        if (
            neighbor_histories
            .shape[1]
            !=
            MAX_NEIGHBORS
        ):
            raise ValueError(
                "Neighbour slot count "
                "changed."
            )

        if neighbor_mask.shape != (
            batch,
            MAX_NEIGHBORS,
        ):
            raise ValueError(
                "neighbor_mask shape "
                "changed."
            )

        if map_context.shape != (
            batch,
            MAP_CONTEXT_DIM,
        ):
            raise ValueError(
                "map_context shape "
                "changed."
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
                neighbor_embedding.sum(
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
                        self
                        .neighbor_gru
                        .hidden_size,
                    ),
                    dtype=(
                        target_history
                        .dtype
                    ),
                    device=(
                        target_history
                        .device
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
                        self
                        .map_encoder[0]
                        .out_features,
                    ),
                    dtype=(
                        target_history
                        .dtype
                    ),
                    device=(
                        target_history
                        .device
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

        output = self.head(
            fused
        )

        return output.reshape(
            batch,
            PREDICTION_HORIZONS,
            3,
        )
