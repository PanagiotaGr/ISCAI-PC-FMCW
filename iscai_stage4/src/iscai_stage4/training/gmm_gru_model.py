import torch
from torch import nn


class GMMGRUPredictor(nn.Module):
    """
    Learned multimodal GRU trajectory predictor.

    For each input history it predicts K trajectory modes.

    Each mode contains, for every future timestep:
        mux, muy
        sigma_x, sigma_y
        rho

    and one learned mixture probability pi_k.
    """

    def __init__(
        self,
        input_size=8,
        hidden=128,
        layers=1,
        dropout=0.0,
        future=10,
        modes=5,
    ):
        super().__init__()

        self.future = future
        self.modes = modes

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # K mixture logits
        self.pi_head = nn.Linear(
            hidden,
            modes,
        )

        # For each mode and future timestep:
        # mux, muy, log_sigma_x, log_sigma_y, rho_raw
        self.trajectory_head = nn.Linear(
            hidden,
            modes * future * 5,
        )

    def forward(self, x):

        _, h = self.gru(x)

        h = self.dropout(h[-1])

        logits = self.pi_head(h)

        pi = torch.softmax(
            logits,
            dim=-1,
        )

        out = self.trajectory_head(h)

        out = out.reshape(
            -1,
            self.modes,
            self.future,
            5,
        )

        mu = out[..., 0:2]

        log_std = out[..., 2:4]

        rho_raw = out[..., 4:5]

        std = torch.exp(
            log_std.clamp(
                min=-7.0,
                max=7.0,
            )
        )

        rho = torch.tanh(rho_raw)

        return {
            "pi": pi,
            "logits": logits,
            "mu": mu,
            "std": std,
            "rho": rho,
        }
