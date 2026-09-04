from __future__ import annotations

ARCH_SPEC = {
    "name": "DualLiDARConv1D64",
    "raw_shape": [460, 2],
    "scr_shape": [216, 2],
    "conv1": {"in_channels": 2, "out_channels": 16, "kernel_size": 5, "padding": 2},
    "conv2": {"in_channels": 16, "out_channels": 32, "kernel_size": 5, "padding": 2},
    "pool": ["adaptive_avg_1", "adaptive_max_1"],
    "branch_output_features": 64,
    "fusion_features": 128,
    "layer_norm_features": 128,
    "hidden_features": 64,
    "dropout": 0.10,
    "output_logits": 64,
    "expected_parameter_count": 18208,
}

import torch
from torch import nn

class LiDARBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, padding=2)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.act2 = nn.GELU()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.mx = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return torch.cat([self.avg(x).squeeze(-1), self.mx(x).squeeze(-1)], dim=1)

class DualLiDARConv1D64(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_branch = LiDARBranch()
        self.scr_branch = LiDARBranch()
        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 64)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.10)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, raw_lidar, scr_lidar):
        if raw_lidar.ndim != 3 or tuple(raw_lidar.shape[1:]) != (2, 460):
            raise ValueError(f"raw_lidar must have shape [B,2,460], got {tuple(raw_lidar.shape)}")
        if scr_lidar.ndim != 3 or tuple(scr_lidar.shape[1:]) != (2, 216):
            raise ValueError(f"scr_lidar must have shape [B,2,216], got {tuple(scr_lidar.shape)}")
        a = self.raw_branch(raw_lidar)
        b = self.scr_branch(scr_lidar)
        x = torch.cat([a, b], dim=1)
        x = self.norm(x)
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)
