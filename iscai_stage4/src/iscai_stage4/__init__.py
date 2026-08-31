"""
Clean Stage4 package.

Stage4 scope:
deterministic/probabilistic trajectory forecasting
and trajectory-uncertainty calibration only.

Beam, receiver-angular posterior, optical link,
ADB and DeepSense are downstream stages.
"""

from .contracts import (
    GaussianHorizonPosterior,
    Stage4ScopeContract,
    TrajectoryGaussianPosterior,
)

__all__ = [
    "GaussianHorizonPosterior",
    "Stage4ScopeContract",
    "TrajectoryGaussianPosterior",
]
