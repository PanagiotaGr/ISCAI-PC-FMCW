from __future__ import annotations

from dataclasses import dataclass


Vec3 = tuple[
    float,
    float,
    float,
]


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True)
class GaussianTrajectoryPrediction:
    """
    Stage4 probabilistic trajectory contract.

    Each future step contains:
        - predicted mean position
        - 3D covariance matrix
    """

    timestamps_s: tuple[float, ...]

    mean_positions_m: tuple[
        Vec3,
        ...
    ]

    covariance_m2: tuple[
        Matrix3,
        ...
    ]

    model_name: str


    def __post_init__(self):

        if len(self.timestamps_s) == 0:
            raise ValueError(
                "Empty prediction horizon."
            )

        if len(self.mean_positions_m) != len(
            self.timestamps_s
        ):
            raise ValueError(
                "Mean trajectory length mismatch."
            )

        if len(self.covariance_m2) != len(
            self.timestamps_s
        ):
            raise ValueError(
                "Covariance length mismatch."
            )
