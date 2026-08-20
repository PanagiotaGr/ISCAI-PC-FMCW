from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math


MEASUREMENT_COMPONENTS = (
    "range_m",
    "radial_velocity_mps",
    "azimuth_rad",
    "elevation_rad",
)

OBSERVATION_FRAME = "Ht"

OBSERVATION_SEMANTICS = (
    "pc_fmcw_like_simulated_measurement"
)

MEASURED_FMCW = False

RADIAL_VELOCITY_SOURCE = (
    "geometry_derived_from_causal_womd_trajectory"
)


Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def _determinant(
    matrix: tuple[tuple[float, ...], ...],
) -> float:
    n = len(matrix)

    if n == 0:
        return 1.0

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        )

    total = 0.0

    for col in range(n):
        minor = tuple(
            tuple(
                matrix[row][j]
                for j in range(n)
                if j != col
            )
            for row in range(1, n)
        )

        total += (
            (-1.0 if col % 2 else 1.0)
            * matrix[0][col]
            * _determinant(minor)
        )

    return total


def _is_symmetric_psd(
    matrix: Matrix4,
    *,
    tolerance: float = 1e-12,
) -> bool:
    for i in range(4):
        for j in range(4):
            if abs(
                matrix[i][j] - matrix[j][i]
            ) > tolerance:
                return False

    # For a real symmetric matrix, PSD iff every
    # principal minor is non-negative.
    indices = range(4)

    for size in range(1, 5):
        for subset in combinations(
            indices,
            size,
        ):
            principal = tuple(
                tuple(
                    matrix[i][j]
                    for j in subset
                )
                for i in subset
            )

            if _determinant(
                principal
            ) < -tolerance:
                return False

    return True


@dataclass(frozen=True)
class MeasurementCovariance:
    """
    Covariance order is fixed by MEASUREMENT_COMPONENTS:

      [range, radial velocity, azimuth, elevation]
    """

    matrix: Matrix4

    def __post_init__(self) -> None:
        if len(self.matrix) != 4:
            raise ValueError(
                "Measurement covariance must be 4x4."
            )

        for row in self.matrix:
            if len(row) != 4:
                raise ValueError(
                    "Measurement covariance must be 4x4."
                )

            if not all(
                math.isfinite(value)
                for value in row
            ):
                raise ValueError(
                    "Measurement covariance must be finite."
                )

        if not _is_symmetric_psd(
            self.matrix
        ):
            raise ValueError(
                "Measurement covariance must be "
                "symmetric positive-semidefinite."
            )


@dataclass(frozen=True)
class PcfmcwLikeObservation:
    """
    Stage-2 simulated PC-FMCW-like observation.

    This is NOT a measured WOMD-LiDAR Doppler observation
    and NOT a real-FMCW hardware measurement.

    track_id/scenario_id are metadata only.
    """

    scenario_id: str
    track_id: str
    object_class: str

    time_index: int
    timestamp_s: float

    range_m: float
    radial_velocity_mps: float
    azimuth_rad: float
    elevation_rad: float

    covariance: MeasurementCovariance

    measurement_valid: bool

    frame_name: str = OBSERVATION_FRAME
    observation_semantics: str = (
        OBSERVATION_SEMANTICS
    )
    measured_fmcw: bool = MEASURED_FMCW
    radial_velocity_source: str = (
        RADIAL_VELOCITY_SOURCE
    )

    def __post_init__(self) -> None:
        if self.time_index < 0:
            raise ValueError(
                "time_index must be non-negative."
            )

        if not math.isfinite(
            self.timestamp_s
        ):
            raise ValueError(
                "timestamp_s must be finite."
            )

        if self.frame_name != "Ht":
            raise ValueError(
                "Stage-2 observation frame must be Ht."
            )

        if self.measured_fmcw:
            raise ValueError(
                "Stage-2 WOMD evaluation must not claim "
                "measured FMCW data."
            )

        if self.measurement_valid:
            values = (
                self.range_m,
                self.radial_velocity_mps,
                self.azimuth_rad,
                self.elevation_rad,
            )

            if not all(
                math.isfinite(value)
                for value in values
            ):
                raise ValueError(
                    "Valid measurements must be finite."
                )

            if self.range_m < 0.0:
                raise ValueError(
                    "Range cannot be negative."
                )

            if not (
                -math.pi
                <= self.azimuth_rad
                <= math.pi
            ):
                raise ValueError(
                    "Azimuth must lie in [-pi, pi]."
                )

            if not (
                -math.pi / 2.0
                <= self.elevation_rad
                <= math.pi / 2.0
            ):
                raise ValueError(
                    "Elevation must lie in "
                    "[-pi/2, pi/2]."
                )

    def measurement_vector(
        self,
    ) -> tuple[float, float, float, float]:
        return (
            self.range_m,
            self.radial_velocity_mps,
            self.azimuth_rad,
            self.elevation_rad,
        )
