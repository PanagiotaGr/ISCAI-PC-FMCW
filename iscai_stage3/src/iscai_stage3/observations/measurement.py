from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi

from iscai_stage2.observations.detection_set import (
    UnlabeledDetection,
)

from iscai_stage3.observations.accessors import (
    detection_azimuth_rad,
    detection_covariance,
    detection_elevation_rad,
    detection_key,
    detection_radial_velocity_mps,
    detection_range_m,
)


Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def wrap_angle_rad(value: float) -> float:
    value = float(value)

    while value > pi:
        value -= 2.0 * pi

    while value <= -pi:
        value += 2.0 * pi

    return value


def _matrix_candidate(covariance):
    if hasattr(covariance, "matrix"):
        return covariance.matrix

    if hasattr(
        covariance,
        "covariance_matrix",
    ):
        return covariance.covariance_matrix

    if hasattr(covariance, "as_matrix"):
        return covariance.as_matrix()

    return covariance


def covariance_matrix_4x4(
    covariance,
) -> Matrix4:
    raw = _matrix_candidate(covariance)

    try:
        rows = tuple(
            tuple(float(x) for x in row)
            for row in raw
        )
    except Exception as exc:
        raise TypeError(
            "Measurement covariance cannot be "
            "converted to a numeric 4x4 matrix."
        ) from exc

    if len(rows) != 4:
        raise ValueError(
            "Measurement covariance must be 4x4."
        )

    if any(len(row) != 4 for row in rows):
        raise ValueError(
            "Measurement covariance must be 4x4."
        )

    for row in rows:
        for value in row:
            if not isfinite(value):
                raise ValueError(
                    "Measurement covariance must "
                    "contain only finite values."
                )

    for i in range(4):
        if rows[i][i] < 0.0:
            raise ValueError(
                "Measurement covariance diagonal "
                "cannot be negative."
            )

        for j in range(4):
            if abs(
                rows[i][j]
                -
                rows[j][i]
            ) > 1e-9:
                raise ValueError(
                    "Measurement covariance must "
                    "be symmetric."
                )

    return rows  # type: ignore[return-value]


@dataclass(frozen=True)
class MeasurementSnapshot:
    """
    One Stage2 algorithm-facing detection
    expressed in the original measurement space:

        [range, radial velocity,
         azimuth, elevation]

    No actor identity.
    No class.
    No evaluator truth.
    """

    detection_key: object

    range_m: float
    radial_velocity_mps: float

    azimuth_rad: float
    elevation_rad: float

    covariance_4x4: Matrix4

    source_detection: UnlabeledDetection

    def __post_init__(self) -> None:
        values = (
            self.range_m,
            self.radial_velocity_mps,
            self.azimuth_rad,
            self.elevation_rad,
        )

        if not all(
            isfinite(float(x))
            for x in values
        ):
            raise ValueError(
                "Measurement values must be finite."
            )

        if self.range_m < 0.0:
            raise ValueError(
                "Range cannot be negative."
            )

        covariance_matrix_4x4(
            self.covariance_4x4
        )


def snapshot_from_detection(
    detection: UnlabeledDetection,
) -> MeasurementSnapshot:
    if not isinstance(
        detection,
        UnlabeledDetection,
    ):
        raise TypeError(
            "Expected Stage2 "
            "UnlabeledDetection."
        )

    covariance = covariance_matrix_4x4(
        detection_covariance(detection)
    )

    return MeasurementSnapshot(
        detection_key=detection_key(
            detection
        ),
        range_m=detection_range_m(
            detection
        ),
        radial_velocity_mps=(
            detection_radial_velocity_mps(
                detection
            )
        ),
        azimuth_rad=wrap_angle_rad(
            detection_azimuth_rad(
                detection
            )
        ),
        elevation_rad=(
            detection_elevation_rad(
                detection
            )
        ),
        covariance_4x4=covariance,
        source_detection=detection,
    )
