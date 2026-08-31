from __future__ import annotations

from functools import lru_cache
from math import isfinite

from iscai_stage2.observations.detection_set import (
    UnlabeledDetection,
    UnlabeledDetectionFrame,
)

from iscai_stage3.contracts import (
    resolve_stage2_algorithm_schema,
)


@lru_cache(maxsize=1)
def _schema():
    return resolve_stage2_algorithm_schema()


def _finite_float(
    value,
    *,
    name: str,
) -> float:
    result = float(value)

    if not isfinite(result):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def frame_timestamp_s(
    frame: UnlabeledDetectionFrame,
) -> float:
    if not isinstance(
        frame,
        UnlabeledDetectionFrame,
    ):
        raise TypeError(
            "Expected UnlabeledDetectionFrame."
        )

    value = getattr(
        frame,
        _schema().frame_timestamp_field,
    )

    return _finite_float(
        value,
        name="frame timestamp",
    )


def frame_detections(
    frame: UnlabeledDetectionFrame,
) -> tuple[UnlabeledDetection, ...]:
    if not isinstance(
        frame,
        UnlabeledDetectionFrame,
    ):
        raise TypeError(
            "Expected UnlabeledDetectionFrame."
        )

    values = getattr(
        frame,
        _schema().frame_detections_field,
    )

    result = tuple(values)

    for detection in result:
        if not isinstance(
            detection,
            UnlabeledDetection,
        ):
            raise TypeError(
                "Frame contains a non-Stage2 "
                "detection object."
            )

    return result


def detection_key(
    detection: UnlabeledDetection,
):
    return getattr(
        detection,
        _schema().detection_key_field,
    )


def detection_range_m(
    detection: UnlabeledDetection,
) -> float:
    return _finite_float(
        getattr(
            detection,
            _schema().range_field,
        ),
        name="range",
    )


def detection_radial_velocity_mps(
    detection: UnlabeledDetection,
) -> float:
    return _finite_float(
        getattr(
            detection,
            _schema().radial_velocity_field,
        ),
        name="radial velocity",
    )


def detection_azimuth_rad(
    detection: UnlabeledDetection,
) -> float:
    return _finite_float(
        getattr(
            detection,
            _schema().azimuth_field,
        ),
        name="azimuth",
    )


def detection_elevation_rad(
    detection: UnlabeledDetection,
) -> float:
    return _finite_float(
        getattr(
            detection,
            _schema().elevation_field,
        ),
        name="elevation",
    )


def detection_covariance(
    detection: UnlabeledDetection,
):
    """
    Return the complete frozen Stage2 R_t
    object without reducing or modifying it.
    """

    return getattr(
        detection,
        _schema().covariance_field,
    )
