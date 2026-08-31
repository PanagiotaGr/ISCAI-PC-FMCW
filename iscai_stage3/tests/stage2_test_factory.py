from __future__ import annotations

from dataclasses import (
    MISSING,
    fields,
)

from iscai_stage2.observations.detection_set import (
    UnlabeledDetection,
    UnlabeledDetectionFrame,
)

from iscai_stage3.contracts import (
    resolve_stage2_algorithm_schema,
)


def _raw_instance(
    cls,
    values,
):
    obj = object.__new__(cls)

    for field in fields(cls):
        if field.name in values:
            value = values[field.name]

        elif field.default is not MISSING:
            value = field.default

        elif (
            field.default_factory
            is not MISSING
        ):
            value = (
                field.default_factory()
            )

        else:
            value = None

        object.__setattr__(
            obj,
            field.name,
            value,
        )

    return obj


def make_detection(
    *,
    key,
    range_m,
    vr_mps,
    az_rad,
    el_rad=0.0,
    variance_scale=1.0,
):
    schema = (
        resolve_stage2_algorithm_schema()
    )

    covariance = (
        (
            0.04 * variance_scale,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.0,
            0.25 * variance_scale,
            0.0,
            0.0,
        ),
        (
            0.0,
            0.0,
            0.0004 * variance_scale,
            0.0,
        ),
        (
            0.0,
            0.0,
            0.0,
            0.0004 * variance_scale,
        ),
    )

    return _raw_instance(
        UnlabeledDetection,
        {
            schema.detection_key_field: key,
            schema.range_field: range_m,
            schema.radial_velocity_field: (
                vr_mps
            ),
            schema.azimuth_field: az_rad,
            schema.elevation_field: el_rad,
            schema.covariance_field: (
                covariance
            ),
        },
    )


def make_frame(
    *,
    timestamp_s,
    detections,
):
    schema = (
        resolve_stage2_algorithm_schema()
    )

    return _raw_instance(
        UnlabeledDetectionFrame,
        {
            schema.frame_timestamp_field: (
                timestamp_s
            ),
            schema.frame_detections_field: (
                tuple(detections)
            ),
        },
    )
