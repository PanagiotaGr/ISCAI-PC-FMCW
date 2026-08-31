from __future__ import annotations

from dataclasses import dataclass, fields

from iscai_stage2.observations.detection_set import (
    UnlabeledDetection,
    UnlabeledDetectionFrame,
)


@dataclass(frozen=True)
class Stage2AlgorithmSchema:
    detection_key_field: str
    range_field: str
    radial_velocity_field: str
    azimuth_field: str
    elevation_field: str
    covariance_field: str

    frame_timestamp_field: str
    frame_detections_field: str


_DETECTION_ALIASES = {
    "detection_key": (
        "detection_key",
        "detection_id",
        "key",
    ),
    "range": (
        "range_m",
        "range",
    ),
    "radial_velocity": (
        "radial_velocity_mps",
        "vr_mps",
        "radial_velocity",
        "vr",
    ),
    "azimuth": (
        "azimuth_rad",
        "az_rad",
        "azimuth",
        "az",
    ),
    "elevation": (
        "elevation_rad",
        "el_rad",
        "elevation",
        "el",
    ),
    "covariance": (
        "covariance",
        "measurement_covariance",
        "covariance_matrix",
        "R_t",
    ),
}


_FRAME_ALIASES = {
    "timestamp": (
        "timestamp_s",
        "time_s",
        "timestamp",
    ),
    "detections": (
        "detections",
        "observations",
    ),
}


_FORBIDDEN_ALGORITHM_FIELDS = {
    "track_id",
    "actor_id",
    "track_index",
    "class_id",
    "object_class",
    "object_type",
    "truth",
    "is_true",
    "source_track_id",
}


def _field_names(cls) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(cls)
    )


def _resolve_unique(
    *,
    available: tuple[str, ...],
    aliases: tuple[str, ...],
    semantic_name: str,
) -> str:
    matches = tuple(
        name
        for name in aliases
        if name in available
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve "
            f"{semantic_name!r}. "
            f"available={available!r}, "
            f"matches={matches!r}"
        )

    return matches[0]


def forbidden_algorithm_fields() -> tuple[str, ...]:
    names = set(
        _field_names(UnlabeledDetection)
    )

    names.update(
        _field_names(
            UnlabeledDetectionFrame
        )
    )

    return tuple(
        sorted(
            names.intersection(
                _FORBIDDEN_ALGORITHM_FIELDS
            )
        )
    )


def resolve_stage2_algorithm_schema(
) -> Stage2AlgorithmSchema:
    detection_fields = _field_names(
        UnlabeledDetection
    )

    frame_fields = _field_names(
        UnlabeledDetectionFrame
    )

    forbidden = (
        forbidden_algorithm_fields()
    )

    if forbidden:
        raise RuntimeError(
            "Stage2 algorithm-facing contract "
            "contains forbidden evaluator/truth "
            f"fields: {forbidden!r}"
        )

    return Stage2AlgorithmSchema(
        detection_key_field=_resolve_unique(
            available=detection_fields,
            aliases=_DETECTION_ALIASES[
                "detection_key"
            ],
            semantic_name="detection_key",
        ),
        range_field=_resolve_unique(
            available=detection_fields,
            aliases=_DETECTION_ALIASES[
                "range"
            ],
            semantic_name="range",
        ),
        radial_velocity_field=(
            _resolve_unique(
                available=detection_fields,
                aliases=_DETECTION_ALIASES[
                    "radial_velocity"
                ],
                semantic_name=(
                    "radial_velocity"
                ),
            )
        ),
        azimuth_field=_resolve_unique(
            available=detection_fields,
            aliases=_DETECTION_ALIASES[
                "azimuth"
            ],
            semantic_name="azimuth",
        ),
        elevation_field=_resolve_unique(
            available=detection_fields,
            aliases=_DETECTION_ALIASES[
                "elevation"
            ],
            semantic_name="elevation",
        ),
        covariance_field=_resolve_unique(
            available=detection_fields,
            aliases=_DETECTION_ALIASES[
                "covariance"
            ],
            semantic_name="covariance",
        ),
        frame_timestamp_field=(
            _resolve_unique(
                available=frame_fields,
                aliases=_FRAME_ALIASES[
                    "timestamp"
                ],
                semantic_name=(
                    "frame_timestamp"
                ),
            )
        ),
        frame_detections_field=(
            _resolve_unique(
                available=frame_fields,
                aliases=_FRAME_ALIASES[
                    "detections"
                ],
                semantic_name=(
                    "frame_detections"
                ),
            )
        ),
    )
