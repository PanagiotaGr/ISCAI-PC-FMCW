from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BeamCandidate:
    """
    Candidate beam target generated from one actor.
    """

    scenario_id: str
    track_index: int
    object_class: str

    timestamp_s: float

    azimuth_rad: float
    elevation_rad: float

    azimuth_std_rad: float
    elevation_std_rad: float

    confidence: float



def validate_candidate(
    candidate: BeamCandidate,
) -> None:

    values = (
        candidate.azimuth_rad,
        candidate.elevation_rad,
        candidate.azimuth_std_rad,
        candidate.elevation_std_rad,
        candidate.confidence,
    )

    if not all(
        math.isfinite(v)
        for v in values
    ):
        raise ValueError(
            "Beam candidate contains non-finite values."
        )


    if (
        candidate.azimuth_std_rad < 0
        or candidate.elevation_std_rad < 0
    ):
        raise ValueError(
            "Negative uncertainty."
        )


    if not (
        0.0
        <= candidate.confidence
        <= 1.0
    ):
        raise ValueError(
            "Invalid confidence."
        )



def build_beam_candidate(
    *,
    scenario_id: str,
    track_index: int,
    object_class: str,
    timestamp_s: float,
    azimuth_rad: float,
    elevation_rad: float,
    azimuth_std_rad: float,
    elevation_std_rad: float,
    confidence: float,
) -> BeamCandidate:

    candidate = BeamCandidate(
        scenario_id=scenario_id,
        track_index=track_index,
        object_class=object_class,

        timestamp_s=timestamp_s,

        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,

        azimuth_std_rad=azimuth_std_rad,
        elevation_std_rad=elevation_std_rad,

        confidence=confidence,
    )

    validate_candidate(candidate)

    return candidate
