from __future__ import annotations

from math import isfinite

from iscai_stage3.beam.candidates import (
    BeamCandidate,
    build_beam_candidate,
)


def build_candidate_from_observation(
    *,
    scenario_id: str,
    track_index: int,
    object_class: str,
    observation,
    azimuth_std_rad: float,
    elevation_std_rad: float,
    confidence: float,
) -> BeamCandidate:
    """
    Stage2 causal observation -> Stage3 beam candidate.

    No prediction.
    No future access.
    No sensor noise.
    """

    if not observation.geometry_valid:
        raise ValueError(
            "Cannot create beam candidate from invalid geometry."
        )

    if observation.azimuth_rad is None:
        raise ValueError(
            "Missing azimuth."
        )

    if observation.elevation_rad is None:
        raise ValueError(
            "Missing elevation."
        )

    if not isfinite(
        observation.timestamp_s
    ):
        raise ValueError(
            "Invalid timestamp."
        )

    return build_beam_candidate(
        scenario_id=scenario_id,
        track_index=track_index,
        object_class=object_class,
        timestamp_s=observation.timestamp_s,
        azimuth_rad=observation.azimuth_rad,
        elevation_rad=observation.elevation_rad,
        azimuth_std_rad=azimuth_std_rad,
        elevation_std_rad=elevation_std_rad,
        confidence=confidence,
    )
