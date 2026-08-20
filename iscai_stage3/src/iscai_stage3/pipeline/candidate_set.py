from __future__ import annotations

from dataclasses import dataclass

from iscai_stage3.beam.candidates import (
    BeamCandidate,
)

from iscai_stage3.pipeline.candidate_builder import (
    build_candidate_from_observation,
)


@dataclass(frozen=True)
class BeamCandidateSet:

    scenario_id: str

    candidates: tuple[
        BeamCandidate,
        ...

    ]


def build_candidate_set(
    *,
    scenario_id: str,
    actor_series,
    azimuth_std_rad: float,
    elevation_std_rad: float,
    confidence: float,
) -> BeamCandidateSet:
    """
    Build causal beam candidates from actors.

    Uses only the last causal observation.
    """

    candidates = []

    for actor in actor_series:

        observations = actor.observations

        if len(observations) == 0:
            continue


        anchor_observation = observations[-1]


        if not anchor_observation.geometry_valid:
            continue


        candidate = (
            build_candidate_from_observation(
                scenario_id=scenario_id,
                track_index=actor.track_index,
                object_class=actor.object_class,
                observation=anchor_observation,
                azimuth_std_rad=azimuth_std_rad,
                elevation_std_rad=elevation_std_rad,
                confidence=confidence,
            )
        )


        candidates.append(candidate)


    return BeamCandidateSet(
        scenario_id=scenario_id,
        candidates=tuple(candidates),
    )
