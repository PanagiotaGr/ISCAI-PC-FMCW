from __future__ import annotations

from dataclasses import dataclass

from iscai_stage2.observations.womd_ideal_adapter import (
    ActorIdealObservationSeries,
)


@dataclass(frozen=True)
class Stage3ActorBeamInput:
    scenario_id: str
    track_index: int
    track_id: str
    object_class: str
    is_sdc: bool

    observations: tuple


def convert_stage2_actor_series(
    series: ActorIdealObservationSeries,
) -> Stage3ActorBeamInput:
    """
    Frozen Stage2 -> Stage3 bridge.

    No geometry modification.
    No prediction.
    No future access.
    """

    return Stage3ActorBeamInput(
        scenario_id=series.scenario_id,
        track_index=series.track_index,
        track_id=series.track_id,
        object_class=series.object_class,
        is_sdc=series.is_sdc,
        observations=series.observations,
    )
