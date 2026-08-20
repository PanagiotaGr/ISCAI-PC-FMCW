from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
    CleanObservationRecord,
    clean_crlb_conditioned_observation,
)
from iscai_stage2.observations.womd_ideal_adapter import (
    IdealObservationScene,
)


@dataclass(frozen=True)
class ActorCleanObservationSeries:
    scenario_id: str

    track_index: int
    track_id: str
    object_class: str

    records: tuple[
        CleanObservationRecord, ...
    ]


@dataclass(frozen=True)
class CleanObservationScene:
    scenario_id: str
    anchor_index: int

    actors: tuple[
        ActorCleanObservationSeries, ...
    ]

    mode: str = "clean_crlb_conditioned"

    association_mode: str = (
        "oracle_womd_track_id"
    )

    measured_fmcw: bool = False

    sensor_noise_applied: bool = False
    missed_detections_applied: bool = False
    false_alarms_applied: bool = False


def build_clean_observation_scene(
    *,
    ideal_scene: IdealObservationScene,
    config: CleanObservationConfig,
) -> CleanObservationScene:

    actors = []

    for actor in ideal_scene.actors:
        records = tuple(
            clean_crlb_conditioned_observation(
                scenario_id=(
                    ideal_scene.scenario_id
                ),
                track_id=actor.track_id,
                object_class=(
                    actor.object_class
                ),
                ideal=ideal,
                config=config,
            )
            for ideal in actor.observations
        )

        actors.append(
            ActorCleanObservationSeries(
                scenario_id=(
                    ideal_scene.scenario_id
                ),
                track_index=(
                    actor.track_index
                ),
                track_id=actor.track_id,
                object_class=(
                    actor.object_class
                ),
                records=records,
            )
        )

    return CleanObservationScene(
        scenario_id=ideal_scene.scenario_id,
        anchor_index=ideal_scene.anchor_index,
        actors=tuple(actors),
    )


def clean_observation_scene_sha256(
    scene: CleanObservationScene,
) -> str:
    encoded = json.dumps(
        asdict(scene),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()
