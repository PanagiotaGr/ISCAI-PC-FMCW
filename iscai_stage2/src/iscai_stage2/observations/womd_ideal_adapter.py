from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from iscai_stage1.actors.womd_adapter import (
    AdaptedScenario,
    object_type_name,
)
from iscai_stage1.contracts.stage1a import (
    HeadlampSurrogateConfig,
    Vec3,
)
from iscai_stage1.geometry.frames import (
    DynamicHeadlampFrame,
    SdcStateW,
    build_dynamic_headlamp_frame,
)
from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
    adjacent_headlamp_velocity_W,
    headlamp_origin_W,
    ideal_causal_observable,
)


@dataclass(frozen=True)
class ActorIdealObservationSeries:
    scenario_id: str

    track_index: int

    # Metadata only; never numeric predictor features.
    track_id: str
    object_class: str

    is_sdc: bool

    observations: tuple[
        IdealCausalObservable, ...
    ]


@dataclass(frozen=True)
class IdealObservationScene:
    scenario_id: str
    anchor_index: int
    sdc_track_index: int

    actors: tuple[
        ActorIdealObservationSeries, ...
    ]

    frame_name: str = "Ht"
    sensor_realistic: bool = False
    measured_fmcw: bool = False

    radial_velocity_source: str = (
        "geometry_derived_from_causal_womd_trajectory"
    )

    angle_source: str = (
        "scene_perception_geometry"
    )


def ideal_observation_scene_sha256(
    scene: IdealObservationScene,
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


def build_causal_dynamic_headlamp_frames(
    adapted: AdaptedScenario,
    *,
    config: HeadlampSurrogateConfig
    | None = None,
) -> tuple[
    tuple[DynamicHeadlampFrame | None, ...],
    tuple[Vec3 | None, ...],
]:
    """
    Build Ht(t) and strict-adjacent headlamp velocity.

    Only causal Stage-1A history is consumed.
    """
    if config is None:
        config = HeadlampSurrogateConfig()

    sdc_actor = adapted.actors[
        adapted.sdc_track_index
    ]

    history = sdc_actor.artifact.history

    expected = adapted.anchor_index + 1

    if len(
        history.timestamps_s
    ) != expected:
        raise RuntimeError(
            "Unexpected SDC causal-history length."
        )

    frames: list[
        DynamicHeadlampFrame | None
    ] = []

    for time_index in range(expected):
        if not history.state_valid[
            time_index
        ]:
            frames.append(None)
            continue

        length_m = (
            history.dimensions_lwh_m[
                time_index
            ][0]
        )

        state = SdcStateW(
            center_w_m=history.position_W_m[
                time_index
            ],
            heading_rad=history.heading_rad[
                time_index
            ],
            length_m=length_m,
            valid=True,
        )

        frames.append(
            build_dynamic_headlamp_frame(
                state,
                config,
            )
        )

    velocities: list[
        Vec3 | None
    ] = [None]

    for time_index in range(
        1,
        expected,
    ):
        previous = frames[
            time_index - 1
        ]
        current = frames[
            time_index
        ]

        if (
            previous is None
            or current is None
        ):
            velocities.append(None)
            continue

        previous_origin = (
            headlamp_origin_W(
                previous.T_Ht_from_W
            )
        )

        current_origin = (
            headlamp_origin_W(
                current.T_Ht_from_W
            )
        )

        velocities.append(
            adjacent_headlamp_velocity_W(
                previous_origin_W_m=(
                    previous_origin
                ),
                current_origin_W_m=(
                    current_origin
                ),
                previous_timestamp_s=(
                    history.timestamps_s[
                        time_index - 1
                    ]
                ),
                current_timestamp_s=(
                    history.timestamps_s[
                        time_index
                    ]
                ),
            )
        )

    return (
        tuple(frames),
        tuple(velocities),
    )


def build_real_ideal_observation_scene(
    *,
    raw_scenario: Any,
    adapted: AdaptedScenario,
    include_sdc: bool = False,
) -> IdealObservationScene:
    """
    Convert frozen Stage-1A causal actor histories into
    noise-free Stage-2 ideal observables.

    Default target set:
      all non-SDC actors.

    No receiver-candidate or tracks_to_predict filtering is
    applied here, so pedestrians/cyclists remain available.
    """
    if (
        str(raw_scenario.scenario_id)
        != adapted.scenario_id
    ):
        raise RuntimeError(
            "Raw/adapted scenario-id mismatch."
        )

    if len(raw_scenario.tracks) != len(
        adapted.actors
    ):
        raise RuntimeError(
            "Raw/adapted track count mismatch."
        )

    (
        dynamic_frames,
        headlamp_velocities,
    ) = build_causal_dynamic_headlamp_frames(
        adapted
    )

    expected = adapted.anchor_index + 1

    actor_series = []

    for actor in adapted.actors:
        is_sdc = (
            actor.track_index
            == adapted.sdc_track_index
        )

        if is_sdc and not include_sdc:
            continue

        history = actor.artifact.history

        if len(
            history.timestamps_s
        ) != expected:
            raise RuntimeError(
                "Actor causal-history length mismatch: "
                f"track={actor.track_index}"
            )

        raw_track = raw_scenario.tracks[
            actor.track_index
        ]

        observations = []

        for time_index in range(
            expected
        ):
            frame = dynamic_frames[
                time_index
            ]

            timestamp_s = (
                history.timestamps_s[
                    time_index
                ]
            )

            if frame is None:
                observations.append(
                    IdealCausalObservable(
                        time_index=time_index,
                        timestamp_s=timestamp_s,
                        actor_position_Ht_m=None,
                        range_m=None,
                        azimuth_rad=None,
                        elevation_rad=None,
                        radial_velocity_mps=None,
                        geometry_valid=False,
                        radial_velocity_valid=False,
                    )
                )
                continue

            observations.append(
                ideal_causal_observable(
                    time_index=time_index,
                    timestamp_s=timestamp_s,

                    actor_position_W_m=(
                        history.position_W_m[
                            time_index
                        ]
                    ),
                    actor_position_valid=(
                        history.state_valid[
                            time_index
                        ]
                    ),

                    actor_velocity_W_mps=(
                        history.velocity_W_mps[
                            time_index
                        ]
                    ),
                    actor_velocity_valid=(
                        history.velocity_valid[
                            time_index
                        ]
                    ),

                    headlamp_velocity_W_mps=(
                        headlamp_velocities[
                            time_index
                        ]
                    ),

                    T_Ht_from_W=(
                        frame.T_Ht_from_W
                    ),
                )
            )

        actor_series.append(
            ActorIdealObservationSeries(
                scenario_id=(
                    adapted.scenario_id
                ),
                track_index=(
                    actor.track_index
                ),
                track_id=str(
                    raw_track.id
                ),
                object_class=(
                    object_type_name(
                        raw_track
                    )
                ),
                is_sdc=is_sdc,
                observations=tuple(
                    observations
                ),
            )
        )

    return IdealObservationScene(
        scenario_id=adapted.scenario_id,
        anchor_index=adapted.anchor_index,
        sdc_track_index=(
            adapted.sdc_track_index
        ),
        actors=tuple(actor_series),
    )
