from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from iscai_stage2.observations.clean_scene import (
    CleanObservationScene,
)
from iscai_stage2.observations.detection import (
    DetectionProbabilityConfig,
    apply_missed_detection,
)
from iscai_stage2.observations.detection_set import (
    AssociatedFilteredRecord,
    DetectionTruthSidecar,
    FalseAlarmConfig,
    UnlabeledDetectionFrame,
    build_unlabeled_detection_frame,
)
from iscai_stage2.observations.gaussian_corruption import (
    GaussianCorruptionConfig,
    gaussian_corrupt_clean_record,
)


DEGRADED_MODE = (
    "gaussian_noise_misses_false_alarms"
)


@dataclass(frozen=True)
class DegradedObservationConfig:
    gaussian_seed: int
    detection_seed: int

    detection_probability: (
        DetectionProbabilityConfig
    )

    false_alarms: (
        FalseAlarmConfig | None
    )


@dataclass(frozen=True)
class DegradedObservationScene:
    """
    Stage-2 degraded observation scene.

    `frames` are algorithm-facing.

    `truth_sidecars` are evaluator-only and MUST NOT be
    passed to MHT/Hough, Kalman, IMM or neural predictors.
    """

    scenario_id: str
    anchor_index: int

    frames: tuple[
        UnlabeledDetectionFrame, ...
    ]

    truth_sidecars: tuple[
        DetectionTruthSidecar, ...
    ]

    mode: str = DEGRADED_MODE

    measured_fmcw: bool = False

    algorithm_input_truth_free: bool = True


def _json_sha256(
    value: object,
) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def degraded_algorithm_sha256(
    scene: DegradedObservationScene,
) -> str:
    """
    Hash only algorithm-visible frames.
    """
    return _json_sha256(
        [
            asdict(frame)
            for frame in scene.frames
        ]
    )


def degraded_truth_sha256(
    scene: DegradedObservationScene,
) -> str:
    """
    Evaluator-only reproducibility hash.
    """
    return _json_sha256(
        [
            asdict(sidecar)
            for sidecar
            in scene.truth_sidecars
        ]
    )


def build_degraded_observation_scene(
    *,
    clean_scene: CleanObservationScene,
    config: DegradedObservationConfig,
) -> DegradedObservationScene:

    expected_frames = (
        clean_scene.anchor_index + 1
    )

    if expected_frames <= 0:
        raise ValueError(
            "Scene must contain at least one "
            "causal timestep."
        )

    for actor in clean_scene.actors:
        if len(actor.records) != expected_frames:
            raise RuntimeError(
                "Clean actor record count does not "
                "match scene causal length."
            )

    frames = []
    truth_sidecars = []

    for time_index in range(
        expected_frames
    ):
        timestamps = {
            actor.records[
                time_index
            ].timestamp_s
            for actor in clean_scene.actors
        }

        if len(timestamps) != 1:
            raise RuntimeError(
                "Actor timestamps disagree within frame."
            )

        timestamp_s = next(
            iter(timestamps)
        )

        associated_records = []

        for actor in clean_scene.actors:
            clean = actor.records[
                time_index
            ]

            gaussian = (
                gaussian_corrupt_clean_record(
                    scenario_id=(
                        clean_scene.scenario_id
                    ),
                    track_id=actor.track_id,
                    object_class=(
                        actor.object_class
                    ),
                    clean=clean,
                    config=(
                        GaussianCorruptionConfig(
                            seed=(
                                config.gaussian_seed
                            )
                        )
                    ),
                )
            )

            # apply_missed_detection returns before using
            # the SNR when the upstream record is invalid.
            if clean.sensing_snr is None:
                sensing_snr_db = 0.0
            else:
                sensing_snr_db = (
                    clean.sensing_snr.snr_db
                )

            filtered = (
                apply_missed_detection(
                    scenario_id=(
                        clean_scene.scenario_id
                    ),
                    track_id=actor.track_id,

                    gaussian=gaussian,

                    sensing_snr_db=(
                        sensing_snr_db
                    ),

                    probability_config=(
                        config
                        .detection_probability
                    ),

                    seed=(
                        config.detection_seed
                    ),
                )
            )

            associated_records.append(
                AssociatedFilteredRecord(
                    track_id=(
                        actor.track_id
                    ),
                    object_class=(
                        actor.object_class
                    ),
                    record=filtered,
                )
            )

        bundle = (
            build_unlabeled_detection_frame(
                scenario_id=(
                    clean_scene.scenario_id
                ),
                time_index=time_index,
                timestamp_s=timestamp_s,

                associated_records=tuple(
                    associated_records
                ),

                false_alarm_config=(
                    config.false_alarms
                ),
            )
        )

        frames.append(
            bundle.frame
        )

        truth_sidecars.append(
            bundle.truth
        )

    return DegradedObservationScene(
        scenario_id=(
            clean_scene.scenario_id
        ),
        anchor_index=(
            clean_scene.anchor_index
        ),
        frames=tuple(frames),
        truth_sidecars=tuple(
            truth_sidecars
        ),
    )
