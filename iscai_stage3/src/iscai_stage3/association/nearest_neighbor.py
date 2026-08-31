from __future__ import annotations

from dataclasses import dataclass

from iscai_stage3.association.contracts import (
    AssociatedDetection,
    AssociatedFrame,
    AssociatedTrack,
    AssociationConfig,
    EstimatedAssociationResult,
)

from iscai_stage3.association.gating import (
    association_cost,
    predict_measurement,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.observations import (
    frame_detections,
    frame_timestamp_s,
    snapshot_from_detection,
)


@dataclass
class _WorkingTrack:
    track_id: str

    history: list[
        AssociatedDetection
    ]

    missed_frames: int = 0
    terminated: bool = False


def _history_for_prediction(
    track: _WorkingTrack,
):
    return tuple(
        (
            item.timestamp_s,
            item.detection,
        )
        for item in track.history
    )


def associate_estimated_gnn(
    sequence: AlgorithmObservationSequence,
    *,
    config: AssociationConfig | None = None,
) -> EstimatedAssociationResult:
    """
    Deterministic gated greedy nearest-neighbor
    association.

    IMPORTANT:
    - uses Stage2 algorithm-facing frames only;
    - does not read evaluator truth;
    - detection_key is carried through but is
      never used in association cost;
    - Stage3 track IDs are newly generated.
    """

    if config is None:
        config = AssociationConfig()

    working_tracks: list[
        _WorkingTrack
    ] = []

    frame_outputs = []

    next_track_number = 0

    previous_timestamp = None

    for frame_index, frame in enumerate(
        sequence.frames
    ):
        timestamp = frame_timestamp_s(
            frame
        )

        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            raise ValueError(
                "Observation frame timestamps "
                "must be strictly increasing."
            )

        previous_timestamp = timestamp

        detections = tuple(
            snapshot_from_detection(d)
            for d in frame_detections(
                frame
            )
        )

        active_tracks = [
            track
            for track in working_tracks
            if not track.terminated
        ]

        candidates = []

        for track_index, track in enumerate(
            active_tracks
        ):
            if not track.history:
                continue

            last = track.history[-1]

            dt_s = (
                timestamp
                -
                last.timestamp_s
            )

            if dt_s <= 0.0:
                raise ValueError(
                    "Association timestamps "
                    "must increase."
                )

            predicted = predict_measurement(
                history=(
                    _history_for_prediction(
                        track
                    )
                ),
                target_timestamp_s=timestamp,
            )

            for detection_index, detection in enumerate(
                detections
            ):
                score = association_cost(
                    predicted=predicted,
                    previous=last.detection,
                    current=detection,
                    dt_s=dt_s,
                    config=config,
                )

                if not score.feasible:
                    continue

                candidates.append(
                    (
                        score.cost,
                        track.track_id,
                        detection_index,
                        track_index,
                    )
                )

        candidates.sort(
            key=lambda x: (
                x[0],
                x[1],
                x[2],
            )
        )

        matched_track_indices = set()
        matched_detection_indices = set()

        assignments = []

        for (
            _cost,
            _track_id,
            detection_index,
            track_index,
        ) in candidates:
            if (
                track_index
                in matched_track_indices
            ):
                continue

            if (
                detection_index
                in matched_detection_indices
            ):
                continue

            track = active_tracks[
                track_index
            ]

            associated = AssociatedDetection(
                track_id=track.track_id,
                frame_index=frame_index,
                timestamp_s=timestamp,
                detection=(
                    detections[
                        detection_index
                    ]
                ),
            )

            track.history.append(
                associated
            )

            track.missed_frames = 0

            matched_track_indices.add(
                track_index
            )

            matched_detection_indices.add(
                detection_index
            )

            assignments.append(
                associated
            )

        for track_index, track in enumerate(
            active_tracks
        ):
            if (
                track_index
                in matched_track_indices
            ):
                continue

            track.missed_frames += 1

            if (
                track.missed_frames
                >
                config.max_missed_frames
            ):
                track.terminated = True

        for detection_index, detection in enumerate(
            detections
        ):
            if (
                detection_index
                in matched_detection_indices
            ):
                continue

            track_id = (
                f"s3trk-"
                f"{next_track_number:06d}"
            )

            next_track_number += 1

            associated = AssociatedDetection(
                track_id=track_id,
                frame_index=frame_index,
                timestamp_s=timestamp,
                detection=detection,
            )

            new_track = _WorkingTrack(
                track_id=track_id,
                history=[associated],
            )

            working_tracks.append(
                new_track
            )

            assignments.append(
                associated
            )

        assignments.sort(
            key=lambda x: x.track_id
        )

        frame_outputs.append(
            AssociatedFrame(
                frame_index=frame_index,
                timestamp_s=timestamp,
                detections=tuple(
                    assignments
                ),
            )
        )

    track_outputs = []

    for track in working_tracks:
        track_outputs.append(
            AssociatedTrack(
                track_id=track.track_id,
                detections=tuple(
                    track.history
                ),
                terminated=(
                    track.terminated
                ),
            )
        )

    track_outputs.sort(
        key=lambda x: x.track_id
    )

    return EstimatedAssociationResult(
        scenario_id=sequence.scenario_id,
        frames=tuple(frame_outputs),
        tracks=tuple(track_outputs),
        truth_used=False,
    )
