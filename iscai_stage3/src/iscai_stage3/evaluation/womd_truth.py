from __future__ import annotations

from collections import Counter
from typing import Iterable

from waymo_open_dataset.protos import scenario_pb2

from iscai_stage3.evaluation.contracts import (
    EvaluationTruth,
    EvaluationTruthTrack,
    TruthTrajectoryPoint,
)


def _object_type_name(
    object_type: int,
) -> str:
    try:
        return (
            scenario_pb2
            .Track
            .ObjectType
            .Name(
                int(object_type)
            )
        )
    except Exception:
        return (
            f"TYPE_UNKNOWN_{int(object_type)}"
        )


def _resolve_eligible_indices(
    scenario,
    *,
    eligible_track_indices:
        Iterable[int] | None,
    eligible_truth_ids:
        Iterable[str] | None,
) -> tuple[int, ...]:
    if (
        eligible_track_indices is None
    ) == (
        eligible_truth_ids is None
    ):
        raise ValueError(
            "Provide exactly one of "
            "eligible_track_indices or "
            "eligible_truth_ids."
        )

    if eligible_track_indices is not None:
        indices = tuple(
            sorted(
                set(
                    int(value)
                    for value in (
                        eligible_track_indices
                    )
                )
            )
        )

    else:
        requested = set(
            str(value)
            for value in (
                eligible_truth_ids
                or ()
            )
        )

        by_id = {
            str(track.id): index
            for index, track in enumerate(
                scenario.tracks
            )
        }

        missing = sorted(
            requested
            -
            set(by_id)
        )

        if missing:
            raise ValueError(
                "Eligible truth IDs absent "
                f"from Scenario: {missing}"
            )

        indices = tuple(
            sorted(
                by_id[value]
                for value in requested
            )
        )

    for index in indices:
        if not (
            0
            <=
            index
            <
            len(scenario.tracks)
        ):
            raise IndexError(
                "Eligible WOMD track index "
                f"out of bounds: {index}"
            )

    return indices


def _exact_future_index(
    timestamps,
    *,
    current_index: int,
    horizon_s: float,
    tolerance_s: float,
) -> int | None:
    anchor_time = float(
        timestamps[current_index]
    )

    target_time = (
        anchor_time
        +
        float(horizon_s)
    )

    matches = []

    for index in range(
        current_index + 1,
        len(timestamps),
    ):
        timestamp = float(
            timestamps[index]
        )

        if abs(
            timestamp
            -
            target_time
        ) <= tolerance_s:
            matches.append(index)

    if len(matches) > 1:
        raise ValueError(
            "Multiple WOMD timestamps match "
            "one evaluation horizon."
        )

    if not matches:
        return None

    return matches[0]


def _position_H0(
    state,
    *,
    T_H0_from_W,
):
    point_W = (
        float(state.center_x),
        float(state.center_y),
        float(state.center_z),
    )

    point_H0 = (
        T_H0_from_W
        .apply_point(
            point_W
        )
    )

    return tuple(
        float(value)
        for value in point_H0
    )


def build_womd_truth_sidecar(
    scenario,
    *,
    T_H0_from_W,
    horizons_s: tuple[
        float,
        ...
    ] = (
        0.1,
        0.3,
        0.5,
        1.0,
    ),
    eligible_track_indices:
        Iterable[int] | None = None,
    eligible_truth_ids:
        Iterable[str] | None = None,
    timestamp_tolerance_s: float = 2e-3,
) -> EvaluationTruth:
    """
    Build evaluator-only WOMD future truth.

    Eligibility is fixed using the caller's
    current/anchor-side eligible set and the
    CURRENT WOMD validity flag only.

    Future states are then used solely to
    create scoring points.

    No temporal interpolation is performed.
    """

    current_index = int(
        scenario.current_time_index
    )

    timestamps = tuple(
        float(value)
        for value in (
            scenario.timestamps_seconds
        )
    )

    if not (
        0
        <=
        current_index
        <
        len(timestamps)
    ):
        raise ValueError(
            "Invalid WOMD current_time_index."
        )

    indices = _resolve_eligible_indices(
        scenario,
        eligible_track_indices=(
            eligible_track_indices
        ),
        eligible_truth_ids=(
            eligible_truth_ids
        ),
    )

    tracks = []

    for index in indices:
        if (
            index
            ==
            int(
                scenario.sdc_track_index
            )
        ):
            continue

        track = scenario.tracks[index]

        if (
            current_index
            >=
            len(track.states)
        ):
            raise ValueError(
                "WOMD track/state alignment "
                "is invalid."
            )

        current_state = (
            track.states[
                current_index
            ]
        )

        if not bool(
            current_state.valid
        ):
            continue

        truth_id = str(
            track.id
        )

        if not truth_id:
            raise ValueError(
                "Eligible WOMD track has "
                "empty truth ID."
            )

        anchor_position = (
            _position_H0(
                current_state,
                T_H0_from_W=(
                    T_H0_from_W
                ),
            )
        )

        points = []

        for horizon in horizons_s:
            future_index = (
                _exact_future_index(
                    timestamps,
                    current_index=(
                        current_index
                    ),
                    horizon_s=float(
                        horizon
                    ),
                    tolerance_s=(
                        timestamp_tolerance_s
                    ),
                )
            )

            if future_index is None:
                continue

            if (
                future_index
                >=
                len(track.states)
            ):
                raise ValueError(
                    "WOMD future state "
                    "alignment is invalid."
                )

            state = (
                track.states[
                    future_index
                ]
            )

            # Future validity controls only
            # availability of this scoring
            # point. It must never determine
            # anchor eligibility.
            if not bool(
                state.valid
            ):
                continue

            points.append(
                TruthTrajectoryPoint(
                    horizon_s=float(
                        horizon
                    ),
                    timestamp_s=float(
                        timestamps[
                            future_index
                        ]
                    ),
                    position_H0_m=(
                        _position_H0(
                            state,
                            T_H0_from_W=(
                                T_H0_from_W
                            ),
                        )
                    ),
                )
            )

        tracks.append(
            EvaluationTruthTrack(
                truth_id=truth_id,
                actor_class=(
                    _object_type_name(
                        track.object_type
                    )
                ),
                anchor_position_H0_m=(
                    anchor_position
                ),
                points=tuple(points),
            )
        )

    tracks.sort(
        key=lambda item:
        item.truth_id
    )

    return EvaluationTruth(
        scenario_id=str(
            scenario.scenario_id
        ),
        anchor_timestamp_s=float(
            timestamps[
                current_index
            ]
        ),
        tracks=tuple(tracks),
    )


def anchor_class_inventory(
    scenario,
    *,
    eligible_track_indices:
        Iterable[int] | None = None,
    eligible_truth_ids:
        Iterable[str] | None = None,
) -> dict[str, int]:
    """
    Current/anchor-only class inventory.

    Future states are never inspected.
    """

    indices = _resolve_eligible_indices(
        scenario,
        eligible_track_indices=(
            eligible_track_indices
        ),
        eligible_truth_ids=(
            eligible_truth_ids
        ),
    )

    current_index = int(
        scenario.current_time_index
    )

    counter = Counter()

    for index in indices:
        if (
            index
            ==
            int(
                scenario.sdc_track_index
            )
        ):
            continue

        track = scenario.tracks[index]

        if (
            current_index
            >=
            len(track.states)
        ):
            raise ValueError(
                "WOMD current-state alignment "
                "is invalid."
            )

        if not bool(
            track.states[
                current_index
            ].valid
        ):
            continue

        counter[
            _object_type_name(
                track.object_type
            )
        ] += 1

    return dict(
        sorted(
            counter.items()
        )
    )
