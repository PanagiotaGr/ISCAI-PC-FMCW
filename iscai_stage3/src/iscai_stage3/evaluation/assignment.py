from __future__ import annotations

from dataclasses import dataclass
from math import (
    isfinite,
    sqrt,
)

from iscai_stage3.evaluation.contracts import (
    EvaluationConfig,
    EvaluationPredictionSet,
    EvaluationTruth,
)


@dataclass(frozen=True)
class EvaluationMatch:
    prediction_id: str
    truth_id: str

    anchor_distance_m: float


@dataclass(frozen=True)
class AssignmentResult:
    matches: tuple[
        EvaluationMatch,
        ...
    ]

    unmatched_prediction_ids: tuple[
        str,
        ...
    ]

    unmatched_truth_ids: tuple[
        str,
        ...
    ]

    assignment_information: str = (
        "anchor_truth_only"
    )

    future_truth_used_for_assignment: bool = False


def _distance_3d(
    a,
    b,
) -> float:
    return sqrt(
        sum(
            (
                float(a[index])
                -
                float(b[index])
            )
            ** 2
            for index in range(3)
        )
    )


def _hungarian_square(
    cost: list[
        list[float]
    ],
) -> list[int]:
    """
    O(n^3) Hungarian algorithm.

    Returns:
        assignment[row] = column

    No SciPy dependency.
    """

    n = len(cost)

    if n == 0:
        return []

    if any(
        len(row) != n
        for row in cost
    ):
        raise ValueError(
            "Hungarian cost matrix must "
            "be square."
        )

    if any(
        not isfinite(value)
        for row in cost
        for value in row
    ):
        raise ValueError(
            "Hungarian costs must be finite."
        )

    u = [
        0.0
        for _ in range(n + 1)
    ]

    v = [
        0.0
        for _ in range(n + 1)
    ]

    p = [
        0
        for _ in range(n + 1)
    ]

    way = [
        0
        for _ in range(n + 1)
    ]

    for i in range(
        1,
        n + 1,
    ):
        p[0] = i

        j0 = 0

        minv = [
            float("inf")
            for _ in range(n + 1)
        ]

        used = [
            False
            for _ in range(n + 1)
        ]

        while True:
            used[j0] = True

            i0 = p[j0]

            delta = float("inf")
            j1 = 0

            for j in range(
                1,
                n + 1,
            ):
                if used[j]:
                    continue

                current = (
                    cost[
                        i0 - 1
                    ][
                        j - 1
                    ]
                    -
                    u[i0]
                    -
                    v[j]
                )

                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0

                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            if not isfinite(delta):
                raise ValueError(
                    "Hungarian assignment "
                    "failed."
                )

            for j in range(
                0,
                n + 1,
            ):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]

            p[j0] = p[j1]

            j0 = j1

            if j0 == 0:
                break

    assignment = [
        -1
        for _ in range(n)
    ]

    for j in range(
        1,
        n + 1,
    ):
        if p[j] != 0:
            assignment[
                p[j] - 1
            ] = j - 1

    if any(
        value < 0
        for value in assignment
    ):
        raise ValueError(
            "Incomplete Hungarian assignment."
        )

    return assignment


def assign_predictions_to_truth(
    predictions: EvaluationPredictionSet,
    truth: EvaluationTruth,
    *,
    config: EvaluationConfig,
) -> AssignmentResult:
    """
    Evaluator-only matching.

    IMPORTANT:
    No future truth point is read here.
    """

    prediction_tracks = tuple(
        sorted(
            predictions.tracks,
            key=lambda track:
            track.prediction_id,
        )
    )

    truth_tracks = tuple(
        sorted(
            truth.tracks,
            key=lambda track:
            track.truth_id,
        )
    )

    prediction_count = len(
        prediction_tracks
    )

    truth_count = len(
        truth_tracks
    )

    if (
        prediction_count == 0
        and
        truth_count == 0
    ):
        return AssignmentResult(
            matches=(),
            unmatched_prediction_ids=(),
            unmatched_truth_ids=(),
        )

    size = max(
        prediction_count,
        truth_count,
    )

    unmatched_cost = (
        config.anchor_assignment_gate_m
        +
        1e-6
    )

    forbidden_cost = (
        3.0
        *
        unmatched_cost
    )

    matrix = []

    for row in range(size):
        costs = []

        for column in range(size):
            if (
                row < prediction_count
                and
                column < truth_count
            ):
                distance = _distance_3d(
                    prediction_tracks[
                        row
                    ].anchor_position_H0_m,
                    truth_tracks[
                        column
                    ].anchor_position_H0_m,
                )

                if (
                    distance
                    <=
                    config
                    .anchor_assignment_gate_m
                ):
                    value = distance
                else:
                    value = (
                        forbidden_cost
                        +
                        distance
                    )

            elif (
                row >= prediction_count
                and
                column >= truth_count
            ):
                value = 0.0

            else:
                value = unmatched_cost

            costs.append(
                float(value)
            )

        matrix.append(costs)

    assignment = (
        _hungarian_square(
            matrix
        )
    )

    matches = []

    matched_prediction_ids = set()
    matched_truth_ids = set()

    for row, column in enumerate(
        assignment
    ):
        if (
            row >= prediction_count
            or
            column >= truth_count
        ):
            continue

        prediction = (
            prediction_tracks[row]
        )

        target = (
            truth_tracks[column]
        )

        distance = _distance_3d(
            prediction.anchor_position_H0_m,
            target.anchor_position_H0_m,
        )

        if (
            distance
            >
            config.anchor_assignment_gate_m
        ):
            continue

        matches.append(
            EvaluationMatch(
                prediction_id=(
                    prediction
                    .prediction_id
                ),
                truth_id=(
                    target.truth_id
                ),
                anchor_distance_m=(
                    distance
                ),
            )
        )

        matched_prediction_ids.add(
            prediction.prediction_id
        )

        matched_truth_ids.add(
            target.truth_id
        )

    matches.sort(
        key=lambda item: (
            item.truth_id,
            item.prediction_id,
        )
    )

    unmatched_predictions = tuple(
        sorted(
            track.prediction_id
            for track in (
                prediction_tracks
            )
            if (
                track.prediction_id
                not in
                matched_prediction_ids
            )
        )
    )

    unmatched_truth = tuple(
        sorted(
            track.truth_id
            for track in truth_tracks
            if (
                track.truth_id
                not in
                matched_truth_ids
            )
        )
    )

    return AssignmentResult(
        matches=tuple(matches),
        unmatched_prediction_ids=(
            unmatched_predictions
        ),
        unmatched_truth_ids=(
            unmatched_truth
        ),
        assignment_information=(
            "anchor_truth_only"
        ),
        future_truth_used_for_assignment=False,
    )
