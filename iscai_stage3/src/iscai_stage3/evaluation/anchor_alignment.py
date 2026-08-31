from __future__ import annotations

from math import isfinite

from iscai_stage3.evaluation.contracts import (
    EvaluationPredictionTrack,
    PredictionTrajectoryPoint,
)
from iscai_stage3.filters.imm import (
    combine_imm_states,
    interact_mode_states,
    predict_imm_mode_state,
)
from iscai_stage3.filters.kalman_ekf import (
    predict_kalman_state,
)


def _normalized_horizons(horizons_s):
    result = tuple(float(x) for x in horizons_s)
    if not result:
        raise ValueError("At least one horizon is required.")
    if any((not isfinite(x)) or x <= 0.0 for x in result):
        raise ValueError("Horizons must be positive and finite.")
    if any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError("Horizons must strictly increase.")
    return result


def align_state_prediction_to_global_anchor(
    state,
    predictor,
    *,
    horizons_s,
    anchor_timestamp_s: float,
    tolerance_s: float = 2e-3,
):
    """
    Align CV/CA/CTRV-style state forecasts to the global WOMD anchor.

    A still-active track may have its latest detection one or two frames
    before the anchor.  We causally propagate from that last state to the
    global anchor and then to anchor+horizon, without reading future truth.
    """
    horizons = _normalized_horizons(horizons_s)
    anchor = float(anchor_timestamp_s)
    state_time = float(state.timestamp_s)
    delta = anchor - state_time

    if delta < -tolerance_s:
        raise ValueError("State timestamp is after the global anchor.")

    if abs(delta) <= 1e-12:
        raw = predictor(state, horizons_s=horizons)
        anchor_position = tuple(float(x) for x in state.position_H0_m)
        future_points = tuple(raw.points)
    else:
        if delta <= 0.0:
            raise ValueError("State/global-anchor ordering is invalid.")

        extended = (
            delta,
            *tuple(delta + h for h in horizons),
        )
        raw = predictor(state, horizons_s=extended)

        if len(raw.points) != len(horizons) + 1:
            raise ValueError("Unexpected aligned prediction point count.")

        anchor_position = tuple(
            float(x)
            for x in raw.points[0].position_H0_m
        )
        future_points = tuple(raw.points[1:])

    if len(future_points) != len(horizons):
        raise ValueError("Unexpected future prediction point count.")

    converted = []
    for horizon, point in zip(horizons, future_points):
        expected_timestamp = anchor + horizon
        if abs(float(point.timestamp_s) - expected_timestamp) > tolerance_s:
            raise ValueError(
                "State-model forecast did not land on global anchor+horizon."
            )

        converted.append(
            PredictionTrajectoryPoint(
                horizon_s=horizon,
                timestamp_s=expected_timestamp,
                position_H0_m=tuple(
                    float(x)
                    for x in point.position_H0_m
                ),
            )
        )

    return EvaluationPredictionTrack(
        prediction_id=str(raw.track_id),
        anchor_position_H0_m=anchor_position,
        points=tuple(converted),
    )


def align_kalman_state_to_global_anchor(
    state,
    *,
    anchor_timestamp_s: float,
    config,
    tolerance_s: float = 2e-3,
):
    anchor = float(anchor_timestamp_s)
    current = float(state.timestamp_s)

    if current > anchor + tolerance_s:
        raise ValueError("Kalman state timestamp is after global anchor.")

    if abs(current - anchor) <= 1e-12:
        return state

    if current >= anchor:
        raise ValueError("Kalman state/global-anchor ordering is invalid.")

    return predict_kalman_state(
        state,
        target_timestamp_s=anchor,
        config=config,
    )


def align_imm_states_to_global_anchor(
    states,
    mode_probabilities,
    *,
    frame_timestamps_s,
    anchor_timestamp_s: float,
    config,
    tolerance_s: float = 2e-3,
):
    """Causal open-loop IMM propagation through missed anchor frames."""
    current_states = tuple(states)
    current_probabilities = tuple(float(x) for x in mode_probabilities)

    state_times = {float(state.timestamp_s) for state in current_states}
    if len(state_times) != 1:
        raise ValueError("IMM mode states are not time-aligned.")

    current_time = float(current_states[0].timestamp_s)
    anchor = float(anchor_timestamp_s)

    if current_time > anchor + tolerance_s:
        raise ValueError("IMM state timestamp is after global anchor.")

    frame_times = tuple(float(x) for x in frame_timestamps_s)
    subsequent = tuple(
        t
        for t in frame_times
        if t > current_time + 1e-12
        and t <= anchor + tolerance_s
    )

    for target_timestamp in subsequent:
        mixed_states, predicted_probabilities, _ = interact_mode_states(
            current_states,
            current_probabilities,
            config=config,
        )

        current_states = tuple(
            predict_imm_mode_state(
                state,
                target_timestamp_s=target_timestamp,
                config=config,
            )
            for state in mixed_states
        )
        current_probabilities = tuple(predicted_probabilities)

    final_time = float(current_states[0].timestamp_s)
    if abs(final_time - anchor) > tolerance_s:
        raise ValueError(
            "IMM open-loop propagation did not reach the global anchor."
        )

    combined = combine_imm_states(
        current_states,
        current_probabilities,
    )
    anchor_position = tuple(
        float(combined.mean_10[i])
        for i in range(3)
    )

    return current_states, current_probabilities, anchor_position
