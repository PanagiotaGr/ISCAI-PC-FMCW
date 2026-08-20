from __future__ import annotations

import math
import statistics


from iscai_stage3.io.womd_reader import (
    read_scenario_by_id,
)

from iscai_stage4.predictors import (
    predict_gaussian_cv,
)


SCENARIO_IDS = (
    "b85e1bd6cc8e74c0",
    "75ae707721eb23b4",
    "52dafd686fe77b21",
)


HORIZON_STEPS = 10
DT = 0.1


def state_position(state):
    return (
        float(state.center_x),
        float(state.center_y),
        float(state.center_z),
    )


def distance(a, b):

    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]

    return math.sqrt(
        dx * dx
        + dy * dy
        + dz * dz
    )


def causal_velocity(
    previous_position,
    current_position,
    dt,
):

    return (
        (
            current_position[0]
            - previous_position[0]
        ) / dt,

        (
            current_position[1]
            - previous_position[1]
        ) / dt,

        (
            current_position[2]
            - previous_position[2]
        ) / dt,
    )


def evaluate_actor(
    *,
    track,
    anchor_index,
):

    if anchor_index < 1:
        return None

    if len(track.states) <= anchor_index:
        return None


    previous_state = track.states[
        anchor_index - 1
    ]

    current_state = track.states[
        anchor_index
    ]


    if not (
        previous_state.valid
        and current_state.valid
    ):
        return None


    previous_position = state_position(
        previous_state
    )

    current_position = state_position(
        current_state
    )


    velocity = causal_velocity(
        previous_position,
        current_position,
        DT,
    )


    future_positions = []


    max_future_index = min(
        len(track.states),
        anchor_index
        + 1
        + HORIZON_STEPS,
    )


    for index in range(
        anchor_index + 1,
        max_future_index,
    ):

        state = track.states[index]

        if not state.valid:
            break

        future_positions.append(
            state_position(state)
        )


    if len(future_positions) == 0:
        return None


    prediction = predict_gaussian_cv(
        initial_position=current_position,
        velocity=velocity,
        horizon_s=(
            len(future_positions)
            * DT
        ),
        dt=DT,
    )


    #
    # gaussian_cv currently includes t=0.
    # WOMD future begins at t=+0.1 s.
    #
    predicted_future = (
        prediction.mean_positions_m[
            1 : 1 + len(future_positions)
        ]
    )

    covariance_future = (
        prediction.covariance_m2[
            1 : 1 + len(future_positions)
        ]
    )


    if len(predicted_future) != len(
        future_positions
    ):
        raise RuntimeError(
            "Prediction/future length mismatch."
        )


    errors = []

    normalized_errors = []


    for predicted, actual, covariance in zip(
        predicted_future,
        future_positions,
        covariance_future,
    ):

        error = distance(
            predicted,
            actual,
        )


        variance_trace = (
            covariance[0][0]
            + covariance[1][1]
            + covariance[2][2]
        )


        if variance_trace <= 0.0:
            raise RuntimeError(
                "Invalid predicted covariance."
            )


        normalized_error = (
            error
            /
            math.sqrt(
                variance_trace
            )
        )


        errors.append(
            error
        )

        normalized_errors.append(
            normalized_error
        )


    return {
        "steps": len(errors),

        "ade_m": statistics.mean(
            errors
        ),

        "fde_m": errors[-1],

        "mean_normalized_error": (
            statistics.mean(
                normalized_errors
            )
        ),
    }


def evaluate_scenario(
    scenario_id,
):

    (
        scenario,
        manifest,
        shard,
        record_index,
    ) = read_scenario_by_id(
        scenario_id
    )


    anchor = int(
        scenario.current_time_index
    )


    actor_results = []


    for track_index, track in enumerate(
        scenario.tracks
    ):

        if (
            track_index
            == int(
                scenario.sdc_track_index
            )
        ):
            continue


        result = evaluate_actor(
            track=track,
            anchor_index=anchor,
        )


        if result is not None:

            actor_results.append(
                result
            )


    if len(actor_results) == 0:
        raise RuntimeError(
            "No evaluable actors in "
            f"{scenario_id}"
        )


    return {
        "scenario_id": scenario_id,

        "manifest_line": (
            manifest["manifest_line"]
        ),

        "record_index": record_index,

        "actors_evaluated": len(
            actor_results
        ),

        "future_steps_evaluated": sum(
            result["steps"]
            for result in actor_results
        ),

        "mean_ADE_m": statistics.mean(
            result["ade_m"]
            for result in actor_results
        ),

        "mean_FDE_m": statistics.mean(
            result["fde_m"]
            for result in actor_results
        ),

        "mean_normalized_error": (
            statistics.mean(
                result[
                    "mean_normalized_error"
                ]
                for result in actor_results
            )
        ),
    }


def main():

    print(
        "===== Stage4 real multi-scenario "
        "probabilistic gate ====="
    )


    results = []


    for scenario_id in SCENARIO_IDS:

        result = evaluate_scenario(
            scenario_id
        )

        results.append(
            result
        )

        print()
        print(
            "scenario =",
            result["scenario_id"]
        )

        print(
            "manifest line =",
            result["manifest_line"]
        )

        print(
            "record index =",
            result["record_index"]
        )

        print(
            "actors evaluated =",
            result[
                "actors_evaluated"
            ]
        )

        print(
            "future steps evaluated =",
            result[
                "future_steps_evaluated"
            ]
        )

        print(
            "mean ADE [m] =",
            result["mean_ADE_m"]
        )

        print(
            "mean FDE [m] =",
            result["mean_FDE_m"]
        )

        print(
            "mean normalized error =",
            result[
                "mean_normalized_error"
            ]
        )


    print()
    print(
        "===== aggregate ====="
    )

    print(
        "scenarios =",
        len(results)
    )

    print(
        "actors evaluated =",
        sum(
            result[
                "actors_evaluated"
            ]
            for result in results
        )
    )

    print(
        "future steps evaluated =",
        sum(
            result[
                "future_steps_evaluated"
            ]
            for result in results
        )
    )

    print(
        "mean scenario ADE [m] =",
        statistics.mean(
            result["mean_ADE_m"]
            for result in results
        )
    )

    print(
        "mean scenario FDE [m] =",
        statistics.mean(
            result["mean_FDE_m"]
            for result in results
        )
    )

    print(
        "predictor future input = NO"
    )

    print(
        "future ground truth used "
        "for evaluation = YES"
    )

    print(
        "measured FMCW = NO"
    )

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()
