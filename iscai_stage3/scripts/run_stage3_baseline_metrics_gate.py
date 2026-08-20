from __future__ import annotations

import math
import statistics

from iscai_stage3.io.womd_reader import (
    read_scenario_by_id,
)

from iscai_stage3.baselines import (
    predict_constant_velocity,
    predict_constant_acceleration,
    predict_kalman_cv,
    predict_ctrv,
)


SCENARIO_IDS = (
    "b85e1bd6cc8e74c0",
    "75ae707721eb23b4",
    "52dafd686fe77b21",
)


DT = 0.1
HORIZON_STEPS = 10


def position(state):
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


def estimate_velocity(
    previous_position,
    current_position,
):
    return (
        (
            current_position[0]
            - previous_position[0]
        ) / DT,
        (
            current_position[1]
            - previous_position[1]
        ) / DT,
        (
            current_position[2]
            - previous_position[2]
        ) / DT,
    )


def estimate_acceleration(
    p_minus_2,
    p_minus_1,
    p_current,
):
    v_previous = estimate_velocity(
        p_minus_2,
        p_minus_1,
    )

    v_current = estimate_velocity(
        p_minus_1,
        p_current,
    )

    return (
        (
            v_current[0]
            - v_previous[0]
        ) / DT,
        (
            v_current[1]
            - v_previous[1]
        ) / DT,
        (
            v_current[2]
            - v_previous[2]
        ) / DT,
    )


def estimate_heading_and_yaw_rate(
    p_minus_2,
    p_minus_1,
    p_current,
):

    vx_previous = (
        p_minus_1[0]
        - p_minus_2[0]
    ) / DT

    vy_previous = (
        p_minus_1[1]
        - p_minus_2[1]
    ) / DT

    vx_current = (
        p_current[0]
        - p_minus_1[0]
    ) / DT

    vy_current = (
        p_current[1]
        - p_minus_1[1]
    ) / DT


    heading_previous = math.atan2(
        vy_previous,
        vx_previous,
    )

    heading_current = math.atan2(
        vy_current,
        vx_current,
    )


    delta = (
        heading_current
        - heading_previous
    )

    while delta > math.pi:
        delta -= 2.0 * math.pi

    while delta < -math.pi:
        delta += 2.0 * math.pi


    yaw_rate = delta / DT

    speed = math.hypot(
        vx_current,
        vy_current,
    )


    return (
        speed,
        heading_current,
        yaw_rate,
    )


def collect_future(
    track,
    anchor,
):

    future = []

    for step in range(
        1,
        HORIZON_STEPS + 1,
    ):

        index = anchor + step

        if index >= len(
            track.states
        ):
            break

        state = track.states[
            index
        ]

        if not state.valid:
            break

        future.append(
            position(state)
        )

    return tuple(future)


def predict_model(
    *,
    model_name,
    current_position,
    velocity,
    acceleration,
    speed,
    heading,
    yaw_rate,
    future_length,
):

    horizon_s = (
        future_length
        * DT
    )


    if model_name == "CV":

        return predict_constant_velocity(
            current_position,
            velocity,
            horizon_s,
            DT,
        )


    if model_name == "CA":

        return predict_constant_acceleration(
            current_position,
            velocity,
            acceleration,
            horizon_s,
            DT,
        )


    if model_name == "KalmanCV":

        return predict_kalman_cv(
            current_position,
            velocity,
            horizon_s,
            DT,
        )


    if model_name == "CTRV":

        return predict_ctrv(
            current_position,
            speed=speed,
            heading_rad=heading,
            yaw_rate=yaw_rate,
            horizon_s=horizon_s,
            dt=DT,
        )


    raise ValueError(
        f"Unknown model: {model_name}"
    )


def evaluate_model(
    model_name,
):

    actor_ade = []
    actor_fde = []

    total_steps = 0
    evaluated_actors = 0


    for scenario_id in SCENARIO_IDS:

        (
            scenario,
            _,
            _,
            _,
        ) = read_scenario_by_id(
            scenario_id
        )

        anchor = int(
            scenario.current_time_index
        )


        if anchor < 2:
            continue


        for track_index, track in enumerate(
            scenario.tracks
        ):

            if (
                track_index
                ==
                int(
                    scenario.sdc_track_index
                )
            ):
                continue


            if len(
                track.states
            ) <= anchor:
                continue


            s_minus_2 = track.states[
                anchor - 2
            ]

            s_minus_1 = track.states[
                anchor - 1
            ]

            s_current = track.states[
                anchor
            ]


            if not (
                s_minus_2.valid
                and s_minus_1.valid
                and s_current.valid
            ):
                continue


            p_minus_2 = position(
                s_minus_2
            )

            p_minus_1 = position(
                s_minus_1
            )

            p_current = position(
                s_current
            )


            velocity = estimate_velocity(
                p_minus_1,
                p_current,
            )


            acceleration = (
                estimate_acceleration(
                    p_minus_2,
                    p_minus_1,
                    p_current,
                )
            )


            (
                speed,
                heading,
                yaw_rate,
            ) = (
                estimate_heading_and_yaw_rate(
                    p_minus_2,
                    p_minus_1,
                    p_current,
                )
            )


            future = collect_future(
                track,
                anchor,
            )


            if len(future) == 0:
                continue


            prediction = predict_model(
                model_name=model_name,
                current_position=p_current,
                velocity=velocity,
                acceleration=acceleration,
                speed=speed,
                heading=heading,
                yaw_rate=yaw_rate,
                future_length=len(future),
            )


            if len(prediction) != len(
                future
            ):
                raise RuntimeError(
                    f"{model_name}: prediction "
                    "length mismatch."
                )


            errors = [
                distance(
                    predicted,
                    actual,
                )
                for predicted, actual in zip(
                    prediction,
                    future,
                )
            ]


            actor_ade.append(
                statistics.mean(
                    errors
                )
            )

            actor_fde.append(
                errors[-1]
            )


            total_steps += len(
                errors
            )

            evaluated_actors += 1


    if evaluated_actors == 0:
        raise RuntimeError(
            f"No actors evaluated for "
            f"{model_name}."
        )


    return {
        "model": model_name,
        "actors": evaluated_actors,
        "steps": total_steps,
        "ADE_m": statistics.mean(
            actor_ade
        ),
        "FDE_m": statistics.mean(
            actor_fde
        ),
    }


def main():

    print(
        "===== Stage3 baseline metrics ====="
    )


    models = (
        "CV",
        "CA",
        "KalmanCV",
        "CTRV",
    )


    results = []


    for model_name in models:

        result = evaluate_model(
            model_name
        )

        results.append(
            result
        )


        print()
        print(
            "model =",
            result["model"]
        )

        print(
            "actors =",
            result["actors"]
        )

        print(
            "future steps =",
            result["steps"]
        )

        print(
            "mean ADE [m] =",
            result["ADE_m"]
        )

        print(
            "mean FDE [m] =",
            result["FDE_m"]
        )


    best = min(
        results,
        key=lambda item:
            item["ADE_m"],
    )


    print()
    print(
        "BEST CLASSICAL BASELINE =",
        best["model"]
    )

    print(
        "best ADE [m] =",
        best["ADE_m"]
    )

    print(
        "best FDE [m] =",
        best["FDE_m"]
    )

    print(
        "predictor future input = NO"
    )

    print(
        "future ground truth "
        "used for evaluation = YES"
    )

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()
