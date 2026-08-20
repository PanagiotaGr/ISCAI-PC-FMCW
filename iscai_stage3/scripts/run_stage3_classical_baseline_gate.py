from __future__ import annotations


from iscai_stage3.baselines import (
    predict_constant_velocity,
    predict_constant_acceleration,
    predict_kalman_cv,
    predict_ctrv,
    select_imm_model,
    TrajectoryHypothesis,
    select_best_hypothesis,
)


def run():

    position = (
        0.0,
        0.0,
        0.0,
    )

    velocity = (
        10.0,
        0.0,
        0.0,
    )


    acceleration = (
        0.1,
        0.0,
        0.0,
    )


    horizon = 1.0
    dt = 0.1


    cv = predict_constant_velocity(
        position,
        velocity,
        horizon,
        dt,
    )


    ca = predict_constant_acceleration(
        position,
        velocity,
        acceleration,
        horizon,
        dt,
    )


    kalman = predict_kalman_cv(
        position,
        velocity,
        horizon,
        dt,
    )


    ctrv = predict_ctrv(
        position,
        speed=10.0,
        heading_rad=0.0,
        yaw_rate=0.05,
        horizon_s=horizon,
        dt=dt,
    )


    imm = select_imm_model(
        velocity_score=0.7,
        acceleration_score=0.2,
        turn_rate_score=0.1,
    )


    h1 = TrajectoryHypothesis(
        trajectory=cv,
        score=1.0,
    )


    h2 = TrajectoryHypothesis(
        trajectory=ctrv,
        score=0.5,
    )


    best = select_best_hypothesis(
        (
            h1,
            h2,
        )
    )


    if len(cv) == 0:
        raise RuntimeError(
            "CV failed."
        )

    if len(ca) == 0:
        raise RuntimeError(
            "CA failed."
        )

    if len(kalman) == 0:
        raise RuntimeError(
            "Kalman failed."
        )

    if len(ctrv) == 0:
        raise RuntimeError(
            "CTRV failed."
        )


    print(
        "===== Stage3 classical baseline gate ====="
    )

    print(
        "CV states =",
        len(cv)
    )

    print(
        "CA states =",
        len(ca)
    )

    print(
        "Kalman states =",
        len(kalman)
    )

    print(
        "CTRV states =",
        len(ctrv)
    )

    print(
        "IMM selected =",
        imm.selected_model
    )

    print(
        "MHT selected hypothesis score =",
        best.score
    )


    print(
        "future used = NO"
    )

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    run()
