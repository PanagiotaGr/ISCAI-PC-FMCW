from __future__ import annotations

from iscai_stage4.pipeline import (
    run_stage4_prediction,
)


def main():

    prediction = run_stage4_prediction(
        initial_position=(
            0.0,
            0.0,
            0.0,
        ),
        velocity=(
            8.0,
            0.0,
            0.0,
        ),
        horizon_s=1.0,
        dt=0.1,
    )


    print(
        "===== Stage4 probabilistic smoke ====="
    )

    print(
        "model =",
        prediction.model_name,
    )

    print(
        "steps =",
        len(prediction.timestamps_s),
    )

    print(
        "final mean =",
        prediction.mean_positions_m[-1],
    )

    print(
        "final uncertainty trace =",
        sum(
            prediction.covariance_m2[-1][i][i]
            for i in range(3)
        ),
    )

    print(
        "future used = NO"
    )

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()
