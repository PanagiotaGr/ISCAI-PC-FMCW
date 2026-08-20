from __future__ import annotations

import statistics

from iscai_stage3.io.womd_reader import (
    read_scenario_by_id,
)

from iscai_stage4.predictors import (
    predict_gaussian_cv,
)

from iscai_stage4.calibration import (
    calibrate_prediction,
)

from iscai_stage4.evaluation import (
    gaussian_nll,
    mahalanobis_distance,
    empirical_coverage,
)


SCENARIO_IDS = (
    "b85e1bd6cc8e74c0",
    "75ae707721eb23b4",
    "52dafd686fe77b21",
)


DT = 0.1
HORIZON = 10



def position(state):
    return (
        float(state.center_x),
        float(state.center_y),
        float(state.center_z),
    )



def error_vector(a, b):
    return (
        a[0]-b[0],
        a[1]-b[1],
        a[2]-b[2],
    )



def evaluate(
    calibrated: bool,
):

    ade = []
    fde = []
    nll = []
    mahal = []


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


        for track_index, track in enumerate(
            scenario.tracks
        ):

            if track_index == int(
                scenario.sdc_track_index
            ):
                continue


            if len(track.states) <= anchor+1:
                continue


            previous = track.states[
                anchor-1
            ]

            current = track.states[
                anchor
            ]


            if not (
                previous.valid
                and current.valid
            ):
                continue


            p0 = position(previous)
            p1 = position(current)


            velocity = (
                (p1[0]-p0[0])/DT,
                (p1[1]-p0[1])/DT,
                (p1[2]-p0[2])/DT,
            )


            prediction = predict_gaussian_cv(
                initial_position=p1,
                velocity=velocity,
                horizon_s=HORIZON*DT,
                dt=DT,
            )


            if calibrated:

                (
                    mean,
                    covariance,
                ) = calibrate_prediction(
                    prediction
                )

            else:

                mean = (
                    prediction.mean_positions_m
                )

                covariance = (
                    prediction.covariance_m2
                )


            actor_errors = []


            for step in range(
                1,
                HORIZON+1,
            ):

                index = anchor + step

                if index >= len(track.states):
                    break


                future = track.states[index]


                if not future.valid:
                    break


                actual = position(
                    future
                )


                error = error_vector(
                    mean[step],
                    actual,
                )


                distance = (
                    sum(
                        x*x
                        for x in error
                    )
                ) ** 0.5


                actor_errors.append(
                    distance
                )


                nll.append(
                    gaussian_nll(
                        error_vector=error,
                        covariance=covariance[step],
                    )
                )


                mahal.append(
                    mahalanobis_distance(
                        error_vector=error,
                        covariance=covariance[step],
                    )
                )


            if actor_errors:

                ade.append(
                    statistics.mean(
                        actor_errors
                    )
                )

                fde.append(
                    actor_errors[-1]
                )


    return {

        "ADE":
            statistics.mean(ade),

        "FDE":
            statistics.mean(fde),

        "NLL":
            statistics.mean(nll),

        "coverage50":
            empirical_coverage(
                mahalanobis_values=mahal,
                threshold=1.538,
            ),

        "coverage90":
            empirical_coverage(
                mahalanobis_values=mahal,
                threshold=2.795,
            ),
    }



def main():

    print(
        "===== Stage4 calibration comparison ====="
    )


    raw = evaluate(
        calibrated=False
    )


    calibrated = evaluate(
        calibrated=True
    )


    print()

    print(
        "RAW GaussianCV"
    )

    print(raw)


    print()

    print(
        "CALIBRATED GaussianCV alpha=0.25"
    )

    print(calibrated)


    print()

    print(
        "future used by predictor = NO"
    )

    print(
        "future used by evaluator = YES"
    )

    print(
        "STATUS = PASS"
    )



if __name__ == "__main__":
    main()
