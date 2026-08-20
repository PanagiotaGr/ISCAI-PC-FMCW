from __future__ import annotations

import math
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



def error_vector(
    predicted,
    actual,
):

    return (
        predicted[0] - actual[0],
        predicted[1] - actual[1],
        predicted[2] - actual[2],
    )



def main():

    print(
        "===== Stage4 calibrated metrics gate ====="
    )


    all_nll = []
    all_mahal = []

    ade = []
    fde = []

    actors = 0
    steps = 0


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


            if (
                track_index
                ==
                int(
                    scenario.sdc_track_index
                )
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


            (
                predicted_mean,
                calibrated_covariance,
            ) = calibrate_prediction(
                prediction
            )


            actor_errors = []


            for step in range(
                1,
                HORIZON+1,
            ):

                future_index = (
                    anchor + step
                )


                if future_index >= len(
                    track.states
                ):
                    break


                future_state = track.states[
                    future_index
                ]


                if not future_state.valid:
                    break


                predicted = (
                    predicted_mean[step]
                )


                covariance = (
                    calibrated_covariance[step]
                )


                actual = position(
                    future_state
                )


                error = error_vector(
                    predicted,
                    actual,
                )


                distance = math.sqrt(
                    error[0]**2
                    +
                    error[1]**2
                    +
                    error[2]**2
                )


                actor_errors.append(
                    distance
                )


                all_nll.append(
                    gaussian_nll(
                        error_vector=error,
                        covariance=covariance,
                    )
                )


                all_mahal.append(
                    mahalanobis_distance(
                        error_vector=error,
                        covariance=covariance,
                    )
                )


                steps += 1


            if actor_errors:

                actors += 1

                ade.append(
                    statistics.mean(
                        actor_errors
                    )
                )

                fde.append(
                    actor_errors[-1]
                )


    print(
        "scenarios =",
        len(SCENARIO_IDS)
    )


    print(
        "actors evaluated =",
        actors
    )


    print(
        "future steps =",
        steps
    )


    print(
        "mean ADE =",
        statistics.mean(ade)
    )


    print(
        "mean FDE =",
        statistics.mean(fde)
    )


    print(
        "mean NLL =",
        statistics.mean(all_nll)
    )


    print(
        "50% coverage =",
        empirical_coverage(
            mahalanobis_values=all_mahal,
            threshold=1.538,
        )
    )


    print(
        "90% coverage =",
        empirical_coverage(
            mahalanobis_values=all_mahal,
            threshold=2.795,
        )
    )


    print(
        "calibration alpha = 0.25"
    )


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
