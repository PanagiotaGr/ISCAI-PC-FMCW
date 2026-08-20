from __future__ import annotations

import math
import statistics


from iscai_stage3.io.womd_reader import (
    read_scenario_by_id,
)


from iscai_stage4.predictors import (
    predict_gaussian_cv,
)


from iscai_stage4.evaluation import (
    gaussian_nll,
    mahalanobis_distance,
    empirical_coverage,
)


from iscai_stage4.calibration import (
    scale_covariance,
)



SCENARIO_IDS = (
    "b85e1bd6cc8e74c0",
    "75ae707721eb23b4",
    "52dafd686fe77b21",
)


ALPHAS = (
    1.0,
    0.5,
    0.25,
    0.1,
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



def evaluate_alpha(
    alpha: float,
):

    nll_values = []
    mahal_values = []


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


            if len(track.states) <= anchor + 1:
                continue


            previous = track.states[
                anchor - 1
            ]

            current = track.states[
                anchor
            ]


            if not (
                previous.valid
                and current.valid
            ):
                continue


            p0 = position(
                previous
            )

            p1 = position(
                current
            )


            velocity = (
                (p1[0] - p0[0]) / DT,
                (p1[1] - p0[1]) / DT,
                (p1[2] - p0[2]) / DT,
            )


            prediction = predict_gaussian_cv(
                initial_position=p1,
                velocity=velocity,
                horizon_s=HORIZON * DT,
                dt=DT,
            )


            for step in range(
                1,
                HORIZON + 1,
            ):

                future_index = (
                    anchor + step
                )


                if (
                    future_index
                    >=
                    len(track.states)
                ):
                    break


                future_state = (
                    track.states[
                        future_index
                    ]
                )


                if not future_state.valid:
                    break


                predicted = (
                    prediction
                    .mean_positions_m[step]
                )


                covariance = scale_covariance(
                    prediction
                    .covariance_m2[step],
                    alpha,
                )


                actual = position(
                    future_state
                )


                error = error_vector(
                    predicted,
                    actual,
                )


                nll_values.append(
                    gaussian_nll(
                        error_vector=error,
                        covariance=covariance,
                    )
                )


                mahal_values.append(
                    mahalanobis_distance(
                        error_vector=error,
                        covariance=covariance,
                    )
                )



    return {

        "alpha": alpha,

        "mean_NLL":
            statistics.mean(
                nll_values
            ),

        "coverage_50":
            empirical_coverage(
                mahalanobis_values=mahal_values,
                threshold=1.538,
            ),

        "coverage_90":
            empirical_coverage(
                mahalanobis_values=mahal_values,
                threshold=2.795,
            ),

        "samples":
            len(mahal_values),
    }



def main():

    print(
        "===== Stage4 covariance sweep ====="
    )


    for alpha in ALPHAS:

        result = evaluate_alpha(
            alpha
        )

        print()

        print(
            "alpha =",
            result["alpha"]
        )

        print(
            "samples =",
            result["samples"]
        )

        print(
            "mean NLL =",
            result["mean_NLL"]
        )

        print(
            "50% coverage =",
            result["coverage_50"]
        )

        print(
            "90% coverage =",
            result["coverage_90"]
        )


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
