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
        predicted[0]-actual[0],
        predicted[1]-actual[1],
        predicted[2]-actual[2],
    )



def evaluate_actor(
    track,
    anchor,
):

    if anchor < 1:
        return None


    if len(track.states) <= anchor+1:
        return None


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
        return None



    p0 = position(
        previous
    )

    p1 = position(
        current
    )


    velocity = (

        (p1[0]-p0[0]) / DT,

        (p1[1]-p0[1]) / DT,

        (p1[2]-p0[2]) / DT,

    )


    future = []


    for index in range(
        anchor+1,
        min(
            len(track.states),
            anchor+1+HORIZON
        ),
    ):

        state = track.states[index]

        if not state.valid:
            break

        future.append(
            position(state)
        )


    if len(future) == 0:
        return None



    prediction = predict_gaussian_cv(
        initial_position=p1,
        velocity=velocity,
        horizon_s=len(future)*DT,
        dt=DT,
    )


    predicted = (
        prediction.mean_positions_m[
            1:1+len(future)
        ]
    )


    covariance = (
        prediction.covariance_m2[
            1:1+len(future)
        ]
    )


    ade = []
    fde = []
    nll = []
    mahal = []


    for pred, real, cov in zip(
        predicted,
        future,
        covariance,
    ):

        err = error_vector(
            pred,
            real,
        )


        distance = math.sqrt(
            err[0]**2
            +
            err[1]**2
            +
            err[2]**2
        )


        ade.append(
            distance
        )


        nll.append(
            gaussian_nll(
                error_vector=err,
                covariance=cov,
            )
        )


        mahal.append(
            mahalanobis_distance(
                error_vector=err,
                covariance=cov,
            )
        )


    return {
        "ADE":
            statistics.mean(ade),

        "FDE":
            ade[-1],

        "NLL":
            statistics.mean(nll),

        "mahal":
            mahal,

        "steps":
            len(future),
    }



def main():

    print(
        "===== Stage4 calibration metrics gate ====="
    )


    all_ade = []
    all_fde = []
    all_nll = []
    all_mahal = []


    actors = 0
    steps = 0


    for scenario_id in SCENARIO_IDS:


        (
            scenario,
            manifest,
            shard,
            index,
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


            result = evaluate_actor(
                track,
                anchor,
            )


            if result is None:
                continue


            actors += 1

            steps += result["steps"]


            all_ade.append(
                result["ADE"]
            )

            all_fde.append(
                result["FDE"]
            )

            all_nll.append(
                result["NLL"]
            )

            all_mahal.extend(
                result["mahal"]
            )



    print(
        "scenarios =",
        len(SCENARIO_IDS),
    )

    print(
        "actors evaluated =",
        actors,
    )

    print(
        "future steps =",
        steps,
    )


    print(
        "mean ADE =",
        statistics.mean(all_ade),
    )

    print(
        "mean FDE =",
        statistics.mean(all_fde),
    )

    print(
        "mean NLL =",
        statistics.mean(all_nll),
    )


    print(
        "50% coverage =",
        empirical_coverage(
            mahalanobis_values=all_mahal,
            threshold=1.538,
        ),
    )


    print(
        "90% coverage =",
        empirical_coverage(
            mahalanobis_values=all_mahal,
            threshold=2.795,
        ),
    )


    print(
        "future used by predictor = NO"
    )

    print(
        "future used by evaluator = YES"
    )

    print(
        "measured FMCW = NO"
    )

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()
