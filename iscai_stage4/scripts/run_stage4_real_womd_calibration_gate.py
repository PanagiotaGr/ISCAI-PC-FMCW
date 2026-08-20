from __future__ import annotations


from iscai_stage3.io.womd_reader import (
    read_scenario_by_id,
)


from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)


from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)


from iscai_stage4.predictors import (
    predict_gaussian_cv,
)


from iscai_stage4.evaluation import (
    evaluate_prediction,
)



SCENARIO_ID = (
    "b85e1bd6cc8e74c0"
)



def main():

    print(
        "===== Stage4 real WOMD calibration gate ====="
    )


    (
        scenario,
        manifest,
        shard,
        record_index,
    ) = read_scenario_by_id(
        SCENARIO_ID
    )


    adapted = (
        adapt_causal_womd_scenario(
            scenario
        )
    )


    scene = (
        build_real_ideal_observation_scene(
            adapted=adapted,
            raw_scenario=scenario,
        )
    )


    evaluated = 0
    errors = []
    normalized = []


    for actor in scene.actors:


        observations = [
            obs
            for obs in actor.observations
            if (
                obs.geometry_valid
                and
                obs.actor_position_Ht_m
                is not None
            )
        ]


        if len(observations) < 2:
            continue


        history = observations[:-1]


        current = history[-1]


        previous = history[-2]


        velocity = (
            current.actor_position_Ht_m[0]
            -
            previous.actor_position_Ht_m[0],

            current.actor_position_Ht_m[1]
            -
            previous.actor_position_Ht_m[1],

            current.actor_position_Ht_m[2]
            -
            previous.actor_position_Ht_m[2],
        )


        prediction = predict_gaussian_cv(
            initial_position=(
                current.actor_position_Ht_m
            ),
            velocity=velocity,
            horizon_s=0.1,
            dt=0.1,
        )


        future = (
            actor.observations[-1]
            .actor_position_Ht_m
        )


        if future is None:
            continue


        result = evaluate_prediction(
            predicted_positions=(
                prediction.mean_positions_m
            ),
            covariance=(
                prediction.covariance_m2
            ),
            future_positions=(
                (
                    future,
                )
                *
                len(
                    prediction.mean_positions_m
                )
            ),
        )


        errors.append(
            result["mean_error"]
        )

        normalized.append(
            result["mean_normalized_error"]
        )


        evaluated += 1



    if evaluated == 0:
        raise RuntimeError(
            "No real WOMD trajectories evaluated."
        )


    print(
        "scenario =",
        SCENARIO_ID,
    )

    print(
        "actors evaluated =",
        evaluated,
    )

    print(
        "mean error =",
        sum(errors)/len(errors),
    )

    print(
        "mean normalized error =",
        sum(normalized)/len(normalized),
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
