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

from iscai_stage3.pipeline.smoke import (
    evaluate_observation_series,
    select_target,
)


SCENARIO_ID = (
    "b85e1bd6cc8e74c0"
)


MEASUREMENT_COVARIANCE = (
    (0.01, 0.0, 0.0),
    (0.0, 0.001, 0.0),
    (0.0, 0.0, 0.001),
)


def main():

    print(
        "===== Stage3 real frozen WOMD gate ====="
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


    evaluated = []

    valid_actors = 0
    skipped_actors = 0
    classes = {}



    for actor in scene.actors:

        try:

            results = (
                evaluate_observation_series(
                    series=actor,
                    measurement_covariance=(
                        MEASUREMENT_COVARIANCE
                    ),
                )
            )


            evaluated.extend(
                results
            )


            valid_actors += 1


            classes[
                actor.object_class
            ] = (
                classes.get(
                    actor.object_class,
                    0,
                )
                + 1
            )


        except ValueError:

            skipped_actors += 1



    if len(evaluated) == 0:

        raise RuntimeError(
            "No valid Stage3 observations."
        )



    print(
        "----- candidate targets -----"
    )


    for item in evaluated:

        print(
            "id=",
            item.target_id,
            "confidence=",
            item.beam_confidence,
            "uncertainty=",
            item.uncertainty_penalty,
            "utility=",
            item.utility,
        )



    selected = select_target(
        evaluated
    )



    print()
    print(
        "scenario =",
        scenario.scenario_id
    )

    print(
        "actors =",
        len(scene.actors)
    )

    print(
        "valid actors =",
        valid_actors
    )

    print(
        "skipped actors =",
        skipped_actors
    )

    print(
        "evaluations =",
        len(evaluated)
    )

    print(
        "classes =",
        classes
    )

    print(
        "selected target =",
        selected.target_id
    )

    print(
        "beam confidence =",
        selected.beam_confidence
    )

    print(
        "uncertainty =",
        selected.uncertainty_penalty
    )

    print(
        "utility =",
        selected.utility
    )

    print(
        "future used = NO"
    )

    print(
        "measured FMCW = NO"
    )

    print(
        "STATUS = PASS"
    )



if __name__ == "__main__":
    main()
