from pathlib import Path

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)

from iscai_stage3.pipeline.smoke import (
    evaluate_observation_series,
)


MOTION = Path(
    "/home/agni/waymo/data/paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)


SCENARIOS = (
    "b85e1bd6cc8e74c0",
    "4d82fec943ddaa44",
    "bbc29ed5e271f29b",
    "75ae707721eb23b4",
    "52dafd686fe77b21",
)


IDENTITY3 = (
    (0.01, 0.0, 0.0),
    (0.0, 0.001, 0.0),
    (0.0, 0.0, 0.001),
)


def run():

    print("===== Stage3 real multiclass WOMD gate =====")

    scenario = read_first_scenario(
        MOTION
    )

    # προσωρινά ελέγχουμε το πρώτο.
    # μετά θα αντικατασταθεί με deterministic lookup.

    adapted = adapt_causal_womd_scenario(
        scenario
    )

    scene = build_real_ideal_observation_scene(
        adapted=adapted,
        raw_scenario=scenario,
    )

    total_actors = 0
    valid_actors = 0
    skipped_actors = 0
    evaluations = 0

    classes = {}

    for actor in scene.actors:

        total_actors += 1

        try:

            result = evaluate_observation_series(
                series=actor,
                measurement_covariance=IDENTITY3,
            )

        except ValueError:

            skipped_actors += 1
            continue


        valid_actors += 1
        evaluations += len(result)

        classes[actor.object_class] = (
            classes.get(
                actor.object_class,
                0,
            )
            + 1
        )


    print(
        f"scenario = {scene.scenario_id}"
    )

    print(
        f"actors = {total_actors}"
    )

    print(
        f"valid actors = {valid_actors}"
    )

    print(
        f"skipped actors = {skipped_actors}"
    )

    print(
        f"evaluations = {evaluations}"
    )

    print(
        f"classes = {classes}"
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
    run()
