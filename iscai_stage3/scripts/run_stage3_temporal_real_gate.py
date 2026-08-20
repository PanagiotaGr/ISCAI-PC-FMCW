from __future__ import annotations


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
    select_target,
)


from iscai_stage3.control.temporal_adb import (
    build_temporal_schedule,
)



MOTION = Path(
    "/home/agni/waymo/data/paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)


SCENARIO_ID = (
    "b85e1bd6cc8e74c0"
)


IDENTITY3 = (
    (0.001, 0.0, 0.0),
    (0.0, 0.001, 0.0),
    (0.0, 0.0, 0.001),
)



def main():

    print(
        "===== Stage3 temporal real WOMD gate ====="
    )


    scenario = read_first_scenario(
        MOTION
    )


    if scenario.scenario_id != SCENARIO_ID:
        raise RuntimeError(
            "Scenario mismatch"
        )


    adapted = adapt_causal_womd_scenario(
        scenario
    )


    scene = build_real_ideal_observation_scene(
        adapted=adapted,
        raw_scenario=scenario,
    )


    all_targets = []

    valid_actors = 0

    skipped_actors = 0


    for actor in scene.actors:

        if actor.is_sdc:
            continue


        try:

            evaluated = evaluate_observation_series(
                series=actor,
                measurement_covariance=IDENTITY3,
            )


        except ValueError:

            skipped_actors += 1
            continue


        valid_actors += 1


        target = select_target(
            evaluated
        )


        all_targets.append(
            target
        )


    if len(all_targets) == 0:
        raise RuntimeError(
            "No valid targets found."
        )


    sequence = build_temporal_schedule(
        all_targets,
        horizon_s=1.0,
        dt=0.1,
    )


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
        "targets evaluated =",
        len(all_targets)
    )


    print(
        "temporal steps =",
        len(sequence.steps)
    )


    print(
        "selected sequence =",
        [
            step.target_id
            for step in sequence.steps
        ]
    )


    print(
        "uncertainty sequence =",
        [
            step.uncertainty
            for step in sequence.steps
        ]
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
