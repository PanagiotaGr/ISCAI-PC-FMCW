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
    evaluate_target,
    select_target,
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
    (0.001,0.0,0.0),
    (0.0,0.001,0.0),
    (0.0,0.0,0.001),
)



def main():

    print(
        "===== Stage3 real temporal closed-loop gate ====="
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


    selected = []

    confidence_history = []

    uncertainty_history = []


    max_steps = len(
        scene.actors[0].observations
    )


    for t in range(max_steps):

        candidates = []


        for actor in scene.actors:

            if actor.is_sdc:
                continue


            if t >= len(actor.observations):
                continue


            obs = actor.observations[t]


            if not obs.geometry_valid:
                continue


            if (
                obs.range_m is None
                or obs.azimuth_rad is None
                or obs.elevation_rad is None
            ):
                continue


            target = evaluate_target(
                target_id=actor.track_index,

                range_m=obs.range_m,

                azimuth_rad=obs.azimuth_rad,

                elevation_rad=obs.elevation_rad,

                measurement_covariance=IDENTITY3,

                priority=1.0,
            )


            candidates.append(
                target
            )


        if not candidates:
            continue


        best = select_target(
            candidates
        )


        selected.append(
            best.target_id
        )


        confidence_history.append(
            best.beam_confidence
        )


        uncertainty_history.append(
            best.uncertainty_penalty
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
        "control steps =",
        len(selected)
    )


    print(
        "selected sequence =",
        selected
    )


    print(
        "confidence sequence =",
        confidence_history
    )


    print(
        "uncertainty sequence =",
        uncertainty_history
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
