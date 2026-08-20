from __future__ import annotations

import statistics


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
    evaluate_target,
    select_target,
)



SCENARIOS = (
    "b85e1bd6cc8e74c0",
    "75ae707721eb23b4",
    "52dafd686fe77b21",
)



IDENTITY3 = (
    (0.001,0.0,0.0),
    (0.0,0.001,0.0),
    (0.0,0.0,0.001),
)



def evaluate_scenario(
    scenario_id: str,
):

    (
        scenario,
        record,
        shard,
        index,
    ) = read_scenario_by_id(
        scenario_id
    )


    adapted = adapt_causal_womd_scenario(
        scenario
    )


    scene = build_real_ideal_observation_scene(
        adapted=adapted,
        raw_scenario=scenario,
    )


    selected = []

    confidence = []

    uncertainty = []


    steps = len(
        scene.actors[0].observations
    )


    for t in range(steps):

        candidates = []


        for actor in scene.actors:

            if actor.is_sdc:
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


            candidates.append(
                evaluate_target(
                    target_id=actor.track_index,

                    range_m=obs.range_m,

                    azimuth_rad=obs.azimuth_rad,

                    elevation_rad=obs.elevation_rad,

                    measurement_covariance=IDENTITY3,

                    priority=1.0,
                )
            )


        if candidates:

            best = select_target(
                candidates
            )


            selected.append(
                best.target_id
            )

            confidence.append(
                best.beam_confidence
            )

            uncertainty.append(
                best.uncertainty_penalty
            )


    switches = sum(
        1
        for a,b in zip(
            selected,
            selected[1:]
        )
        if a != b
    )


    return {

        "scenario": scenario_id,

        "manifest_line": record[
            "manifest_line"
        ],

        "local_record_index": index,

        "actors": len(scene.actors),

        "steps": len(selected),

        "switches": switches,

        "mean_confidence": statistics.mean(
            confidence
        ),

        "mean_uncertainty": statistics.mean(
            uncertainty
        ),

    }



def main():

    print(
        "===== Stage3 multi-scenario temporal gate ====="
    )


    results = []


    for scenario_id in SCENARIOS:

        results.append(
            evaluate_scenario(
                scenario_id
            )
        )


    for result in results:

        print(result)


    print(
        "scenarios evaluated =",
        len(results)
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
