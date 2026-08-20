from __future__ import annotations

import json
from pathlib import Path

from iscai_stage3.io.womd_reader import (
    read_scenario_by_id,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)


SCENARIOS = [
    "b85e1bd6cc8e74c0",
    "75ae707721eb23b4",
    "52dafd686fe77b21",
]


PAST = 10
FUTURE = 10


OUTPUT = Path(
    "exports/stage5/womd_trajectories.json"
)


def xy(state):

    return [
        float(state.center_x),
        float(state.center_y),
    ]


def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = []


    for scenario_id in SCENARIOS:

        scenario, _, _, _ = (
            read_scenario_by_id(
                scenario_id
            )
        )


        adapted = (
            adapt_causal_womd_scenario(
                scenario
            )
        )


        anchor = adapted.anchor_index


        for actor in scenario.tracks:

            states = actor.states


            if len(states) <= (
                anchor + FUTURE
            ):
                continue


            past = []
            future = []


            valid = True


            for i in range(
                anchor - PAST + 1,
                anchor + 1,
            ):

                state = states[i]

                if not state.valid:
                    valid = False
                    break

                past.append(
                    xy(state)
                )


            if not valid:
                continue


            for i in range(
                anchor + 1,
                anchor + FUTURE + 1,
            ):

                state = states[i]

                if not state.valid:
                    valid = False
                    break

                future.append(
                    xy(state)
                )


            if not valid:
                continue


            samples.append(
                {
                    "past": past,
                    "future": future,
                    "scenario_id": scenario_id,
                    "track_index": actor.id,
                }
            )


        print(
            scenario_id,
            "total samples =",
            len(samples),
        )


    OUTPUT.write_text(
        json.dumps(
            samples
        )
    )


    print()
    print(
        "Exported samples =",
        len(samples)
    )

    print(
        "written =",
        OUTPUT
    )


if __name__ == "__main__":
    main()
