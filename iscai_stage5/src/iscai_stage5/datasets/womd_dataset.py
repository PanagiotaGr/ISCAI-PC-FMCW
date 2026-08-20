from __future__ import annotations

from iscai_stage5.datasets.trajectory_dataset import (
    TrajectorySample,
)


PAST_LENGTH = 10
FUTURE_LENGTH = 10


def _xy(position):
    return (
        float(position[0]),
        float(position[1]),
    )


def build_samples_from_actor(
    history,
    *,
    anchor_index: int,
):
    """
    Build causal trajectory samples.

    Past:
        [t-9 ... t]

    Future:
        [t+1 ... t+10]

    Future is only target.
    """

    samples = []


    start = (
        PAST_LENGTH - 1
    )


    for t in range(
        start,
        anchor_index,
    ):

        past = []

        valid = True


        for i in range(
            t - PAST_LENGTH + 1,
            t + 1,
        ):

            if not history.state_valid[i]:
                valid = False
                break

            past.append(
                _xy(
                    history.position_W_m[i]
                )
            )


        if not valid:
            continue


        future = []


        for i in range(
            t + 1,
            t + FUTURE_LENGTH + 1,
        ):

            if i >= len(
                history.timestamps_s
            ):
                valid = False
                break


            if not history.state_valid[i]:
                valid = False
                break


            future.append(
                _xy(
                    history.position_W_m[i]
                )
            )


        if not valid:
            continue


        samples.append(
            TrajectorySample(
                past=tuple(past),
                future=tuple(future),
            )
        )


    return samples
