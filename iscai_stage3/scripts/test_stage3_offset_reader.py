from iscai_stage3.io.womd_reader import (
    read_scenario_by_id,
)


SCENARIO_ID = "b85e1bd6cc8e74c0"


def main():

    (
        scenario,
        record,
        shard,
        local_record_index,
    ) = read_scenario_by_id(
        SCENARIO_ID
    )

    print(
        "===== Stage3 exact motion reader ====="
    )

    print(
        "requested scenario =",
        SCENARIO_ID,
    )

    print(
        "loaded scenario    =",
        scenario.scenario_id,
    )

    print(
        "manifest line      =",
        record["manifest_line"],
    )

    print(
        "source shard       =",
        record["source_shard"],
    )

    print(
        "compact shard      =",
        shard,
    )

    print(
        "local record index =",
        local_record_index,
    )

    print(
        "tracks             =",
        len(scenario.tracks),
    )

    print(
        "anchor             =",
        scenario.current_time_index,
    )

    if scenario.scenario_id != SCENARIO_ID:
        raise RuntimeError(
            "Exact scenario-id gate failed."
        )

    if scenario.current_time_index != 10:
        raise RuntimeError(
            "Unexpected WOMD anchor index."
        )

    if len(scenario.tracks) == 0:
        raise RuntimeError(
            "Motion scenario contains zero tracks."
        )

    print(
        "future states inspected = NO"
    )

    print(
        "STATUS = PASS"
    )


if __name__ == "__main__":
    main()
