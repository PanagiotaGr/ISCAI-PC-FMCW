from pathlib import Path
import json
import struct
import time

import numpy as np

from waymo_open_dataset.protos import scenario_pb2

from iscai_stage1.contracts.stage1a import (
    HeadlampSurrogateConfig,
)
from iscai_stage1.geometry.frames import (
    SdcStateW,
    build_anchor_frames,
)


SOURCE = Path(
    "stage6_real/data/"
    "real_stage6_predictions_balanced.npz"
)

ROOT = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)

OUTPUT = Path(
    "stage6_real/data/"
    "real_stage6_oracle_future_boxes.npz"
)

REPORT = Path(
    "stage6_real/reports/"
    "real_stage6_oracle_future_boxes_summary.json"
)

MAX_FILES = 5
FUTURE = 10


def main():

    print("=" * 78)
    print(
        "STAGE 6 REAL ORACLE FUTURE BOXES"
    )
    print("=" * 78)

    source = np.load(
        SOURCE
    )

    wanted_scenario = source[
        "scenario_id"
    ]

    wanted_track = source[
        "track_index"
    ]

    wanted_class = source[
        "actor_class"
    ]

    n = len(
        wanted_track
    )

    wanted = {
        (
            str(
                wanted_scenario[i]
            ),
            int(
                wanted_track[i]
            ),
        ): i
        for i in range(n)
    }

    print(
        "wanted actors =",
        n
    )

    current_center_H = np.zeros(
        (
            n,
            3,
        ),
        dtype=np.float32,
    )

    current_heading_H = np.zeros(
        n,
        dtype=np.float32,
    )

    current_lwh = np.zeros(
        (
            n,
            3,
        ),
        dtype=np.float32,
    )

    future_center_H = np.zeros(
        (
            n,
            FUTURE,
            3,
        ),
        dtype=np.float32,
    )

    future_heading_H = np.zeros(
        (
            n,
            FUTURE,
        ),
        dtype=np.float32,
    )

    future_lwh = np.zeros(
        (
            n,
            FUTURE,
            3,
        ),
        dtype=np.float32,
    )

    found = np.zeros(
        n,
        dtype=bool,
    )

    files = list(
        ROOT.glob(
            "*tfrecord*"
        )
    )[:MAX_FILES]

    t0 = time.perf_counter()

    for file_index, path in enumerate(
        files,
        start=1,
    ):

        print(
            f"reading file "
            f"{file_index}/{len(files)}: "
            f"{path.name}",
            flush=True,
        )

        with path.open("rb") as f:

            while True:

                header = f.read(12)

                if not header:
                    break

                if len(header) != 12:
                    raise RuntimeError(
                        "Truncated TFRecord header"
                    )

                length = struct.unpack(
                    "<Q",
                    header[:8],
                )[0]

                payload = f.read(
                    length
                )

                crc = f.read(4)

                if (
                    len(payload) != length
                    or len(crc) != 4
                ):
                    raise RuntimeError(
                        "Truncated TFRecord record"
                    )

                scenario = (
                    scenario_pb2.Scenario()
                )

                scenario.ParseFromString(
                    payload
                )

                sid = str(
                    scenario.scenario_id
                )

                current = int(
                    scenario.current_time_index
                )

                if (
                    scenario.sdc_track_index
                    >= len(
                        scenario.tracks
                    )
                ):
                    continue

                sdc_track = (
                    scenario.tracks[
                        scenario.sdc_track_index
                    ]
                )

                if (
                    len(sdc_track.states)
                    <= current
                ):
                    continue

                sdc = (
                    sdc_track.states[
                        current
                    ]
                )

                if (
                    not sdc.valid
                    or
                    sdc.length <= 0
                ):
                    continue

                frames = build_anchor_frames(
                    SdcStateW(
                        center_w_m=(
                            sdc.center_x,
                            sdc.center_y,
                            0.0,
                        ),
                        heading_rad=
                            sdc.heading,
                        length_m=
                            sdc.length,
                    ),
                    HeadlampSurrogateConfig(),
                )

                for track_index, track in enumerate(
                    scenario.tracks
                ):

                    key = (
                        sid,
                        int(
                            track_index
                        ),
                    )

                    if key not in wanted:
                        continue

                    out_i = wanted[
                        key
                    ]

                    if found[
                        out_i
                    ]:
                        continue

                    if (
                        len(track.states)
                        <
                        current
                        +
                        FUTURE
                        +
                        1
                    ):
                        continue

                    current_state = (
                        track.states[
                            current
                        ]
                    )

                    future_states = list(
                        track.states[
                            current + 1:
                            current + 1 + FUTURE
                        ]
                    )

                    if (
                        not current_state.valid
                        or
                        len(future_states)
                        != FUTURE
                        or
                        not all(
                            s.valid
                            for s in future_states
                        )
                    ):
                        continue

                    p_current = (
                        frames.T_H0_from_W
                        .apply_point(
                            (
                                current_state.center_x,
                                current_state.center_y,
                                current_state.center_z,
                            )
                        )
                    )

                    current_center_H[
                        out_i
                    ] = p_current

                    current_heading_H[
                        out_i
                    ] = (
                        current_state.heading
                        -
                        sdc.heading
                    )

                    current_lwh[
                        out_i
                    ] = [
                        current_state.length,
                        current_state.width,
                        current_state.height,
                    ]

                    for t, state in enumerate(
                        future_states
                    ):

                        p_H = (
                            frames.T_H0_from_W
                            .apply_point(
                                (
                                    state.center_x,
                                    state.center_y,
                                    state.center_z,
                                )
                            )
                        )

                        future_center_H[
                            out_i,
                            t,
                        ] = p_H

                        future_heading_H[
                            out_i,
                            t,
                        ] = (
                            state.heading
                            -
                            sdc.heading
                        )

                        future_lwh[
                            out_i,
                            t,
                        ] = [
                            state.length,
                            state.width,
                            state.height,
                        ]

                    found[
                        out_i
                    ] = True

        print(
            "found =",
            int(
                found.sum()
            ),
            "/",
            n,
            flush=True,
        )

    if not found.all():

        missing = np.where(
            ~found
        )[0]

        print(
            "missing indices =",
            missing[:20].tolist()
        )

        raise RuntimeError(
            f"Missing "
            f"{len(missing)} "
            f"balanced actors."
        )

    if not np.isfinite(
        future_center_H
    ).all():
        raise RuntimeError(
            "Non-finite GT centers."
        )

    if np.any(
        future_lwh <= 0
    ):
        raise RuntimeError(
            "Invalid GT dimensions."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT,

        scenario_id=
            wanted_scenario,

        track_index=
            wanted_track,

        actor_class=
            wanted_class,

        current_center_H_m=
            current_center_H,

        current_heading_H_rad=
            current_heading_H,

        current_lwh_m=
            current_lwh,

        future_center_H_m=
            future_center_H,

        future_heading_H_rad=
            future_heading_H,

        future_lwh_m=
            future_lwh,
    )

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    report = {
        "status":
            "PASS",

        "data_source":
            "real_WOMD_validation_ground_truth",

        "actors":
            int(n),

        "future_steps":
            FUTURE,

        "coordinate_frame":
            "H0_anchor_headlamp_frame",

        "ground_truth_role":
            (
                "Evaluation-only oracle future "
                "box geometry. Never used as "
                "predictor/controller input."
            ),

        "future_used_as_input":
            False,

        "found_all":
            bool(
                found.all()
            ),

        "output":
            str(
                OUTPUT
            ),

        "generation_time_seconds":
            float(
                elapsed
            ),
    }

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("=" * 78)
    print(
        "ORACLE FUTURE BOXES COMPLETE"
    )
    print("=" * 78)

    print(
        "actors =",
        n
    )

    print(
        "future shape =",
        future_center_H.shape
    )

    print(
        "all found =",
        bool(
            found.all()
        )
    )

    print(
        "time =",
        elapsed,
        "sec"
    )

    print()
    print(
        "Saved:",
        OUTPUT
    )

    print(
        "Report:",
        REPORT
    )


if __name__ == "__main__":
    main()
