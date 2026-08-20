from pathlib import Path
import json
import math
import struct
import time

import numpy as np
import torch

from waymo_open_dataset.protos import scenario_pb2

from iscai_stage1.contracts.stage1a import (
    HeadlampSurrogateConfig,
)
from iscai_stage1.geometry.frames import (
    SdcStateW,
    build_anchor_frames,
)
from iscai_stage4.training.gru_model import (
    GaussianGRUPredictor,
)


ROOT = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)

CHECKPOINT = Path(
    "/home/agni/waymo/iscai_stage4/"
    "checkpoints/stage4_auto_cuda/"
    "ALL_gaussian_h128_l1_d0.0_lr0.001_b32.pt"
)

OUTPUT = Path(
    "stage6_real/data/"
    "real_stage6_predictions_balanced.npz"
)

REPORT = Path(
    "stage6_real/reports/"
    "real_stage6_predictions_balanced_summary.json"
)

CLASSES = (
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST",
)

MAX_FILES = 5
PER_CLASS = 247

HISTORY = 10
FUTURE = 10

SEED = 42
BATCH_SIZE = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def track_type_name(track):

    enum_desc = (
        track.DESCRIPTOR
        .fields_by_name["object_type"]
        .enum_type
    )

    enum_value = (
        enum_desc
        .values_by_number
        .get(track.object_type)
    )

    if enum_value is None:
        return None

    return (
        enum_value.name
        .replace("TYPE_", "")
        .upper()
    )


def wrap_angle(x):
    return (
        x + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def rotate_covariance(
    sx,
    sy,
    rho,
    yaw,
):

    cov = np.asarray(
        [
            [
                sx * sx,
                rho * sx * sy,
            ],
            [
                rho * sx * sy,
                sy * sy,
            ],
        ],
        dtype=np.float64,
    )

    c = math.cos(yaw)
    s = math.sin(yaw)

    R = np.asarray(
        [
            [c, -s],
            [s,  c],
        ],
        dtype=np.float64,
    )

    cov_h = (
        R
        @ cov
        @ R.T
    )

    sx_h = math.sqrt(
        max(
            cov_h[0, 0],
            1e-12,
        )
    )

    sy_h = math.sqrt(
        max(
            cov_h[1, 1],
            1e-12,
        )
    )

    rho_h = (
        cov_h[0, 1]
        /
        (sx_h * sy_h)
    )

    return (
        sx_h,
        sy_h,
        float(
            np.clip(
                rho_h,
                -0.999,
                0.999,
            )
        ),
    )


def collect_all_valid():

    pools = {
        cls: []
        for cls in CLASSES
    }

    files = list(
        ROOT.glob(
            "*tfrecord*"
        )
    )[:MAX_FILES]

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

                current = int(
                    scenario.current_time_index
                )

                history_start = (
                    current
                    -
                    HISTORY
                    +
                    1
                )

                future_start = (
                    current + 1
                )

                future_end = (
                    future_start
                    +
                    FUTURE
                )

                if history_start < 0:
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

                config = (
                    HeadlampSurrogateConfig()
                )

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
                    config,
                )

                for track_index, track in enumerate(
                    scenario.tracks
                ):

                    cls = track_type_name(
                        track
                    )

                    if cls not in pools:
                        continue

                    if (
                        len(track.states)
                        <
                        future_end
                    ):
                        continue

                    past = list(
                        track.states[
                            history_start:
                            current + 1
                        ]
                    )

                    future = list(
                        track.states[
                            future_start:
                            future_end
                        ]
                    )

                    if (
                        len(past) != HISTORY
                        or
                        len(future) != FUTURE
                    ):
                        continue

                    if not all(
                        s.valid
                        for s in past
                    ):
                        continue

                    if not all(
                        s.valid
                        for s in future
                    ):
                        continue

                    anchor_actor = (
                        past[-1]
                    )

                    if (
                        anchor_actor.length <= 0
                        or
                        anchor_actor.width <= 0
                        or
                        anchor_actor.height <= 0
                    ):
                        continue

                    x = []

                    for s in past:

                        p_E = (
                            frames.T_E0_from_W
                            .apply_point(
                                (
                                    s.center_x,
                                    s.center_y,
                                    0.0,
                                )
                            )
                        )

                        x.append(
                            [
                                p_E[0],
                                p_E[1],
                                s.velocity_x,
                                s.velocity_y,
                                math.sin(
                                    s.heading
                                ),
                                math.cos(
                                    s.heading
                                ),
                                s.length,
                                s.width,
                            ]
                        )

                    center_H = (
                        frames.T_H0_from_W
                        .apply_point(
                            (
                                anchor_actor.center_x,
                                anchor_actor.center_y,
                                anchor_actor.center_z,
                            )
                        )
                    )

                    pools[cls].append(
                        {
                            "x":
                                np.asarray(
                                    x,
                                    dtype=np.float32,
                                ),

                            "scenario_id":
                                str(
                                    scenario.scenario_id
                                ),

                            "track_index":
                                int(
                                    track_index
                                ),

                            "actor_class":
                                cls,

                            "length_m":
                                float(
                                    anchor_actor.length
                                ),

                            "width_m":
                                float(
                                    anchor_actor.width
                                ),

                            "height_m":
                                float(
                                    anchor_actor.height
                                ),

                            "actor_heading_W":
                                float(
                                    anchor_actor.heading
                                ),

                            "sdc_heading_W":
                                float(
                                    sdc.heading
                                ),

                            "center_z_H":
                                float(
                                    center_H[2]
                                ),

                            "frames":
                                frames,

                            "headlamp_yaw":
                                float(
                                    config.yaw_rad
                                ),
                        }
                    )

    return pools


def main():

    print("=" * 78)
    print(
        "STAGE 6 REAL CLASS-BALANCED "
        "PROBABILISTIC SOURCE"
    )
    print("=" * 78)

    print(
        "DEVICE:",
        DEVICE
    )

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Target per class:",
        PER_CLASS
    )

    rng = np.random.default_rng(
        SEED
    )

    t0 = time.perf_counter()

    pools = collect_all_valid()

    available = {
        cls: len(pools[cls])
        for cls in CLASSES
    }

    print()
    print(
        "Available:",
        available
    )

    for cls in CLASSES:

        if (
            len(pools[cls])
            <
            PER_CLASS
        ):
            raise RuntimeError(
                f"Not enough {cls}: "
                f"{len(pools[cls])}"
            )

    balanced = []

    selected_source_indices = {}

    for cls in CLASSES:

        idx = rng.choice(
            len(pools[cls]),
            size=PER_CLASS,
            replace=False,
        )

        idx = np.sort(
            idx
        )

        selected_source_indices[
            cls
        ] = idx.tolist()

        balanced.extend(
            pools[cls][int(i)]
            for i in idx
        )

    # Deterministically shuffle the final balanced set,
    # while preserving equal class counts.
    order = rng.permutation(
        len(balanced)
    )

    balanced = [
        balanced[int(i)]
        for i in order
    ]

    print(
        "Balanced actors:",
        len(balanced)
    )


    # ========================================================
    # Gaussian inference
    # ========================================================

    model = GaussianGRUPredictor(
        input_size=8,
        hidden=128,
        layers=1,
        dropout=0.0,
        future=FUTURE,
    ).to(
        DEVICE
    )

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.eval()


    scenario_ids = []
    track_indices = []
    classes = []

    lengths = []
    widths = []
    heights = []

    headings_H = []

    means_H = []
    stds_H = []
    rhos_H = []


    with torch.no_grad():

        for start in range(
            0,
            len(balanced),
            BATCH_SIZE,
        ):

            items = balanced[
                start:
                start + BATCH_SIZE
            ]

            x = np.stack(
                [
                    item["x"]
                    for item in items
                ],
                axis=0,
            )

            out = model(
                torch.from_numpy(
                    x
                ).to(
                    DEVICE
                )
            )

            mu_E = (
                out["mu"]
                .cpu()
                .numpy()
            )

            std_E = (
                out["std"]
                .cpu()
                .numpy()
            )

            rho_E = (
                out["rho"][
                    ...,
                    0
                ]
                .cpu()
                .numpy()
            )

            for i, meta in enumerate(
                items
            ):

                frames = (
                    meta["frames"]
                )

                heading_H = wrap_angle(
                    meta[
                        "actor_heading_W"
                    ]
                    -
                    meta[
                        "sdc_heading_W"
                    ]
                )

                mean_H = np.zeros(
                    (
                        FUTURE,
                        3,
                    ),
                    dtype=np.float32,
                )

                std_H = np.zeros(
                    (
                        FUTURE,
                        2,
                    ),
                    dtype=np.float32,
                )

                rho_H = np.zeros(
                    FUTURE,
                    dtype=np.float32,
                )

                for t in range(
                    FUTURE
                ):

                    p_W = (
                        frames.T_W_from_E0
                        .apply_point(
                            (
                                float(
                                    mu_E[
                                        i,
                                        t,
                                        0
                                    ]
                                ),
                                float(
                                    mu_E[
                                        i,
                                        t,
                                        1
                                    ]
                                ),
                                0.0,
                            )
                        )
                    )

                    p_H = (
                        frames.T_H0_from_W
                        .apply_point(
                            p_W
                        )
                    )

                    mean_H[t] = [
                        p_H[0],
                        p_H[1],
                        meta[
                            "center_z_H"
                        ],
                    ]

                    (
                        sx,
                        sy,
                        rho,
                    ) = rotate_covariance(
                        float(
                            std_E[
                                i,
                                t,
                                0
                            ]
                        ),
                        float(
                            std_E[
                                i,
                                t,
                                1
                            ]
                        ),
                        float(
                            rho_E[
                                i,
                                t
                            ]
                        ),
                        -meta[
                            "headlamp_yaw"
                        ],
                    )

                    std_H[t] = [
                        sx,
                        sy,
                    ]

                    rho_H[t] = (
                        rho
                    )

                scenario_ids.append(
                    meta[
                        "scenario_id"
                    ]
                )

                track_indices.append(
                    meta[
                        "track_index"
                    ]
                )

                classes.append(
                    meta[
                        "actor_class"
                    ]
                )

                lengths.append(
                    meta[
                        "length_m"
                    ]
                )

                widths.append(
                    meta[
                        "width_m"
                    ]
                )

                heights.append(
                    meta[
                        "height_m"
                    ]
                )

                headings_H.append(
                    heading_H
                )

                means_H.append(
                    mean_H
                )

                stds_H.append(
                    std_H
                )

                rhos_H.append(
                    rho_H
                )

            print(
                f"inference="
                f"{min(start+BATCH_SIZE,len(balanced))}"
                f"/{len(balanced)}",
                flush=True,
            )


    scenario_ids = np.asarray(
        scenario_ids,
        dtype="U64",
    )

    track_indices = np.asarray(
        track_indices,
        dtype=np.int32,
    )

    classes = np.asarray(
        classes,
        dtype="U16",
    )

    lengths = np.asarray(
        lengths,
        dtype=np.float32,
    )

    widths = np.asarray(
        widths,
        dtype=np.float32,
    )

    heights = np.asarray(
        heights,
        dtype=np.float32,
    )

    headings_H = np.asarray(
        headings_H,
        dtype=np.float32,
    )

    means_H = np.asarray(
        means_H,
        dtype=np.float32,
    )

    stds_H = np.asarray(
        stds_H,
        dtype=np.float32,
    )

    rhos_H = np.asarray(
        rhos_H,
        dtype=np.float32,
    )


    # ========================================================
    # Validation
    # ========================================================

    expected_total = (
        PER_CLASS
        *
        len(CLASSES)
    )

    assert len(classes) == expected_total

    final_counts = {
        cls:
            int(
                np.sum(
                    classes == cls
                )
            )
        for cls in CLASSES
    }

    for cls in CLASSES:
        assert (
            final_counts[cls]
            ==
            PER_CLASS
        )

    assert (
        means_H.shape
        ==
        (
            expected_total,
            FUTURE,
            3,
        )
    )

    assert (
        stds_H.shape
        ==
        (
            expected_total,
            FUTURE,
            2,
        )
    )

    assert (
        rhos_H.shape
        ==
        (
            expected_total,
            FUTURE,
        )
    )

    assert np.isfinite(
        means_H
    ).all()

    assert np.isfinite(
        stds_H
    ).all()

    assert np.all(
        stds_H > 0
    )


    # ========================================================
    # Save
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT,

        scenario_id=
            scenario_ids,

        track_index=
            track_indices,

        actor_class=
            classes,

        length_m=
            lengths,

        width_m=
            widths,

        height_m=
            heights,

        current_heading_H_rad=
            headings_H,

        future_mean_H_m=
            means_H,

        future_std_xy_H_m=
            stds_H,

        future_rho_xy_H=
            rhos_H,
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
            "real_WOMD_validation",

        "sampling":
            "class_balanced",

        "seed":
            SEED,

        "max_files":
            MAX_FILES,

        "available_valid_samples":
            available,

        "selected_per_class":
            PER_CLASS,

        "actors":
            expected_total,

        "class_counts":
            final_counts,

        "future_steps":
            FUTURE,

        "coordinate_frame":
            "H0_anchor_headlamp_frame",

        "prediction":
            "Stage4 trained Gaussian GRU",

        "checkpoint":
            str(
                CHECKPOINT
            ),

        "future_used_as_input":
            False,

        "geometry_assumptions": {
            "future_xy":
                "Gaussian predicted",

            "box_dimensions":
                (
                    "causal current WOMD "
                    "length/width/height"
                ),

            "future_heading":
                (
                    "causal current relative "
                    "heading held constant"
                ),

            "future_z":
                (
                    "causal current H0 "
                    "vertical center held constant"
                ),
        },

        "mean_std_x_m":
            float(
                stds_H[
                    :,
                    :,
                    0
                ].mean()
            ),

        "mean_std_y_m":
            float(
                stds_H[
                    :,
                    :,
                    1
                ].mean()
            ),

        "generation_time_seconds":
            float(
                elapsed
            ),

        "output":
            str(
                OUTPUT
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
        "BALANCED STAGE 6 SOURCE COMPLETE"
    )
    print("=" * 78)

    print(
        "available =",
        available
    )

    print(
        "selected =",
        final_counts
    )

    print(
        "actors =",
        expected_total
    )

    print(
        "mean shape =",
        means_H.shape
    )

    print(
        "mean std x =",
        report[
            "mean_std_x_m"
        ]
    )

    print(
        "mean std y =",
        report[
            "mean_std_y_m"
        ]
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
