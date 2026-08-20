from pathlib import Path
from collections import defaultdict
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


# ============================================================
# Configuration
# ============================================================

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
    "real_stage6_predictions_rolling.npz"
)

REPORT = Path(
    "stage6_real/reports/"
    "real_stage6_predictions_rolling_summary.json"
)

CLASSES = (
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST",
)

MAX_FILES = 5

HISTORY = 10
FUTURE = 10

SEQUENCES_PER_CLASS = 10
ANCHORS_PER_SEQUENCE = 20

SEED = 42
BATCH_SIZE = 128

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Helpers
# ============================================================

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


def read_records(path):

    with path.open("rb") as f:

        record_index = 0

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

            payload = f.read(length)
            crc = f.read(4)

            if (
                len(payload) != length
                or
                len(crc) != 4
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

            yield (
                record_index,
                scenario,
            )

            record_index += 1


def sequence_is_valid(
    scenario,
    track_index,
):

    current = int(
        scenario.current_time_index
    )

    first_anchor = current

    last_anchor = (
        first_anchor
        +
        ANCHORS_PER_SEQUENCE
        -
        1
    )

    history_start = (
        first_anchor
        -
        HISTORY
        +
        1
    )

    final_future_end = (
        last_anchor
        +
        FUTURE
        +
        1
    )

    if history_start < 0:
        return False

    sdc_track = scenario.tracks[
        scenario.sdc_track_index
    ]

    track = scenario.tracks[
        track_index
    ]

    if (
        len(sdc_track.states)
        <
        final_future_end
    ):
        return False

    if (
        len(track.states)
        <
        final_future_end
    ):
        return False

    # Actor needs one continuous valid interval covering
    # history, all rolling anchors and future evaluation.
    states = track.states[
        history_start:
        final_future_end
    ]

    if not all(
        s.valid
        for s in states
    ):
        return False

    # Every anchor needs a valid SDC state because the
    # headlamp frame is rebuilt at every controller frame.
    for anchor in range(
        first_anchor,
        last_anchor + 1,
    ):

        sdc = sdc_track.states[
            anchor
        ]

        actor = track.states[
            anchor
        ]

        if (
            not sdc.valid
            or
            sdc.length <= 0
        ):
            return False

        if (
            actor.length <= 0
            or
            actor.width <= 0
            or
            actor.height <= 0
        ):
            return False

    return True


# ============================================================
# Pass 1: candidate rolling sequences
# ============================================================

def collect_candidates(files):

    pools = {
        cls: []
        for cls in CLASSES
    }

    for file_index, path in enumerate(
        files
    ):

        print(
            f"candidate scan "
            f"{file_index + 1}/{len(files)}: "
            f"{path.name}",
            flush=True,
        )

        for (
            record_index,
            scenario,
        ) in read_records(path):

            for track_index, track in enumerate(
                scenario.tracks
            ):

                cls = track_type_name(
                    track
                )

                if cls not in pools:
                    continue

                if not sequence_is_valid(
                    scenario,
                    track_index,
                ):
                    continue

                pools[cls].append(
                    (
                        file_index,
                        record_index,
                        int(track_index),
                        str(
                            scenario.scenario_id
                        ),
                    )
                )

    return pools


# ============================================================
# Build one rolling controller-frame sample
# ============================================================

def build_sample(
    scenario,
    track_index,
    cls,
    anchor,
    sequence_index,
    anchor_position,
):

    track = scenario.tracks[
        track_index
    ]

    sdc_track = scenario.tracks[
        scenario.sdc_track_index
    ]

    sdc = sdc_track.states[
        anchor
    ]

    actor = track.states[
        anchor
    ]

    history_start = (
        anchor
        -
        HISTORY
        +
        1
    )

    past = list(
        track.states[
            history_start:
            anchor + 1
        ]
    )

    if len(past) != HISTORY:
        raise RuntimeError(
            "Invalid rolling history length"
        )

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

    x = []

    for state in past:

        p_E = (
            frames.T_E0_from_W
            .apply_point(
                (
                    state.center_x,
                    state.center_y,
                    0.0,
                )
            )
        )

        # Preserve exactly the same model-input convention
        # used by the canonical balanced Stage-6 builder.
        x.append(
            [
                p_E[0],
                p_E[1],
                state.velocity_x,
                state.velocity_y,
                math.sin(
                    state.heading
                ),
                math.cos(
                    state.heading
                ),
                state.length,
                state.width,
            ]
        )

    current_center_H = (
        frames.T_H0_from_W
        .apply_point(
            (
                actor.center_x,
                actor.center_y,
                actor.center_z,
            )
        )
    )

    heading_H = wrap_angle(
        actor.heading
        -
        sdc.heading
    )

    return {
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
            int(track_index),

        "sequence_index":
            int(sequence_index),

        "anchor_position":
            int(anchor_position),

        "anchor_time_index":
            int(anchor),

        "actor_class":
            cls,

        "length_m":
            float(actor.length),

        "width_m":
            float(actor.width),

        "height_m":
            float(actor.height),

        "current_center_H_m":
            np.asarray(
                current_center_H,
                dtype=np.float32,
            ),

        "current_heading_H_rad":
            float(heading_H),

        "actor_heading_W":
            float(actor.heading),

        "sdc_center_W_m":
            np.asarray(
                [
                    sdc.center_x,
                    sdc.center_y,
                    0.0,
                ],
                dtype=np.float32,
            ),

        "sdc_heading_W":
            float(sdc.heading),

        "sdc_length_m":
            float(sdc.length),

        "center_z_H":
            float(
                current_center_H[2]
            ),

        "headlamp_yaw":
            float(
                config.yaw_rad
            ),

        "frames":
            frames,
    }


# ============================================================
# Main
# ============================================================

def main():

    t0 = time.perf_counter()

    rng = np.random.default_rng(
        SEED
    )

    files = list(
        ROOT.glob(
            "*tfrecord*"
        )
    )[:MAX_FILES]

    if not files:
        raise RuntimeError(
            f"No TFRecords found in {ROOT}"
        )

    print("=" * 78)
    print(
        "STAGE 6 ROLLING CAUSAL PREDICTION BUILDER"
    )
    print("=" * 78)

    print(
        "files =",
        len(files)
    )

    print(
        "sequences/class =",
        SEQUENCES_PER_CLASS
    )

    print(
        "anchors/sequence =",
        ANCHORS_PER_SEQUENCE
    )

    print(
        "controller dt = 0.1 s"
    )

    print(
        "device =",
        DEVICE
    )

    # --------------------------------------------------------
    # Candidate discovery
    # --------------------------------------------------------

    pools = collect_candidates(
        files
    )

    selected = []

    sequence_counter = 0

    print()
    print("CANDIDATE SEQUENCES")

    for cls in CLASSES:

        candidates = pools[
            cls
        ]

        print(
            cls,
            "=",
            len(candidates),
        )

        if (
            len(candidates)
            <
            SEQUENCES_PER_CLASS
        ):
            raise RuntimeError(
                f"Not enough {cls} sequences"
            )

        order = rng.permutation(
            len(candidates)
        )

        chosen = [
            candidates[
                int(i)
            ]
            for i in order[
                :SEQUENCES_PER_CLASS
            ]
        ]

        for candidate in chosen:

            selected.append(
                (
                    sequence_counter,
                    cls,
                    candidate,
                )
            )

            sequence_counter += 1

    # Group selections by file/record for second pass.
    lookup = defaultdict(
        list
    )

    for (
        sequence_index,
        cls,
        candidate,
    ) in selected:

        (
            file_index,
            record_index,
            track_index,
            scenario_id,
        ) = candidate

        lookup[
            (
                file_index,
                record_index,
            )
        ].append(
            (
                sequence_index,
                cls,
                track_index,
                scenario_id,
            )
        )

    # --------------------------------------------------------
    # Generate rolling samples
    # --------------------------------------------------------

    samples = []

    for file_index, path in enumerate(
        files
    ):

        for (
            record_index,
            scenario,
        ) in read_records(path):

            key = (
                file_index,
                record_index,
            )

            if key not in lookup:
                continue

            current = int(
                scenario.current_time_index
            )

            for (
                sequence_index,
                cls,
                track_index,
                expected_scenario_id,
            ) in lookup[key]:

                if (
                    str(
                        scenario.scenario_id
                    )
                    !=
                    expected_scenario_id
                ):
                    raise RuntimeError(
                        "Scenario identity mismatch"
                    )

                for anchor_position in range(
                    ANCHORS_PER_SEQUENCE
                ):

                    anchor = (
                        current
                        +
                        anchor_position
                    )

                    samples.append(
                        build_sample(
                            scenario,
                            track_index,
                            cls,
                            anchor,
                            sequence_index,
                            anchor_position,
                        )
                    )

    expected_samples = (
        len(CLASSES)
        *
        SEQUENCES_PER_CLASS
        *
        ANCHORS_PER_SEQUENCE
    )

    if (
        len(samples)
        !=
        expected_samples
    ):
        raise RuntimeError(
            f"Expected {expected_samples} samples, "
            f"got {len(samples)}"
        )

    samples.sort(
        key=lambda x: (
            x["sequence_index"],
            x["anchor_position"],
        )
    )

    print()
    print(
        "rolling controller frames =",
        len(samples)
    )

    # --------------------------------------------------------
    # Gaussian inference
    # --------------------------------------------------------

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
    sequence_indices = []
    anchor_positions = []
    anchor_time_indices = []

    classes = []

    lengths = []
    widths = []
    heights = []

    sdc_centers_W = []
    sdc_headings_W = []
    sdc_lengths = []

    current_centers_H = []
    current_headings_H = []

    means_H = []
    stds_H = []
    rhos_H = []

    with torch.no_grad():

        for start in range(
            0,
            len(samples),
            BATCH_SIZE,
        ):

            items = samples[
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

                frames = meta[
                    "frames"
                ]

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

                    rho_H[t] = rho

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

                sequence_indices.append(
                    meta[
                        "sequence_index"
                    ]
                )

                anchor_positions.append(
                    meta[
                        "anchor_position"
                    ]
                )

                anchor_time_indices.append(
                    meta[
                        "anchor_time_index"
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

                sdc_centers_W.append(
                    meta[
                        "sdc_center_W_m"
                    ]
                )

                sdc_headings_W.append(
                    meta[
                        "sdc_heading_W"
                    ]
                )

                sdc_lengths.append(
                    meta[
                        "sdc_length_m"
                    ]
                )

                current_centers_H.append(
                    meta[
                        "current_center_H_m"
                    ]
                )

                current_headings_H.append(
                    meta[
                        "current_heading_H_rad"
                    ]
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
                f"{min(start+BATCH_SIZE,len(samples))}"
                f"/{len(samples)}",
                flush=True,
            )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    scenario_ids = np.asarray(
        scenario_ids,
        dtype="U64",
    )

    track_indices = np.asarray(
        track_indices,
        dtype=np.int32,
    )

    sequence_indices = np.asarray(
        sequence_indices,
        dtype=np.int32,
    )

    anchor_positions = np.asarray(
        anchor_positions,
        dtype=np.int32,
    )

    anchor_time_indices = np.asarray(
        anchor_time_indices,
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

    sdc_centers_W = np.asarray(
        sdc_centers_W,
        dtype=np.float32,
    )

    sdc_headings_W = np.asarray(
        sdc_headings_W,
        dtype=np.float32,
    )

    sdc_lengths = np.asarray(
        sdc_lengths,
        dtype=np.float32,
    )

    current_centers_H = np.asarray(
        current_centers_H,
        dtype=np.float32,
    )

    current_headings_H = np.asarray(
        current_headings_H,
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

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert (
        means_H.shape
        ==
        (
            expected_samples,
            FUTURE,
            3,
        )
    )

    assert (
        stds_H.shape
        ==
        (
            expected_samples,
            FUTURE,
            2,
        )
    )

    assert (
        rhos_H.shape
        ==
        (
            expected_samples,
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

    # Every sequence must consist of exactly consecutive
    # controller anchors.
    for seq in np.unique(
        sequence_indices
    ):

        mask = (
            sequence_indices == seq
        )

        a = anchor_time_indices[
            mask
        ]

        p = anchor_positions[
            mask
        ]

        order = np.argsort(
            p
        )

        a = a[order]

        assert len(a) == (
            ANCHORS_PER_SEQUENCE
        )

        assert np.all(
            np.diff(a) == 1
        )

    # Class balance by sequence.
    sequence_classes = {}

    for seq in np.unique(
        sequence_indices
    ):

        cls_values = np.unique(
            classes[
                sequence_indices == seq
            ]
        )

        assert len(cls_values) == 1

        sequence_classes[
            str(cls_values[0])
        ] = (
            sequence_classes.get(
                str(cls_values[0]),
                0,
            )
            +
            1
        )

    for cls in CLASSES:

        assert (
            sequence_classes.get(
                cls,
                0,
            )
            ==
            SEQUENCES_PER_CLASS
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

        sequence_index=
            sequence_indices,

        anchor_position=
            anchor_positions,

        anchor_time_index=
            anchor_time_indices,

        actor_class=
            classes,

        length_m=
            lengths,

        width_m=
            widths,

        height_m=
            heights,

        sdc_center_W_m=
            sdc_centers_W,

        sdc_heading_W_rad=
            sdc_headings_W,

        sdc_length_m=
            sdc_lengths,

        current_center_H_m=
            current_centers_H,

        current_heading_H_rad=
            current_headings_H,

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

        "checkpoint":
            str(
                CHECKPOINT
            ),

        "causal_prediction":
            True,

        "future_used_as_input":
            False,

        "controller_frame_interval_s":
            0.1,

        "history_steps":
            HISTORY,

        "future_steps":
            FUTURE,

        "sequences_per_class":
            SEQUENCES_PER_CLASS,

        "anchors_per_sequence":
            ANCHORS_PER_SEQUENCE,

        "total_sequences":
            int(
                len(
                    np.unique(
                        sequence_indices
                    )
                )
            ),

        "total_controller_frames":
            int(
                len(classes)
            ),

        "sequence_counts":
            sequence_classes,

        "semantics":
            (
                "Each sequence contains consecutive WOMD anchor "
                "times for the same scenario/track pair. The "
                "headlamp frame and causal predictor history are "
                "rebuilt at every 100 ms anchor. Future states "
                "are not used as predictor inputs."
            ),

        "runtime_s":
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
        "ROLLING CAUSAL PREDICTION BUILD COMPLETE"
    )
    print("=" * 78)

    print(
        "sequences =",
        report[
            "total_sequences"
        ]
    )

    print(
        "controller frames =",
        report[
            "total_controller_frames"
        ]
    )

    print(
        "class sequence counts =",
        sequence_classes
    )

    print(
        "runtime =",
        elapsed,
        "sec"
    )

    print()
    print(
        "Saved:",
        OUTPUT
    )

    print(
        "Saved:",
        REPORT
    )


if __name__ == "__main__":
    main()
