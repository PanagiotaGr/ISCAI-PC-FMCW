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


# ============================================================
# CONFIG
# ============================================================

VAL_ROOT = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)

CHECKPOINT = Path(
    "/home/agni/waymo/iscai_stage4/"
    "checkpoints/stage4_auto_cuda/"
    "ALL_gaussian_h128_l1_d0.0_lr0.001_b32.pt"
)

OUTPUT_NPZ = Path(
    "stage6_real/data/"
    "real_stage6_predictions.npz"
)

OUTPUT_JSON = Path(
    "stage6_real/reports/"
    "real_stage6_predictions_summary.json"
)

MAX_FILES = 5
MAX_ACTORS = 5000

HISTORY = 10
FUTURE = 10

ALLOWED_CLASSES = {
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST",
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 42


# ============================================================
# HELPERS
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


def wrap_angle(angle):

    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def rotate_covariance_2d(
    sx,
    sy,
    rho,
    yaw,
):
    """
    Rotate the Stage-4 2D Gaussian covariance
    from E0 coordinates into H0 coordinates.

    For the current frozen headlamp baseline yaw=0,
    this is numerically unchanged, but we perform
    the general transform explicitly.
    """

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

    rotated = (
        R
        @ cov
        @ R.T
    )

    sx_h = math.sqrt(
        max(
            rotated[0, 0],
            1e-12,
        )
    )

    sy_h = math.sqrt(
        max(
            rotated[1, 1],
            1e-12,
        )
    )

    rho_h = (
        rotated[0, 1]
        /
        (
            sx_h * sy_h
        )
    )

    rho_h = float(
        np.clip(
            rho_h,
            -0.999,
            0.999,
        )
    )

    return (
        sx_h,
        sy_h,
        rho_h,
    )


# ============================================================
# REAL WOMD + GAUSSIAN INPUT EXTRACTION
# ============================================================

def load_real_actor_samples():

    files = list(
        VAL_ROOT.glob(
            "*tfrecord*"
        )
    )[:MAX_FILES]

    samples = []

    for path in files:

        with path.open("rb") as f:

            while True:

                header = f.read(12)

                if not header:
                    break

                if len(header) != 12:
                    raise RuntimeError(
                        f"Truncated TFRecord header: {path}"
                    )

                length = struct.unpack(
                    "<Q",
                    header[:8],
                )[0]

                payload = f.read(
                    length
                )

                data_crc = f.read(4)

                if (
                    len(payload) != length
                    or
                    len(data_crc) != 4
                ):
                    raise RuntimeError(
                        f"Truncated TFRecord record: {path}"
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
                    current
                    +
                    1
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

                ego_state = SdcStateW(
                    center_w_m=(
                        sdc.center_x,
                        sdc.center_y,
                        0.0,
                    ),
                    heading_rad=
                        sdc.heading,
                    length_m=
                        sdc.length,
                )

                headlamp_config = (
                    HeadlampSurrogateConfig()
                )

                frames = (
                    build_anchor_frames(
                        ego_state,
                        headlamp_config,
                    )
                )

                # E0-from-H0 rotation is the configured
                # headlamp extrinsic. H0-from-E0 therefore
                # has the inverse yaw.
                headlamp_yaw_E0 = (
                    headlamp_config.yaw_rad
                )

                for track_index, track in enumerate(
                    scenario.tracks
                ):

                    actor_class = (
                        track_type_name(
                            track
                        )
                    )

                    if (
                        actor_class
                        not in
                        ALLOWED_CLASSES
                    ):
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

                    future_gt = list(
                        track.states[
                            future_start:
                            future_end
                        ]
                    )

                    if (
                        len(past)
                        != HISTORY
                        or
                        len(future_gt)
                        != FUTURE
                    ):
                        continue

                    if not all(
                        s.valid
                        for s in past
                    ):
                        continue

                    if not all(
                        s.valid
                        for s in future_gt
                    ):
                        continue

                    current_actor = past[-1]

                    if (
                        current_actor.length <= 0
                        or
                        current_actor.width <= 0
                        or
                        current_actor.height <= 0
                    ):
                        continue

                    x = []

                    for s in past:

                        point_E = (
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
                                point_E[0],
                                point_E[1],
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

                    # Current actor center in H0.
                    current_center_H = (
                        frames.T_H0_from_W
                        .apply_point(
                            (
                                current_actor.center_x,
                                current_actor.center_y,
                                current_actor.center_z,
                            )
                        )
                    )

                    samples.append(
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
                                actor_class,

                            "length_m":
                                float(
                                    current_actor.length
                                ),

                            "width_m":
                                float(
                                    current_actor.width
                                ),

                            "height_m":
                                float(
                                    current_actor.height
                                ),

                            # Causal approximation used
                            # for future box orientation.
                            "current_heading_W":
                                float(
                                    current_actor.heading
                                ),

                            "sdc_heading_W":
                                float(
                                    sdc.heading
                                ),

                            "current_center_z_H":
                                float(
                                    current_center_H[2]
                                ),

                            "frames":
                                frames,

                            "headlamp_yaw_E0":
                                float(
                                    headlamp_yaw_E0
                                ),
                        }
                    )

                    if (
                        len(samples)
                        >=
                        MAX_ACTORS
                    ):
                        return samples

    return samples


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "STAGE 6 REAL PROBABILISTIC "
        "FUTURE BOX PREDICTION SOURCE"
    )
    print("=" * 78)

    print(
        "DEVICE:",
        DEVICE
    )

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    print(
        "Checkpoint:",
        CHECKPOINT
    )

    print(
        "Max actors:",
        MAX_ACTORS
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    t0 = time.perf_counter()

    print()
    print(
        "Loading real WOMD metadata..."
    )

    samples = (
        load_real_actor_samples()
    )

    if not samples:
        raise RuntimeError(
            "No valid WOMD actor samples."
        )

    print(
        "Loaded actors:",
        len(samples)
    )


    # ========================================================
    # MODEL
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

    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()


    # ========================================================
    # BUFFERS
    # ========================================================

    scenario_ids = []
    track_indices = []
    actor_classes = []

    lengths = []
    widths = []
    heights = []

    current_headings_H = []

    mu_H_all = []
    std_H_all = []
    rho_H_all = []


    BATCH = 128


    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        for start in range(
            0,
            len(samples),
            BATCH,
        ):

            batch_samples = (
                samples[
                    start:
                    start + BATCH
                ]
            )

            x = np.stack(
                [
                    s["x"]
                    for s in batch_samples
                ],
                axis=0,
            )

            x_t = torch.from_numpy(
                x
            ).to(
                DEVICE
            )

            out = model(
                x_t
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
                batch_samples
            ):

                frames = (
                    meta["frames"]
                )

                # Future box orientation approximation:
                # preserve the causal current actor
                # heading relative to anchor SDC heading.
                heading_H = wrap_angle(
                    meta[
                        "current_heading_W"
                    ]
                    -
                    meta[
                        "sdc_heading_W"
                    ]
                )

                mu_H = np.zeros(
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

                    # Stage-4 mean is in E0 coordinates.
                    point_W = (
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

                    # Convert predicted centroid to H0.
                    point_H_planar = (
                        frames.T_H0_from_W
                        .apply_point(
                            point_W
                        )
                    )

                    # Stage 4 predicts only planar x,y.
                    # Preserve causal current vertical
                    # center in H0 as the explicit
                    # Stage-6 box-height assumption.
                    mu_H[t] = [
                        float(
                            point_H_planar[0]
                        ),
                        float(
                            point_H_planar[1]
                        ),
                        float(
                            meta[
                                "current_center_z_H"
                            ]
                        ),
                    ]

                    (
                        sx_h,
                        sy_h,
                        rho_h,
                    ) = rotate_covariance_2d(
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
                        yaw=(
                            -meta[
                                "headlamp_yaw_E0"
                            ]
                        ),
                    )

                    std_H[t] = [
                        sx_h,
                        sy_h,
                    ]

                    rho_H[t] = (
                        rho_h
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

                actor_classes.append(
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

                current_headings_H.append(
                    heading_H
                )

                mu_H_all.append(
                    mu_H
                )

                std_H_all.append(
                    std_H
                )

                rho_H_all.append(
                    rho_H
                )


            print(
                f"actors="
                f"{min(start+BATCH, len(samples))}"
                f"/{len(samples)}",
                flush=True,
            )


    # ========================================================
    # ARRAYS
    # ========================================================

    mu_H_all = np.asarray(
        mu_H_all,
        dtype=np.float32,
    )

    std_H_all = np.asarray(
        std_H_all,
        dtype=np.float32,
    )

    rho_H_all = np.asarray(
        rho_H_all,
        dtype=np.float32,
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

    track_indices = np.asarray(
        track_indices,
        dtype=np.int32,
    )

    current_headings_H = np.asarray(
        current_headings_H,
        dtype=np.float32,
    )

    actor_classes = np.asarray(
        actor_classes,
        dtype="U16",
    )

    scenario_ids = np.asarray(
        scenario_ids,
        dtype="U64",
    )


    # ========================================================
    # CHECKS
    # ========================================================

    n = len(samples)

    assert mu_H_all.shape == (
        n,
        FUTURE,
        3,
    )

    assert std_H_all.shape == (
        n,
        FUTURE,
        2,
    )

    assert rho_H_all.shape == (
        n,
        FUTURE,
    )

    if not np.isfinite(
        mu_H_all
    ).all():
        raise RuntimeError(
            "Non-finite future means."
        )

    if not np.isfinite(
        std_H_all
    ).all():
        raise RuntimeError(
            "Non-finite future std."
        )

    if np.any(
        std_H_all <= 0
    ):
        raise RuntimeError(
            "Non-positive Gaussian std."
        )

    if np.any(
        np.abs(
            rho_H_all
        ) >= 1.0
    ):
        raise RuntimeError(
            "Invalid rho."
        )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,

        scenario_id=
            scenario_ids,

        track_index=
            track_indices,

        actor_class=
            actor_classes,

        length_m=
            lengths,

        width_m=
            widths,

        height_m=
            heights,

        current_heading_H_rad=
            current_headings_H,

        future_mean_H_m=
            mu_H_all,

        future_std_xy_H_m=
            std_H_all,

        future_rho_xy_H=
            rho_H_all,
    )


    elapsed = (
        time.perf_counter()
        -
        t0
    )


    class_counts = {
        cls:
            int(
                np.sum(
                    actor_classes
                    ==
                    cls
                )
            )
        for cls in sorted(
            ALLOWED_CLASSES
        )
    }


    summary = {
        "status":
            "PASS",

        "data_source":
            "real_WOMD_validation",

        "checkpoint":
            str(
                CHECKPOINT
            ),

        "actors":
            int(n),

        "future_steps":
            FUTURE,

        "coordinate_frame":
            "H0_anchor_headlamp_frame",

        "headlamp_surrogate":
            "front_face_midpoint_surrogate",

        "future_used_as_input":
            False,

        "class_counts":
            class_counts,

        "prediction": {
            "distribution":
                "bivariate_Gaussian_xy",

            "mean_shape":
                list(
                    mu_H_all.shape
                ),

            "std_shape":
                list(
                    std_H_all.shape
                ),

            "rho_shape":
                list(
                    rho_H_all.shape
                ),
        },

        "box_geometry": {
            "dimensions":
                (
                    "causal current WOMD "
                    "length/width/height"
                ),

            "orientation":
                (
                    "causal current actor heading "
                    "relative to anchor SDC"
                ),

            "vertical_center":
                (
                    "causal current actor H0 z; "
                    "Stage-4 predictor is planar"
                ),
        },

        "scientific_scope": (
            "Stage-4 predicts future planar centroid "
            "distributions. Stage-6 preserves causal current "
            "box dimensions, current relative heading and "
            "vertical center when constructing future boxes. "
            "No future box geometry is used as model input."
        ),

        "generation_time_seconds":
            float(
                elapsed
            ),

        "output_npz":
            str(
                OUTPUT_NPZ
            ),
    }


    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )


    print()
    print("=" * 78)
    print(
        "REAL STAGE 6 PREDICTION SOURCE COMPLETE"
    )
    print("=" * 78)

    print(
        "actors =",
        n
    )

    print(
        "classes =",
        class_counts
    )

    print(
        "mean shape =",
        mu_H_all.shape
    )

    print(
        "std shape =",
        std_H_all.shape
    )

    print(
        "mean std x =",
        float(
            std_H_all[
                :,
                :,
                0
            ].mean()
        )
    )

    print(
        "mean std y =",
        float(
            std_H_all[
                :,
                :,
                1
            ].mean()
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
        OUTPUT_NPZ
    )

    print(
        "Summary:",
        OUTPUT_JSON
    )


if __name__ == "__main__":
    main()
