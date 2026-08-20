
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage3.baselines.causal_state import (
    estimate_causal_motion_state,
)

from iscai_stage3.baselines.constant_velocity import (
    predict_constant_velocity,
)

from iscai_stage3.baselines.constant_acceleration import (
    predict_constant_acceleration,
)

from iscai_stage3.baselines.ctrv import (
    predict_ctrv,
)

from iscai_stage3.baselines.kalman import (
    predict_kalman_cv,
)


DATA = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/training/motion"
)

REPORT = Path(
    "reports/block5_measurement_driven_imm_v2.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)

MAX_FILES = 50


# Same smoke-level measurement covariance contract
# used by the current PC-FMCW layer.
VAR_RANGE = 0.10 ** 2
VAR_VR = 0.10 ** 2
VAR_AZ = 0.005 ** 2
VAR_EL = 0.005 ** 2


def wrap_angle(value):

    return math.atan2(
        math.sin(value),
        math.cos(value),
    )


def range_of(point):

    x, y, z = point

    return math.sqrt(
        x*x + y*y + z*z
    )


def measurement_from_actual(
    current,
    previous,
    dt,
):

    x, y, z = current

    r = range_of(
        current
    )

    r_prev = range_of(
        previous
    )

    vr = (
        r - r_prev
    ) / dt

    az = math.atan2(
        y,
        x,
    )

    horizontal = math.sqrt(
        x*x + y*y
    )

    el = math.atan2(
        z,
        horizontal,
    )

    return (
        r,
        vr,
        az,
        el,
    )


def measurement_from_prediction(
    predicted,
    previous_actual,
    dt,
):

    x, y, z = predicted

    r = range_of(
        predicted
    )

    r_prev = range_of(
        previous_actual
    )

    vr = (
        r - r_prev
    ) / dt

    az = math.atan2(
        y,
        x,
    )

    horizontal = math.sqrt(
        x*x + y*y
    )

    el = math.atan2(
        z,
        horizontal,
    )

    return (
        r,
        vr,
        az,
        el,
    )


def compute_acceleration(
    positions,
    timestamps,
):

    p0 = positions[-3]
    p1 = positions[-2]
    p2 = positions[-1]

    t0 = timestamps[-3]
    t1 = timestamps[-2]
    t2 = timestamps[-1]

    dt01 = t1 - t0
    dt12 = t2 - t1

    if (
        dt01 <= 1e-9
        or dt12 <= 1e-9
    ):
        return (
            0.0,
            0.0,
            0.0,
        )

    v01 = (
        (p1[0] - p0[0]) / dt01,
        (p1[1] - p0[1]) / dt01,
        (p1[2] - p0[2]) / dt01,
    )

    v12 = (
        (p2[0] - p1[0]) / dt12,
        (p2[1] - p1[1]) / dt12,
        (p2[2] - p1[2]) / dt12,
    )

    dt_v = 0.5 * (
        dt01 + dt12
    )

    return (
        (v12[0] - v01[0]) / dt_v,
        (v12[1] - v01[1]) / dt_v,
        (v12[2] - v01[2]) / dt_v,
    )


def compute_yaw_rate(
    positions,
    timestamps,
):

    p0 = positions[-3]
    p1 = positions[-2]
    p2 = positions[-1]

    t0 = timestamps[-3]
    t1 = timestamps[-2]
    t2 = timestamps[-1]

    dt01 = t1 - t0
    dt12 = t2 - t1

    if (
        dt01 <= 1e-9
        or dt12 <= 1e-9
    ):
        return 0.0

    vx1 = (
        p1[0] - p0[0]
    ) / dt01

    vy1 = (
        p1[1] - p0[1]
    ) / dt01

    vx2 = (
        p2[0] - p1[0]
    ) / dt12

    vy2 = (
        p2[1] - p1[1]
    ) / dt12

    speed1 = math.hypot(
        vx1,
        vy1,
    )

    speed2 = math.hypot(
        vx2,
        vy2,
    )

    if (
        speed1 <= 1e-6
        or speed2 <= 1e-6
    ):
        return 0.0

    heading1 = math.atan2(
        vy1,
        vx1,
    )

    heading2 = math.atan2(
        vy2,
        vx2,
    )

    delta_heading = wrap_angle(
        heading2 - heading1
    )

    return (
        delta_heading
        /
        dt12
    )


files = sorted(
    DATA.glob(
        "paired-from-training.tfrecord-*"
    )
)[:MAX_FILES]


actors = 0
updates = 0
probability_valid = 0

dominant = Counter()

nis_sum = {
    model: 0.0
    for model in MODELS
}

nis_count = {
    model: 0
    for model in MODELS
}

probability_sum = {
    model: 0.0
    for model in MODELS
}

entropy_sum = 0.0

nonuniform_updates = 0

classes = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for file in files:

    scenario = read_first_scenario(
        file
    )

    adapted = adapt_causal_womd_scenario(
        scenario
    )


    for actor in adapted.actors:

        track_index = actor.track_index

        if not (
            0 <= track_index < len(
                scenario.tracks
            )
        ):
            continue


        object_type = int(
            scenario.tracks[
                track_index
            ].object_type
        )

        actor_type = {
            1: "VEHICLE",
            2: "PEDESTRIAN",
            3: "CYCLIST",
        }.get(
            object_type,
            "UNKNOWN",
        )


        if actor_type not in classes:
            continue


        history = actor.artifact.history

        positions = []
        timestamps = []


        for i, valid in enumerate(
            history.state_valid
        ):

            if not valid:
                continue

            positions.append(
                (
                    float(
                        history.position_W_m[i][0]
                    ),
                    float(
                        history.position_W_m[i][1]
                    ),
                    float(
                        history.position_W_m[i][2]
                    ),
                )
            )

            timestamps.append(
                float(
                    history.timestamps_s[i]
                )
            )


        if len(positions) < 4:
            continue


        actor_updates = 0


        # Predict sample k using only samples < k.
        for k in range(
            3,
            len(positions)
        ):

            past_positions = (
                positions[
                    k-3:k
                ]
            )

            past_timestamps = (
                timestamps[
                    k-3:k
                ]
            )


            dt = (
                timestamps[k]
                -
                timestamps[k-1]
            )


            if dt <= 1e-9:
                continue


            motion = (
                estimate_causal_motion_state(
                    positions=tuple(
                        past_positions
                    ),
                    timestamps=tuple(
                        past_timestamps
                    ),
                )
            )


            acceleration = (
                compute_acceleration(
                    past_positions,
                    past_timestamps,
                )
            )


            vx, vy, vz = (
                motion.velocity
            )

            speed = math.hypot(
                vx,
                vy,
            )

            heading = (
                math.atan2(
                    vy,
                    vx,
                )
                if speed > 1e-9
                else 0.0
            )


            yaw_rate = (
                compute_yaw_rate(
                    past_positions,
                    past_timestamps,
                )
            )


            outputs = {

                "CV":
                    predict_constant_velocity(
                        position=
                            motion.position,

                        velocity=
                            motion.velocity,

                        horizon_s=
                            dt,

                        dt=
                            dt,
                    ),

                "CA":
                    predict_constant_acceleration(
                        position=
                            motion.position,

                        velocity=
                            motion.velocity,

                        acceleration=
                            acceleration,

                        horizon_s=
                            dt,

                        dt=
                            dt,
                    ),

                "CTRV":
                    predict_ctrv(
                        position=
                            motion.position,

                        speed=
                            speed,

                        heading_rad=
                            heading,

                        yaw_rate=
                            yaw_rate,

                        horizon_s=
                            dt,

                        dt=
                            dt,
                    ),

                "KALMAN":
                    predict_kalman_cv(
                        position=
                            motion.position,

                        velocity=
                            motion.velocity,

                        horizon_s=
                            dt,

                        dt=
                            dt,
                    ),
            }


            z = measurement_from_actual(
                current=
                    positions[k],

                previous=
                    positions[k-1],

                dt=
                    dt,
            )


            log_likelihoods = {}


            for model in MODELS:

                trajectory = outputs[
                    model
                ]

                if not trajectory:
                    continue


                predicted_point = (
                    trajectory[-1]
                )


                h = measurement_from_prediction(
                    predicted=
                        predicted_point,

                    previous_actual=
                        positions[k-1],

                    dt=
                        dt,
                )


                residual = (
                    z[0] - h[0],
                    z[1] - h[1],
                    wrap_angle(
                        z[2] - h[2]
                    ),
                    wrap_angle(
                        z[3] - h[3]
                    ),
                )


                nis = (
                    residual[0]**2
                    /
                    VAR_RANGE
                    +
                    residual[1]**2
                    /
                    VAR_VR
                    +
                    residual[2]**2
                    /
                    VAR_AZ
                    +
                    residual[3]**2
                    /
                    VAR_EL
                )


                if not math.isfinite(
                    nis
                ):
                    continue


                nis_sum[model] += nis
                nis_count[model] += 1


                # Use log likelihood to avoid
                # numerical underflow for large NIS.
                log_likelihoods[
                    model
                ] = (
                    -0.5 * nis
                )


            if len(
                log_likelihoods
            ) != len(MODELS):
                continue


            max_log = max(
                log_likelihoods.values()
            )


            weights = {
                model:
                    math.exp(
                        log_likelihoods[model]
                        -
                        max_log
                    )
                for model in MODELS
            }


            normalizer = sum(
                weights.values()
            )


            if (
                not math.isfinite(
                    normalizer
                )
                or normalizer <= 0.0
            ):
                continue


            probabilities = {
                model:
                    weights[model]
                    /
                    normalizer
                for model in MODELS
            }


            probability_error = abs(
                sum(
                    probabilities.values()
                )
                -
                1.0
            )


            if probability_error > 1e-9:
                continue


            probability_valid += 1


            for model in MODELS:

                probability_sum[
                    model
                ] += (
                    probabilities[
                        model
                    ]
                )


            best = max(
                MODELS,
                key=lambda model: (
                    probabilities[
                        model
                    ],
                    -MODELS.index(
                        model
                    ),
                ),
            )


            dominant[
                best
            ] += 1


            entropy = 0.0

            for probability in (
                probabilities.values()
            ):

                if probability > 0.0:

                    entropy -= (
                        probability
                        *
                        math.log(
                            probability
                        )
                    )


            entropy_sum += entropy


            spread = (
                max(
                    probabilities.values()
                )
                -
                min(
                    probabilities.values()
                )
            )


            if spread > 1e-3:
                nonuniform_updates += 1


            updates += 1
            actor_updates += 1


        if actor_updates > 0:

            actors += 1

            classes[
                actor_type
            ] += 1



mean_nis = {
    model:
        (
            nis_sum[model]
            /
            nis_count[model]
            if nis_count[model] > 0
            else 0.0
        )
    for model in MODELS
}


mean_probability = {
    model:
        (
            probability_sum[model]
            /
            updates
            if updates > 0
            else 0.0
        )
    for model in MODELS
}


mean_entropy = (
    entropy_sum
    /
    updates
    if updates > 0
    else 0.0
)


max_entropy = math.log(
    len(MODELS)
)


normalized_entropy = (
    mean_entropy
    /
    max_entropy
    if max_entropy > 0
    else 0.0
)


nonuniform_fraction = (
    nonuniform_updates
    /
    updates
    if updates > 0
    else 0.0
)


payload = {

    "files":
        len(files),

    "actors":
        actors,

    "updates":
        updates,

    "models":
        list(MODELS),

    "probability_valid":
        probability_valid,

    "dominant_model_distribution":
        dict(
            dominant
        ),

    "mean_probability_by_model":
        mean_probability,

    "mean_nis_by_model":
        mean_nis,

    "mean_probability_entropy":
        mean_entropy,

    "maximum_entropy":
        max_entropy,

    "normalized_entropy":
        normalized_entropy,

    "nonuniform_updates":
        nonuniform_updates,

    "nonuniform_fraction":
        nonuniform_fraction,

    "vehicle":
        classes["VEHICLE"],

    "pedestrian":
        classes["PEDESTRIAN"],

    "cyclist":
        classes["CYCLIST"],

    "evaluation_mode":
        "causal_one_step_ahead",

    "future_used":
        False,
}


status = (
    "PASS"
    if (
        actors > 0
        and updates > 0
        and probability_valid
        ==
        updates
    )
    else
    "FAIL"
)


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode()
).hexdigest()


report = {
    **payload,

    "sha256":
        sha,

    "status":
        status,
}


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 Measurement Driven IMM V2 ====="
)

print(
    "files =",
    len(files)
)

print(
    "actors =",
    actors
)

print(
    "updates =",
    updates
)

print(
    "probability_valid =",
    probability_valid
)

print(
    "dominant_model_distribution =",
    dict(
        dominant
    )
)

print(
    "mean_probability_by_model =",
    mean_probability
)

print(
    "mean_nis_by_model =",
    mean_nis
)

print(
    "mean_probability_entropy =",
    mean_entropy
)

print(
    "maximum_entropy =",
    max_entropy
)

print(
    "normalized_entropy =",
    normalized_entropy
)

print(
    "nonuniform_updates =",
    nonuniform_updates
)

print(
    "nonuniform_fraction =",
    nonuniform_fraction
)

print(
    "vehicle =",
    classes["VEHICLE"]
)

print(
    "pedestrian =",
    classes["PEDESTRIAN"]
)

print(
    "cyclist =",
    classes["CYCLIST"]
)

print(
    "evaluation_mode = causal_one_step_ahead"
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    sha
)

print(
    "STATUS =",
    status
)

print(
    "report =",
    REPORT
)


if status != "PASS":
    raise SystemExit(1)
