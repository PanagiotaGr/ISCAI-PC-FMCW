
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, Counter
from pathlib import Path


OBS = Path(
    "artifacts/stage5_pcfmcw_observations.json"
)

TRAJ = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

REPORT = Path(
    "reports/block5_measurement_driven_imm_evaluation.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)


def wrap_angle(x):

    return math.atan2(
        math.sin(x),
        math.cos(x),
    )


def measurement_from_point(
    point,
):

    x = float(point[0])
    y = float(point[1])
    z = float(point[2])

    r = math.sqrt(
        x*x + y*y + z*z
    )

    az = math.atan2(
        y,
        x,
    )

    elev = math.atan2(
        z,
        math.sqrt(
            x*x+y*y
        ),
    )

    return [
        r,
        0.0,
        az,
        elev,
    ]


observations = json.loads(
    OBS.read_text()
)

trajectories = json.loads(
    TRAJ.read_text()
)


obs_groups = {}

for item in observations:

    key = (
        item["scenario_id"],
        int(item["track_index"]),
    )

    obs_groups[key] = item


traj_groups = defaultdict(dict)


for item in trajectories:

    key = (
        item["scenario_id"],
        int(item["track_index"]),
    )

    traj_groups[key][
        item["model_name"]
    ] = item["trajectory"]


actors = 0
probability_valid = 0

dominant = Counter()

nis_sum = Counter()
nis_count = Counter()

entropy_sum = 0.0


for key, models in traj_groups.items():

    if key not in obs_groups:
        continue


    if not all(
        m in models
        for m in MODELS
    ):
        continue


    obs = obs_groups[key]


    z = [
        float(obs["range_m"]),
        float(obs["radial_velocity_mps"]),
        float(obs["azimuth_rad"]),
        float(obs["elevation_rad"]),
    ]


    likelihoods = {}


    for model in MODELS:

        prediction = models[model][0]

        h = measurement_from_point(
            prediction
        )


        residual = []


        for i in range(4):

            d = z[i] - h[i]

            if i == 2:
                d = wrap_angle(d)

            residual.append(d)


        covariance = obs[
            "measurement_covariance"
        ]


        nis = 0.0


        for i in range(4):

            variance = float(
                covariance[i][i]
            )

            if variance > 0:

                nis += (
                    residual[i]
                    *
                    residual[i]
                    /
                    variance
                )


        likelihood = math.exp(
            -0.5 * nis
        )


        likelihoods[model] = likelihood

        nis_sum[model] += nis
        nis_count[model] += 1


    total = sum(
        likelihoods.values()
    )


    if total <= 0:
        continue


    probabilities = {
        m:
        likelihoods[m]/total
        for m in MODELS
    }


    if abs(
        sum(probabilities.values())
        -
        1.0
    ) < 1e-9:

        probability_valid += 1


    best = max(
        probabilities,
        key=probabilities.get
    )

    dominant[best] += 1


    entropy = 0.0

    for p in probabilities.values():

        if p > 0:
            entropy -= (
                p
                *
                math.log(p)
            )

    entropy_sum += entropy

    actors += 1



mean_nis = {}

for model in MODELS:

    mean_nis[model] = (
        nis_sum[model]
        /
        nis_count[model]
        if nis_count[model]
        else 0.0
    )


mean_entropy = (
    entropy_sum / actors
    if actors
    else 0.0
)


payload = {

    "actors":
        actors,

    "models":
        list(MODELS),

    "probability_valid":
        probability_valid,

    "dominant_model_distribution":
        dict(dominant),

    "mean_nis_by_model":
        mean_nis,

    "mean_probability_entropy":
        mean_entropy,

    "future_used":
        False,
}


status = (
    "PASS"
    if (
        actors > 0
        and probability_valid == actors
    )
    else
    "FAIL"
)


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
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
    "===== Stage5 Measurement Driven IMM Evaluation ====="
)

print(
    "actors =",
    actors
)

print(
    "probability_valid =",
    probability_valid
)

print(
    "dominant_model_distribution =",
    dict(dominant)
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
