from __future__ import annotations


import json
from pathlib import Path


from iscai_stage4.predictor.stage4_input import (
    Stage4ActorInput,
)

from iscai_stage4.predictor.deterministic_gru import (
    DeterministicGRUPredictor,
)

from iscai_stage4.predictor.gaussian_gru import (
    GaussianGRUPredictor,
)

from iscai_stage4.predictor.gmm_predictor import (
    GMMPredictor,
)



REPORT = Path(
    "reports/block4_real_predictor_smoke.json"
)



actor = Stage4ActorInput(

    time_index=20,

    position=(
        10.0,
        5.0,
        0.0,
    ),

    velocity=(
        8.0,
        0.5,
        0.0,
    ),

    heading_rad=0.1,

    range_m=20.0,

    radial_velocity_mps=-7.5,

    measurement_covariance=(

        (
            0.1,
            0.0,
            0.0,
        ),

        (
            0.0,
            0.1,
            0.0,
        ),

        (
            0.0,
            0.0,
            0.2,
        ),
    ),

    length_m=4.5,

    width_m=1.8,

    height_m=1.6,

    actor_class="vehicle",
)



deterministic = (
    DeterministicGRUPredictor(
        horizon=10
    )
    .predict(actor)
)



gaussian = (
    GaussianGRUPredictor(
        horizon=10
    )
    .predict(actor)
)



gmm = (
    GMMPredictor(
        horizon=10
    )
    .predict(actor)
)



records = {


    "deterministic_horizon":
        deterministic.horizon,


    "gaussian_horizon":
        gaussian.horizon,


    "gmm_modes":
        gmm.mode_count,


    "gmm_probability_sum":
        gmm.probabilities_sum(),


    "future_used":
        False,

}



if records["deterministic_horizon"] != 10:

    raise RuntimeError(
        "Invalid deterministic horizon."
    )


if records["gaussian_horizon"] != 10:

    raise RuntimeError(
        "Invalid gaussian horizon."
    )


if records["gmm_modes"] != 5:

    raise RuntimeError(
        "Invalid GMM modes."
    )


if abs(
    records["gmm_probability_sum"]
    -
    1.0
) > 1e-9:

    raise RuntimeError(
        "Invalid GMM probabilities."
    )



REPORT.parent.mkdir(
    exist_ok=True
)


REPORT.write_text(
    json.dumps(
        records,
        indent=2,
        sort_keys=True,
    )
)



print(
    "===== Stage4 Real Predictor Smoke ====="
)

print(
    "deterministic horizon =",
    records["deterministic_horizon"]
)

print(
    "gaussian horizon =",
    records["gaussian_horizon"]
)

print(
    "gmm modes =",
    records["gmm_modes"]
)

print(
    "future_used = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)
