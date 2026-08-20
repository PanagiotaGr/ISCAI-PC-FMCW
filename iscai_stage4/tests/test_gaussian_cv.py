
from iscai_stage4.predictors import (
    predict_gaussian_cv,
)


def test_gaussian_cv_prediction():

    prediction = predict_gaussian_cv(
        initial_position=(0.0,0.0,0.0),
        velocity=(10.0,0.0,0.0),
        horizon_s=1.0,
        dt=0.1,
    )

    assert prediction.model_name == "GaussianCV"

    assert len(
        prediction.timestamps_s
    ) == 11


    assert (
        prediction.mean_positions_m[-1][0]
        > 9.9
    )


    assert (
        prediction.covariance_m2[-1][0][0]
        >
        prediction.covariance_m2[0][0][0]
    )
