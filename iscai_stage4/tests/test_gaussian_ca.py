from iscai_stage4.predictors import (
    predict_gaussian_ca,
)


def test_gaussian_ca_prediction():

    prediction = predict_gaussian_ca(
        initial_position=(0.0,0.0,0.0),
        velocity=(5.0,0.0,0.0),
        acceleration=(2.0,0.0,0.0),
        horizon_s=1.0,
        dt=0.1,
    )


    assert prediction.model_name == "GaussianCA"


    assert len(
        prediction.timestamps_s
    ) == 11


    assert (
        prediction.mean_positions_m[-1][0]
        > 5.9
    )
