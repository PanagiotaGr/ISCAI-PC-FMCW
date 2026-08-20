from __future__ import annotations


from iscai_stage4.calibration import (
    scale_covariance,
)


CALIBRATION_ALPHA = 0.25



def calibrate_prediction(
    prediction,
    *,
    alpha: float = CALIBRATION_ALPHA,
):
    """
    Apply covariance calibration.

    Mean trajectory is unchanged.
    Only uncertainty is calibrated.
    """


    calibrated_covariance = tuple(
        scale_covariance(
            covariance,
            alpha,
        )
        for covariance in (
            prediction.covariance_m2
        )
    )


    return (
        prediction.mean_positions_m,
        calibrated_covariance,
    )
