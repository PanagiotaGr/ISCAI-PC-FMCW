from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GaussianComponent:

    weight: float

    mean_positions_m: tuple

    covariance_m2: tuple



@dataclass(frozen=True)
class GMMPrediction:

    components: tuple

    model_name: str = "GaussianMixture"



def build_two_mode_gmm(
    *,
    base_prediction,
):
    """
    Simple two-mode probabilistic wrapper.

    Mode 1:
        original GaussianCV

    Mode 2:
        lateral alternative hypothesis
    """


    mean = (
        base_prediction.mean_positions_m
    )

    covariance = (
        base_prediction.covariance_m2
    )


    shifted = tuple(
        (
            p[0],
            p[1] + 1.5,
            p[2],
        )
        for p in mean
    )


    components = (

        GaussianComponent(
            weight=0.7,
            mean_positions_m=mean,
            covariance_m2=covariance,
        ),


        GaussianComponent(
            weight=0.3,
            mean_positions_m=shifted,
            covariance_m2=covariance,
        ),

    )


    return GMMPrediction(
        components=components,
    )
