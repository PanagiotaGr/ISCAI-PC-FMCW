from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CalibrationMetrics:

    nll: float

    brier: float

    empirical_coverage: float

    ece: float



def gaussian_nll(
    error_squared: float,
    variance: float,
) -> float:
    """
    Simplified Gaussian negative log likelihood.
    """

    if variance <= 0:

        raise ValueError(
            "Variance must be positive."
        )


    return (
        0.5
        *
        (
            math.log(
                2.0
                *
                math.pi
                *
                variance
            )
            +
            error_squared / variance
        )
    )



def brier_score(
    probability: float,
    outcome: float,
) -> float:

    return (
        probability
        -
        outcome
    ) ** 2



def empirical_coverage(
    errors,
    threshold,
):

    if len(errors) == 0:

        raise ValueError(
            "Empty errors."
        )


    inside = sum(
        1
        for e in errors
        if e <= threshold
    )


    return (
        inside
        /
        len(errors)
    )



def expected_calibration_error(
    predicted,
    observed,
):

    if len(predicted) != len(observed):

        raise ValueError(
            "Length mismatch."
        )


    if len(predicted) == 0:

        raise ValueError(
            "Empty calibration data."
        )


    return sum(
        abs(
            p-o
        )
        for p,o
        in zip(
            predicted,
            observed,
        )
    ) / len(predicted)



def evaluate_calibration(
    errors,
    variance,
    predicted,
    observed,
):

    nll = sum(
        gaussian_nll(
            e*e,
            variance,
        )
        for e in errors
    ) / len(errors)


    brier = sum(
        brier_score(
            p,
            o,
        )
        for p,o
        in zip(
            predicted,
            observed,
        )
    ) / len(predicted)


    coverage = empirical_coverage(
        errors,
        threshold=variance**0.5,
    )


    ece = expected_calibration_error(
        predicted,
        observed,
    )


    return CalibrationMetrics(

        nll=nll,

        brier=brier,

        empirical_coverage=coverage,

        ece=ece,
    )
