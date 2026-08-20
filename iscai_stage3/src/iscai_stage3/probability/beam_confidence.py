from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BeamConfidence:
    """
    Confidence of a predicted beam direction.
    """

    angular_error_rad: float
    sigma_rad: float
    confidence: float



def gaussian_confidence(
    *,
    error_rad: float,
    sigma_rad: float,
) -> float:
    """
    Gaussian beam likelihood.

    p ~ exp(-0.5*(e/sigma)^2)

    """

    if not math.isfinite(error_rad):
        raise ValueError(
            "Error must be finite."
        )

    if (
        not math.isfinite(sigma_rad)
        or sigma_rad <= 0.0
    ):
        raise ValueError(
            "sigma must be positive."
        )


    normalized = (
        error_rad / sigma_rad
    )

    return math.exp(
        -0.5 * normalized * normalized
    )



def beam_confidence_from_covariance(
    *,
    angular_error_rad: float,
    azimuth_variance: float,
    elevation_variance: float,
) -> BeamConfidence:
    """
    Combine azimuth/elevation uncertainty.

    sigma_theta =
        sqrt(
          sigma_az^2 +
          sigma_el^2
        )

    """

    if azimuth_variance < 0:
        raise ValueError(
            "Negative azimuth variance."
        )

    if elevation_variance < 0:
        raise ValueError(
            "Negative elevation variance."
        )


    sigma = math.sqrt(
        azimuth_variance
        +
        elevation_variance
    )


    confidence = gaussian_confidence(
        error_rad=angular_error_rad,
        sigma_rad=sigma,
    )


    return BeamConfidence(
        angular_error_rad=(
            angular_error_rad
        ),
        sigma_rad=sigma,
        confidence=confidence,
    )
