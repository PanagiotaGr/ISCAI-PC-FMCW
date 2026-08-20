
from __future__ import annotations

import math


def compute_measurement_uncertainty(
    range_m: float,
    snr_db: float,
):
    """
    Deterministic measurement uncertainty model.

    Smoke-level model:
    - larger range -> larger uncertainty
    - lower SNR -> larger uncertainty

    Future replacement:
    CRLB-derived PC-FMCW covariance.
    """

    snr_linear = max(
        10.0 ** (snr_db / 10.0),
        1e-12,
    )


    range_factor = max(
        range_m,
        1.0,
    )


    sigma_range = (
        0.05
        *
        range_factor
        /
        math.sqrt(
            snr_linear
        )
    )


    sigma_radial_velocity = (
        0.05
        *
        range_factor
        /
        math.sqrt(
            snr_linear
        )
    )


    sigma_azimuth = (
        0.002
        *
        range_factor
        /
        math.sqrt(
            snr_linear
        )
    )


    sigma_elevation = (
        0.002
        *
        range_factor
        /
        math.sqrt(
            snr_linear
        )
    )


    R = [
        [
            sigma_range ** 2,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            sigma_radial_velocity ** 2,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            sigma_azimuth ** 2,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            sigma_elevation ** 2,
        ],
    ]


    return {
        "sigma_range_m":
            sigma_range,

        "sigma_radial_velocity_mps":
            sigma_radial_velocity,

        "sigma_azimuth_rad":
            sigma_azimuth,

        "sigma_elevation_rad":
            sigma_elevation,

        "covariance":
            R,
    }
