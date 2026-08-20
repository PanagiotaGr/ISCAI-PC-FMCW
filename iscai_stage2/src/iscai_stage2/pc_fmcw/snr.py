from __future__ import annotations

import math


def snr_db_to_linear(
    snr_db: float,
) -> float:
    if not math.isfinite(snr_db):
        raise ValueError(
            "SNR in dB must be finite."
        )

    return 10.0 ** (
        snr_db / 10.0
    )


def snr_linear_to_db(
    snr_linear: float,
) -> float:
    if (
        not math.isfinite(snr_linear)
        or snr_linear <= 0.0
    ):
        raise ValueError(
            "Linear SNR must be finite and positive."
        )

    return 10.0 * math.log10(
        snr_linear
    )


def awgn_noise_power_for_snr(
    *,
    signal_power: float,
    snr_db: float,
) -> float:
    """
    Notebook communication-AWGN relation:

        noise_power = signal_power / gamma

    Important:
    this is NOT a Stage-2 sensing link-budget model.
    """
    if (
        not math.isfinite(signal_power)
        or signal_power < 0.0
    ):
        raise ValueError(
            "Signal power must be finite and non-negative."
        )

    gamma = snr_db_to_linear(
        snr_db
    )

    return signal_power / gamma
