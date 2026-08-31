from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage5.beam_codebook import (
    BeamDecisionCell,
)


PARTA_RAW_DATA_RATE_BPS = (
    1_000_000_000.0
)

PARTA_REFERENCE_SNR_DB = (
    20.0
)

PARTA_REFERENCE_SIGNAL_POWER_NORM = (
    1.0
)

PARTA_REFERENCE_NOISE_POWER_NORM = (
    0.01
)

PARTA_REFERENCE_SNR_LINEAR = (
    PARTA_REFERENCE_SIGNAL_POWER_NORM
    /
    PARTA_REFERENCE_NOISE_POWER_NORM
)

POWER_UNIT_SEMANTICS = (
    "normalized_PartA_waveform_power_"
    "units_not_absolute_watts"
)

HEADLAMP_FRAME_SEMANTICS = (
    "Stage1_H0_is_configured_headlamp_"
    "surrogate_frame_no_second_extrinsic"
)

OPTICAL_GAIN_MODEL = (
    "constructed_Stage5_2D_Gaussian_"
    "pointing_gain"
)

OPTICAL_GAIN_HPBW_RULE = (
    "azimuth_HPBW_equals_frozen_"
    "decision_cell_width_and_"
    "baseline_elevation_HPBW_equals_"
    "azimuth_HPBW"
)

DBPSK_BER_MODEL = (
    "Stage5_analytic_DBPSK_AWGN_"
    "BER_equals_0p5_exp_minus_SNR"
)

EFFECTIVE_RATE_MODEL = (
    "Rraw_times_1_minus_BER_times_"
    "max_0_1_minus_K_Tbeam_over_Tframe"
)

MANDATORY_GEOMETRIC_LOSS = (
    1.0
)

MANDATORY_ATMOSPHERIC_LOSS = (
    1.0
)

MANDATORY_ADDITIONAL_POINTING_LOSS = (
    1.0
)


@dataclass(
    frozen=True
)
class PointingError:
    delta_azimuth_rad: float

    delta_elevation_rad: float


@dataclass(
    frozen=True
)
class EffectiveRateResult:
    raw_overhead_fraction: float

    usable_overhead_fraction: float

    payload_fraction: float

    effective_rate_bps: float


@dataclass(
    frozen=True
)
class OpticalLinkResult:
    pointing_error: PointingError

    optical_gain: float

    geometric_loss: float

    atmospheric_loss: float

    additional_pointing_loss: float

    received_power_normalized: float

    noise_power_normalized: float

    snr_linear: float

    snr_db: float

    dbpsk_ber: float

    effective_rate: EffectiveRateResult


def wrap_angle_rad(
    angle_rad: float,
) -> float:
    angle = float(
        angle_rad
    )

    if not math.isfinite(
        angle
    ):
        raise ValueError(
            "angle_rad must be finite."
        )

    return math.atan2(
        math.sin(
            angle
        ),
        math.cos(
            angle
        ),
    )


def pointing_error(
    *,
    receiver_azimuth_rad: float,
    receiver_elevation_rad: float,
    beam_center_azimuth_rad: float,
    beam_center_elevation_rad: float = 0.0,
) -> PointingError:
    """
    Angular error in the already-established Stage1 H0
    headlamp-surrogate frame.

    No second H0->headlamp transform is applied here.
    """

    values = (
        receiver_azimuth_rad,
        receiver_elevation_rad,
        beam_center_azimuth_rad,
        beam_center_elevation_rad,
    )

    if not all(
        math.isfinite(
            float(
                value
            )
        )
        for value in values
    ):
        raise ValueError(
            "Pointing angles must be finite."
        )

    return PointingError(
        delta_azimuth_rad=(
            wrap_angle_rad(
                float(
                    receiver_azimuth_rad
                )
                -
                float(
                    beam_center_azimuth_rad
                )
            )
        ),

        delta_elevation_rad=(
            float(
                receiver_elevation_rad
            )
            -
            float(
                beam_center_elevation_rad
            )
        ),
    )


def constructed_optical_pointing_gain(
    *,
    delta_azimuth_rad: float,
    delta_elevation_rad: float,
    azimuth_hpbw_rad: float,
    elevation_hpbw_rad: float | None = None,
) -> float:
    """
    Stage5 constructed normalized 2-D Gaussian pointing gain:

        Gopt =
          exp(
            -4 ln(2) [
              (daz / HPBWaz)^2
              +
              (del / HPBWel)^2
            ]
          )

    Baseline elevation HPBW:
        equal to azimuth HPBW.

    This is not claimed to be:
      - a measured headlamp radiation pattern,
      - a Part-A optical channel model,
      - an absolute antenna/optical gain in SI units.
    """

    daz = float(
        delta_azimuth_rad
    )

    dele = float(
        delta_elevation_rad
    )

    hpbw_az = float(
        azimuth_hpbw_rad
    )

    hpbw_el = (
        hpbw_az
        if elevation_hpbw_rad is None
        else
        float(
            elevation_hpbw_rad
        )
    )

    values = (
        daz,
        dele,
        hpbw_az,
        hpbw_el,
    )

    if not all(
        math.isfinite(
            value
        )
        for value in values
    ):
        raise ValueError(
            "Optical pointing parameters "
            "must be finite."
        )

    if (
        hpbw_az
        <=
        0.0
        or
        hpbw_el
        <=
        0.0
    ):
        raise ValueError(
            "Optical HPBW values "
            "must be positive."
        )

    exponent = (
        -4.0
        *
        math.log(
            2.0
        )
        *
        (
            (
                daz
                /
                hpbw_az
            )
            **
            2
            +
            (
                dele
                /
                hpbw_el
            )
            **
            2
        )
    )

    gain = math.exp(
        exponent
    )

    if (
        gain
        <
        0.0
        or
        gain
        >
        1.0
        +
        1e-15
    ):
        raise RuntimeError(
            "Constructed optical gain "
            "outside [0,1]."
        )

    return float(
        min(
            1.0,
            gain
        )
    )


def optical_gain_for_cell(
    *,
    receiver_azimuth_rad: float,
    receiver_elevation_rad: float,
    cell: BeamDecisionCell,
    beam_center_elevation_rad: float = 0.0,
) -> tuple[
    PointingError,
    float,
]:
    hpbw_az = (
        float(
            cell.upper_azimuth_rad
        )
        -
        float(
            cell.lower_azimuth_rad
        )
    )

    error = pointing_error(
        receiver_azimuth_rad=(
            receiver_azimuth_rad
        ),

        receiver_elevation_rad=(
            receiver_elevation_rad
        ),

        beam_center_azimuth_rad=(
            cell.center_azimuth_rad
        ),

        beam_center_elevation_rad=(
            beam_center_elevation_rad
        ),
    )

    gain = (
        constructed_optical_pointing_gain(
            delta_azimuth_rad=(
                error.delta_azimuth_rad
            ),

            delta_elevation_rad=(
                error.delta_elevation_rad
            ),

            azimuth_hpbw_rad=(
                hpbw_az
            ),

            elevation_hpbw_rad=(
                hpbw_az
            ),
        )
    )

    return (
        error,
        gain,
    )


def _validate_loss(
    value,
    *,
    name,
):
    result = float(
        value
    )

    if (
        not math.isfinite(
            result
        )
        or
        result
        <
        0.0
        or
        result
        >
        1.0
    ):
        raise ValueError(
            f"{name} must lie in [0,1]."
        )

    return result


def normalized_received_power(
    *,
    optical_gain: float,
    geometric_loss: float = (
        MANDATORY_GEOMETRIC_LOSS
    ),
    atmospheric_loss: float = (
        MANDATORY_ATMOSPHERIC_LOSS
    ),
    additional_pointing_loss: float = (
        MANDATORY_ADDITIONAL_POINTING_LOSS
    ),
) -> float:
    """
    Normalized Stage5 link-budget adapter:

        PRX_norm
          =
          P_ref
          * Gopt
          * Lgeo
          * Latm
          * Lpoint_extra

    with P_ref = 1 from the frozen Part-A unit-amplitude
    waveform-power convention.

    Mandatory core baseline:
        Lgeo = Latm = Lpoint_extra = 1.

    This deliberately avoids inventing unsupported aperture,
    atmosphere or absolute optical-power constants.
    """

    gain = _validate_loss(
        optical_gain,
        name="optical_gain",
    )

    geo = _validate_loss(
        geometric_loss,
        name="geometric_loss",
    )

    atm = _validate_loss(
        atmospheric_loss,
        name="atmospheric_loss",
    )

    extra = _validate_loss(
        additional_pointing_loss,
        name=(
            "additional_pointing_loss"
        ),
    )

    return float(
        PARTA_REFERENCE_SIGNAL_POWER_NORM
        *
        gain
        *
        geo
        *
        atm
        *
        extra
    )


def snr_from_normalized_power(
    received_power_normalized: float,
) -> tuple[
    float,
    float,
]:
    power = float(
        received_power_normalized
    )

    if (
        not math.isfinite(
            power
        )
        or
        power
        <
        0.0
    ):
        raise ValueError(
            "received_power_normalized "
            "must be finite and nonnegative."
        )

    snr_linear = (
        power
        /
        PARTA_REFERENCE_NOISE_POWER_NORM
    )

    if snr_linear == 0.0:

        snr_db = (
            float(
                "-inf"
            )
        )

    else:

        snr_db = (
            10.0
            *
            math.log10(
                snr_linear
            )
        )

    return (
        float(
            snr_linear
        ),
        float(
            snr_db
        ),
    )


def dbpsk_ber_from_snr(
    snr_linear: float,
) -> float:
    """
    Stage5 analytic DBPSK/AWGN mapping:

        BER = 0.5 * exp(-gamma)

    This is a constructed analytic continuation of the
    Part-A DPSK/AWGN receiver test; it is not a paper-reported
    BER curve.
    """

    gamma = float(
        snr_linear
    )

    if (
        not math.isfinite(
            gamma
        )
        or
        gamma
        <
        0.0
    ):
        raise ValueError(
            "snr_linear must be "
            "finite and nonnegative."
        )

    return float(
        0.5
        *
        math.exp(
            -gamma
        )
    )


def effective_rate(
    *,
    ber: float,
    probing_beam_count: int,
    beam_probe_time_s: float,
    frame_time_s: float,
    raw_data_rate_bps: float = (
        PARTA_RAW_DATA_RATE_BPS
    ),
) -> EffectiveRateResult:
    """
    Frozen Stage5 effective-rate equation:

        overhead_raw = K * Tbeam / Tframe

        payload_fraction
          = max(0, 1 - overhead_raw)

        Reff
          = Rraw
            * (1 - BER)
            * payload_fraction

    Formula freezes in Block5.6.

    Actual timing values are supplied by the latency contract
    and are frozen/evaluated in Block5.7.
    """

    probability = float(
        ber
    )

    k = int(
        probing_beam_count
    )

    t_beam = float(
        beam_probe_time_s
    )

    t_frame = float(
        frame_time_s
    )

    raw_rate = float(
        raw_data_rate_bps
    )

    if (
        not math.isfinite(
            probability
        )
        or
        probability
        <
        0.0
        or
        probability
        >
        1.0
    ):
        raise ValueError(
            "ber must lie in [0,1]."
        )

    if k < 0:
        raise ValueError(
            "probing_beam_count "
            "cannot be negative."
        )

    if (
        not math.isfinite(
            t_beam
        )
        or
        t_beam
        <
        0.0
    ):
        raise ValueError(
            "beam_probe_time_s must "
            "be finite and nonnegative."
        )

    if (
        not math.isfinite(
            t_frame
        )
        or
        t_frame
        <=
        0.0
    ):
        raise ValueError(
            "frame_time_s must "
            "be finite and positive."
        )

    if (
        not math.isfinite(
            raw_rate
        )
        or
        raw_rate
        <
        0.0
    ):
        raise ValueError(
            "raw_data_rate_bps must "
            "be finite and nonnegative."
        )

    overhead_raw = (
        float(
            k
        )
        *
        t_beam
        /
        t_frame
    )

    usable_overhead = min(
        1.0,
        max(
            0.0,
            overhead_raw
        ),
    )

    payload_fraction = max(
        0.0,
        1.0
        -
        overhead_raw
    )

    rate = (
        raw_rate
        *
        (
            1.0
            -
            probability
        )
        *
        payload_fraction
    )

    return EffectiveRateResult(
        raw_overhead_fraction=float(
            overhead_raw
        ),

        usable_overhead_fraction=float(
            usable_overhead
        ),

        payload_fraction=float(
            payload_fraction
        ),

        effective_rate_bps=float(
            rate
        ),
    )


def evaluate_optical_link(
    *,
    receiver_azimuth_rad: float,
    receiver_elevation_rad: float,
    cell: BeamDecisionCell,
    probing_beam_count: int,
    beam_probe_time_s: float,
    frame_time_s: float,
    geometric_loss: float = (
        MANDATORY_GEOMETRIC_LOSS
    ),
    atmospheric_loss: float = (
        MANDATORY_ATMOSPHERIC_LOSS
    ),
    additional_pointing_loss: float = (
        MANDATORY_ADDITIONAL_POINTING_LOSS
    ),
) -> OpticalLinkResult:

    (
        error,
        gain,
    ) = optical_gain_for_cell(
        receiver_azimuth_rad=(
            receiver_azimuth_rad
        ),

        receiver_elevation_rad=(
            receiver_elevation_rad
        ),

        cell=(
            cell
        ),
    )

    received = normalized_received_power(
        optical_gain=(
            gain
        ),

        geometric_loss=(
            geometric_loss
        ),

        atmospheric_loss=(
            atmospheric_loss
        ),

        additional_pointing_loss=(
            additional_pointing_loss
        ),
    )

    (
        snr_linear,
        snr_db,
    ) = snr_from_normalized_power(
        received
    )

    ber = dbpsk_ber_from_snr(
        snr_linear
    )

    rate = effective_rate(
        ber=(
            ber
        ),

        probing_beam_count=(
            probing_beam_count
        ),

        beam_probe_time_s=(
            beam_probe_time_s
        ),

        frame_time_s=(
            frame_time_s
        ),
    )

    return OpticalLinkResult(
        pointing_error=(
            error
        ),

        optical_gain=(
            gain
        ),

        geometric_loss=float(
            geometric_loss
        ),

        atmospheric_loss=float(
            atmospheric_loss
        ),

        additional_pointing_loss=float(
            additional_pointing_loss
        ),

        received_power_normalized=(
            received
        ),

        noise_power_normalized=(
            PARTA_REFERENCE_NOISE_POWER_NORM
        ),

        snr_linear=(
            snr_linear
        ),

        snr_db=(
            snr_db
        ),

        dbpsk_ber=(
            ber
        ),

        effective_rate=(
            rate
        ),
    )
