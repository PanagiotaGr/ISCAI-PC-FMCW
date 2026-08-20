from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage2.pc_fmcw.snr import (
    snr_db_to_linear,
    snr_linear_to_db,
)


FIXED_SNR_MODEL = "fixed_snr"

FACTORIZED_POWER_LAW_MODEL = (
    "factorized_power_law"
)

FACTORIZED_MODEL_SEMANTICS = (
    "stage2_declared_sensing_snr_assumption_not_part_a"
)


@dataclass(frozen=True)
class SensingSnrInputs:
    range_m: float

    # Echo-amplitude coefficient analogous to alpha_i
    # in the Part-A received-signal expression.
    target_coefficient_amplitude: float = 1.0

    # Multiplicative received-power factor.
    # 1 = no visibility penalty.
    # Must be >0 for a valid detected measurement.
    visibility_power_factor: float = 1.0

    # Explicit place for class/dimensions/clutter/angular
    # effects without inventing hidden class-specific values.
    extra_gain_db: float = 0.0


@dataclass(frozen=True)
class SensingSnrResult:
    snr_db: float
    snr_linear: float

    model_name: str

    range_power_factor: float
    target_power_factor: float
    visibility_power_factor: float
    extra_power_factor: float

    model_semantics: str


@dataclass(frozen=True)
class FixedSensingSnrConfig:
    snr_db: float

    def __post_init__(self) -> None:
        if not math.isfinite(
            self.snr_db
        ):
            raise ValueError(
                "Fixed sensing SNR must be finite."
            )


@dataclass(frozen=True)
class FactorizedSensingSnrConfig:
    """
    Explicit Stage-2 SNR assumption.

    No default is provided for range_power_exponent because
    Part A does not define a range-dependent sensing link
    budget. The exponent must therefore be an explicit
    experimental/configuration choice.
    """

    reference_range_m: float
    reference_snr_db: float

    range_power_exponent: float

    reference_target_coefficient_amplitude: float = 1.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(
                self.reference_range_m
            )
            or self.reference_range_m
            <= 0.0
        ):
            raise ValueError(
                "reference_range_m must be finite "
                "and positive."
            )

        if not math.isfinite(
            self.reference_snr_db
        ):
            raise ValueError(
                "reference_snr_db must be finite."
            )

        if (
            not math.isfinite(
                self.range_power_exponent
            )
            or self.range_power_exponent
            < 0.0
        ):
            raise ValueError(
                "range_power_exponent must be finite "
                "and non-negative."
            )

        if (
            not math.isfinite(
                self.reference_target_coefficient_amplitude
            )
            or
            self.reference_target_coefficient_amplitude
            <= 0.0
        ):
            raise ValueError(
                "Reference target coefficient must "
                "be finite and positive."
            )


def _validate_inputs(
    inputs: SensingSnrInputs,
) -> None:
    if (
        not math.isfinite(
            inputs.range_m
        )
        or inputs.range_m <= 0.0
    ):
        raise ValueError(
            "Sensing range must be finite and positive."
        )

    if (
        not math.isfinite(
            inputs.target_coefficient_amplitude
        )
        or
        inputs.target_coefficient_amplitude
        <= 0.0
    ):
        raise ValueError(
            "Target coefficient amplitude must be "
            "finite and positive."
        )

    if (
        not math.isfinite(
            inputs.visibility_power_factor
        )
        or inputs.visibility_power_factor
        <= 0.0
    ):
        raise ValueError(
            "Visibility power factor must be finite "
            "and strictly positive for a detected "
            "measurement."
        )

    if not math.isfinite(
        inputs.extra_gain_db
    ):
        raise ValueError(
            "extra_gain_db must be finite."
        )


def fixed_sensing_snr(
    *,
    inputs: SensingSnrInputs,
    config: FixedSensingSnrConfig,
) -> SensingSnrResult:
    """
    Fixed-SNR experimental baseline.

    Geometry inputs are validated but do not alter SNR.
    """
    _validate_inputs(
        inputs
    )

    gamma = snr_db_to_linear(
        config.snr_db
    )

    return SensingSnrResult(
        snr_db=config.snr_db,
        snr_linear=gamma,

        model_name=FIXED_SNR_MODEL,

        range_power_factor=1.0,
        target_power_factor=1.0,
        visibility_power_factor=1.0,
        extra_power_factor=1.0,

        model_semantics=(
            "stage2_fixed_snr_experimental_baseline"
        ),
    )


def factorized_sensing_snr(
    *,
    inputs: SensingSnrInputs,
    config: FactorizedSensingSnrConfig,
) -> SensingSnrResult:
    """
    Explicit configurable Stage-2 sensing-SNR model:

      gamma =
          gamma_ref
          * (r_ref / r)^p
          * (alpha / alpha_ref)^2
          * q_visibility
          * 10^(G_extra_dB / 10)

    alpha is an amplitude coefficient, hence alpha^2
    enters received power/SNR.

    This model is NOT claimed to be reproduced from Part A.
    """
    _validate_inputs(
        inputs
    )

    gamma_ref = snr_db_to_linear(
        config.reference_snr_db
    )

    range_factor = (
        config.reference_range_m
        / inputs.range_m
    ) ** config.range_power_exponent

    amplitude_ratio = (
        inputs.target_coefficient_amplitude
        /
        config.reference_target_coefficient_amplitude
    )

    target_power_factor = (
        amplitude_ratio ** 2
    )

    visibility_factor = (
        inputs.visibility_power_factor
    )

    extra_power_factor = (
        10.0 ** (
            inputs.extra_gain_db
            / 10.0
        )
    )

    gamma = (
        gamma_ref
        * range_factor
        * target_power_factor
        * visibility_factor
        * extra_power_factor
    )

    if (
        not math.isfinite(gamma)
        or gamma <= 0.0
    ):
        raise ValueError(
            "Computed sensing SNR must be finite "
            "and positive."
        )

    return SensingSnrResult(
        snr_db=snr_linear_to_db(
            gamma
        ),
        snr_linear=gamma,

        model_name=(
            FACTORIZED_POWER_LAW_MODEL
        ),

        range_power_factor=(
            range_factor
        ),
        target_power_factor=(
            target_power_factor
        ),
        visibility_power_factor=(
            visibility_factor
        ),
        extra_power_factor=(
            extra_power_factor
        ),

        model_semantics=(
            FACTORIZED_MODEL_SEMANTICS
        ),
    )
