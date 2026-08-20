from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage2.pc_fmcw.reference import (
    FROZEN_PART_A,
    PartAReferenceParameters,
)
from iscai_stage2.pc_fmcw.snr import (
    snr_db_to_linear,
)


@dataclass(frozen=True)
class PartACrlbBounds:
    """
    Single-target CRLB from Part-A notebook Eq. (7).

    These are lower bounds on measurement variance,
    not realized estimator error.
    """

    snr_db: float
    snr_linear: float

    range_variance_m2: float
    range_std_m: float

    radial_velocity_variance_m2ps2: float
    radial_velocity_std_mps: float


@dataclass(frozen=True)
class RangeCouplingProxy:
    """
    Notebook's qualitative two-target range-FIM proxy.

    This is NOT a complete multi-target FIM.
    """

    delta_range_m: float
    range_resolution_m: float

    coupling_coefficient: float
    std_inflation_factor: float


def part_a_eq7_crlb(
    snr_db: float,
    *,
    parameters: PartAReferenceParameters
    = FROZEN_PART_A,
) -> PartACrlbBounds:
    """
    Reproduce the Part-A notebook Eq. (7).

    Frozen notebook convention:
        T  = T_chirp
        Tc = T_chirp
        M  = M_chirps
    """
    gamma = snr_db_to_linear(
        snr_db
    )

    c = parameters.c_mps
    B = parameters.bandwidth_hz

    T = parameters.chirp_duration_s
    T_c = parameters.chirp_duration_s

    M = parameters.m_chirps

    wavelength = parameters.wavelength_m

    range_variance = (
        (
            c * T
            / (2.0 * B)
        ) ** 2
        * 3.0
        / (
            8.0
            * math.pi ** 2
            * gamma
            * M
            * T_c ** 2
        )
    )

    velocity_variance = (
        (
            wavelength / 2.0
        ) ** 2
        * 3.0
        / (
            8.0
            * math.pi ** 2
            * gamma
            * T_c ** 2
            * M ** 3
        )
    )

    if (
        range_variance < 0.0
        or velocity_variance < 0.0
    ):
        raise RuntimeError(
            "CRLB variance became negative."
        )

    return PartACrlbBounds(
        snr_db=float(snr_db),
        snr_linear=gamma,

        range_variance_m2=(
            range_variance
        ),
        range_std_m=math.sqrt(
            range_variance
        ),

        radial_velocity_variance_m2ps2=(
            velocity_variance
        ),
        radial_velocity_std_mps=(
            math.sqrt(
                velocity_variance
            )
        ),
    )


def notebook_range_coupling_proxy(
    delta_range_m: float,
    *,
    parameters: PartAReferenceParameters
    = FROZEN_PART_A,
) -> RangeCouplingProxy:
    """
    Reproduce the Gaussian range-overlap proxy from
    the notebook:

        rho = exp(
            -0.5 * (delta_R / delta_R_res)^2
        )

    For F = F0 [[1,rho],[rho,1]], the standard-deviation
    inflation is sqrt(diag(F^-1) / diag(F0^-1))
    = 1 / sqrt(1-rho^2).
    """
    if (
        not math.isfinite(delta_range_m)
        or delta_range_m < 0.0
    ):
        raise ValueError(
            "Range separation must be finite "
            "and non-negative."
        )

    resolution = (
        parameters.range_resolution_m
    )

    rho = math.exp(
        -0.5
        * (
            delta_range_m
            / resolution
        ) ** 2
    )

    denominator = (
        1.0 - rho * rho
    )

    if denominator <= 0.0:
        raise ValueError(
            "Two-target range proxy is singular "
            "for zero/indistinguishable separation."
        )

    inflation = (
        1.0
        / math.sqrt(denominator)
    )

    return RangeCouplingProxy(
        delta_range_m=delta_range_m,
        range_resolution_m=resolution,
        coupling_coefficient=rho,
        std_inflation_factor=(
            inflation
        ),
    )


def coupled_range_std_m(
    *,
    single_target_range_std_m: float,
    coupling: RangeCouplingProxy,
) -> float:
    if (
        not math.isfinite(
            single_target_range_std_m
        )
        or single_target_range_std_m
        < 0.0
    ):
        raise ValueError(
            "Range standard deviation must be "
            "finite and non-negative."
        )

    return (
        single_target_range_std_m
        * coupling.std_inflation_factor
    )
