from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
    clean_crlb_conditioned_observation,
)
from iscai_stage2.observations.gaussian_corruption import (
    GaussianCorruptionConfig,
    gaussian_corrupt_clean_record,
)
from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)


COMPONENTS = (
    "range",
    "radial_velocity",
    "azimuth",
    "elevation",
)


@dataclass(frozen=True)
class ComponentMonteCarloStats:
    component: str

    configured_std: float

    empirical_mean_error: float
    empirical_rmse: float
    empirical_std: float

    normalized_mean: float
    normalized_rmse: float
    normalized_std: float

    rmse_ratio_to_configured_std: float


@dataclass(frozen=True)
class MeasurementMonteCarloReport:
    samples: int
    sensing_snr_db: float

    range_stats: ComponentMonteCarloStats
    radial_velocity_stats: ComponentMonteCarloStats
    azimuth_stats: ComponentMonteCarloStats
    elevation_stats: ComponentMonteCarloStats


def _stats(
    *,
    component: str,
    errors: list[float],
    configured_std: float,
) -> ComponentMonteCarloStats:
    if not errors:
        raise ValueError(
            "Monte-Carlo error vector cannot be empty."
        )

    if (
        not math.isfinite(configured_std)
        or configured_std <= 0.0
    ):
        raise ValueError(
            "Configured standard deviation must "
            "be finite and positive."
        )

    n = len(errors)

    mean_error = (
        sum(errors) / n
    )

    rmse = math.sqrt(
        sum(
            value * value
            for value in errors
        ) / n
    )

    variance = (
        sum(
            (
                value - mean_error
            ) ** 2
            for value in errors
        ) / n
    )

    empirical_std = math.sqrt(
        variance
    )

    normalized = [
        value / configured_std
        for value in errors
    ]

    normalized_mean = (
        sum(normalized) / n
    )

    normalized_rmse = math.sqrt(
        sum(
            value * value
            for value in normalized
        ) / n
    )

    normalized_variance = (
        sum(
            (
                value - normalized_mean
            ) ** 2
            for value in normalized
        ) / n
    )

    normalized_std = math.sqrt(
        normalized_variance
    )

    return ComponentMonteCarloStats(
        component=component,

        configured_std=configured_std,

        empirical_mean_error=(
            mean_error
        ),
        empirical_rmse=rmse,
        empirical_std=empirical_std,

        normalized_mean=(
            normalized_mean
        ),
        normalized_rmse=(
            normalized_rmse
        ),
        normalized_std=(
            normalized_std
        ),

        rmse_ratio_to_configured_std=(
            rmse / configured_std
        ),
    )


def run_measurement_monte_carlo(
    *,
    samples: int,

    sensing_snr_db: float,

    azimuth_std_rad: float,
    elevation_std_rad: float,

    seed_base: int = 100_000,
) -> MeasurementMonteCarloReport:
    """
    Validate the actual Stage-2 corruption pipeline:

        ideal
          -> clean CRLB-conditioned observation
          -> Gaussian corruption

    The ideal target is deliberately placed away from angular
    boundaries so wrapping/clipping does not alter Gaussian
    statistics.

    Each sample gets a distinct deterministic track ID so the
    SHA256 noise generator produces independent-like draws while
    remaining exactly reproducible.
    """

    if samples <= 0:
        raise ValueError(
            "samples must be positive."
        )

    ideal = IdealCausalObservable(
        time_index=5,
        timestamp_s=0.5,

        actor_position_Ht_m=(
            30.0,
            0.0,
            0.0,
        ),

        range_m=30.0,
        azimuth_rad=0.0,
        elevation_rad=0.0,

        radial_velocity_mps=-5.0,

        geometry_valid=True,
        radial_velocity_valid=True,
    )

    clean_config = (
        CleanObservationConfig(
            sensing_snr_db=(
                sensing_snr_db
            ),
            azimuth_std_rad=(
                azimuth_std_rad
            ),
            elevation_std_rad=(
                elevation_std_rad
            ),
        )
    )

    range_errors: list[float] = []
    vr_errors: list[float] = []
    azimuth_errors: list[float] = []
    elevation_errors: list[float] = []

    configured_stds = None

    for sample_index in range(
        samples
    ):
        track_id = (
            f"mc-{sample_index}"
        )

        clean = (
            clean_crlb_conditioned_observation(
                scenario_id="mc-scenario",
                track_id=track_id,
                object_class="TYPE_VEHICLE",

                ideal=ideal,
                config=clean_config,
            )
        )

        if (
            not clean.measurement_valid
            or clean.measurement is None
        ):
            raise RuntimeError(
                "Monte-Carlo clean measurement "
                "unexpectedly invalid."
            )

        noisy = (
            gaussian_corrupt_clean_record(
                scenario_id="mc-scenario",
                track_id=track_id,
                object_class="TYPE_VEHICLE",

                clean=clean,

                config=(
                    GaussianCorruptionConfig(
                        seed=(
                            seed_base
                            + sample_index
                        )
                    )
                ),
            )
        )

        if (
            not noisy.measurement_valid
            or noisy.measurement is None
        ):
            raise RuntimeError(
                "Monte-Carlo Gaussian measurement "
                "unexpectedly invalid."
            )

        measurement = noisy.measurement

        range_errors.append(
            measurement.range_m
            - ideal.range_m
        )

        vr_errors.append(
            measurement.radial_velocity_mps
            - ideal.radial_velocity_mps
        )

        azimuth_errors.append(
            measurement.azimuth_rad
            - ideal.azimuth_rad
        )

        elevation_errors.append(
            measurement.elevation_rad
            - ideal.elevation_rad
        )

        if configured_stds is None:
            covariance = (
                measurement.covariance.matrix
            )

            configured_stds = (
                math.sqrt(
                    covariance[0][0]
                ),
                math.sqrt(
                    covariance[1][1]
                ),
                math.sqrt(
                    covariance[2][2]
                ),
                math.sqrt(
                    covariance[3][3]
                ),
            )

    if configured_stds is None:
        raise RuntimeError(
            "Monte-Carlo covariance unavailable."
        )

    range_std, vr_std, az_std, el_std = (
        configured_stds
    )

    return MeasurementMonteCarloReport(
        samples=samples,
        sensing_snr_db=(
            sensing_snr_db
        ),

        range_stats=_stats(
            component="range",
            errors=range_errors,
            configured_std=range_std,
        ),

        radial_velocity_stats=_stats(
            component="radial_velocity",
            errors=vr_errors,
            configured_std=vr_std,
        ),

        azimuth_stats=_stats(
            component="azimuth",
            errors=azimuth_errors,
            configured_std=az_std,
        ),

        elevation_stats=_stats(
            component="elevation",
            errors=elevation_errors,
            configured_std=el_std,
        ),
    )


def assert_monte_carlo_consistent(
    report: MeasurementMonteCarloReport,
    *,
    normalized_mean_tolerance: float = 0.04,
    normalized_rmse_tolerance: float = 0.04,
    normalized_std_tolerance: float = 0.04,
) -> None:
    """
    Gate against configured Gaussian covariance.

    For correctly generated zero-mean Gaussian residuals:
        normalized mean  ~ 0
        normalized RMSE  ~ 1
        normalized std   ~ 1
    """

    stats = (
        report.range_stats,
        report.radial_velocity_stats,
        report.azimuth_stats,
        report.elevation_stats,
    )

    for item in stats:
        if abs(
            item.normalized_mean
        ) > normalized_mean_tolerance:
            raise AssertionError(
                f"{item.component}: normalized mean "
                f"{item.normalized_mean} exceeds tolerance."
            )

        if abs(
            item.normalized_rmse - 1.0
        ) > normalized_rmse_tolerance:
            raise AssertionError(
                f"{item.component}: normalized RMSE "
                f"{item.normalized_rmse} inconsistent "
                "with configured covariance."
            )

        if abs(
            item.normalized_std - 1.0
        ) > normalized_std_tolerance:
            raise AssertionError(
                f"{item.component}: normalized std "
                f"{item.normalized_std} inconsistent "
                "with configured covariance."
            )
