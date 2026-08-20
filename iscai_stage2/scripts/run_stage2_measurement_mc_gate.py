from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from iscai_stage2.validation.measurement_mc import (
    assert_monte_carlo_consistent,
    run_measurement_monte_carlo,
)


REPORT = Path(
    "/home/agni/waymo/iscai_stage2/"
    "reports/block3e_measurement_mc_gate.json"
)


SNR_VALUES_DB = (
    0.0,
    10.0,
    20.0,
)

SAMPLES_PER_SNR = 30_000

AZIMUTH_STD_RAD = math.radians(
    1.0
)

ELEVATION_STD_RAD = math.radians(
    1.0
)


reports = []


for index, snr_db in enumerate(
    SNR_VALUES_DB
):
    report = (
        run_measurement_monte_carlo(
            samples=SAMPLES_PER_SNR,

            sensing_snr_db=snr_db,

            azimuth_std_rad=(
                AZIMUTH_STD_RAD
            ),
            elevation_std_rad=(
                ELEVATION_STD_RAD
            ),

            seed_base=(
                20260810
                + index * 1_000_000
            ),
        )
    )

    assert_monte_carlo_consistent(
        report,
        normalized_mean_tolerance=0.03,
        normalized_rmse_tolerance=0.03,
        normalized_std_tolerance=0.03,
    )

    reports.append(
        report
    )


# ------------------------------------------------------------
# CRLB scaling gate
# std ∝ 1 / sqrt(SNR_linear)
# therefore every +10 dB -> std / sqrt(10)
# ------------------------------------------------------------

for lower, higher in zip(
    reports[:-1],
    reports[1:],
):
    expected_ratio = math.sqrt(
        10.0
    )

    actual_range_ratio = (
        lower.range_stats.configured_std
        /
        higher.range_stats.configured_std
    )

    actual_vr_ratio = (
        lower.radial_velocity_stats
        .configured_std
        /
        higher.radial_velocity_stats
        .configured_std
    )

    if not math.isclose(
        actual_range_ratio,
        expected_ratio,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise RuntimeError(
            "Range CRLB SNR scaling gate failed."
        )

    if not math.isclose(
        actual_vr_ratio,
        expected_ratio,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise RuntimeError(
            "Velocity CRLB SNR scaling gate failed."
        )


payload = {
    "status": "PASS",

    "block": (
        "stage2_block3e_measurement_mc_gate"
    ),

    "samples_per_snr": (
        SAMPLES_PER_SNR
    ),

    "snr_values_db": list(
        SNR_VALUES_DB
    ),

    "measurement_order": [
        "range",
        "radial_velocity",
        "azimuth",
        "elevation",
    ],

    "range_velocity_covariance_source": (
        "frozen_part_a_eq7_crlb"
    ),

    "angular_covariance_source": (
        "explicit_stage2_angular_assumption"
    ),

    "gaussian_corruption": (
        "sha256_box_muller_v1"
    ),

    "acceptance": {
        "normalized_mean_abs_max":
            0.03,

        "normalized_rmse_error_max":
            0.03,

        "normalized_std_error_max":
            0.03,
    },

    "reports": [
        asdict(report)
        for report in reports
    ],

    "crlb_snr_scaling_verified":
        True,

    "measured_fmcw": False,
}


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== Stage2 measurement Monte-Carlo gate ====="
)

print(
    "samples per SNR      =",
    SAMPLES_PER_SNR,
)

for report in reports:
    print()
    print(
        "SNR [dB]             =",
        report.sensing_snr_db,
    )

    for stats in (
        report.range_stats,
        report.radial_velocity_stats,
        report.azimuth_stats,
        report.elevation_stats,
    ):
        print(
            f"{stats.component:18s} "
            f"sigma={stats.configured_std:.9g} "
            f"mean/sigma={stats.normalized_mean:+.4f} "
            f"RMSE/sigma={stats.normalized_rmse:.4f} "
            f"std/sigma={stats.normalized_std:.4f}"
        )

print()
print(
    "CRLB SNR scaling     = PASS"
)

print(
    "measured FMCW        = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)
