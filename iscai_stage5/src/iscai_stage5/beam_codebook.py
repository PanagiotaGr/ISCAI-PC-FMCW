from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


SUPPORTED_BEAM_COUNTS = (
    16,
    32,
    64,
)

DECISION_CELL_SEMANTICS = (
    "nonoverlapping_probability_accounting_cells"
)

PHYSICAL_GAIN_SEMANTICS = (
    "separate_physical_gain_patterns_may_overlap"
)

BEAM_PROBABILITY_SEMANTICS = (
    "posterior_probability_mass_inside_decision_cell"
)

EXPECTED_GAIN_SCORE_SEMANTICS = (
    "posterior_expectation_of_physical_beam_gain"
)


@dataclass(
    frozen=True
)
class BeamDecisionCell:
    index: int

    center_azimuth_rad: float

    lower_azimuth_rad: float

    upper_azimuth_rad: float


@dataclass(
    frozen=True
)
class UniformAzimuthCodebook:
    beam_count: int

    support_min_azimuth_rad: float

    support_max_azimuth_rad: float

    decision_width_rad: float

    cells: tuple[
        BeamDecisionCell,
        ...
    ]

    decision_cell_semantics: str = (
        DECISION_CELL_SEMANTICS
    )

    physical_gain_semantics: str = (
        PHYSICAL_GAIN_SEMANTICS
    )


@dataclass(
    frozen=True
)
class BeamProbabilityMass:
    masses: tuple[
        float,
        ...
    ]

    inside_support_mass: float

    outside_support_mass: float

    sample_count: int

    semantics: str = (
        BEAM_PROBABILITY_SEMANTICS
    )


@dataclass(
    frozen=True
)
class ExpectedBeamGainScores:
    scores: tuple[
        float,
        ...
    ]

    sample_count: int

    semantics: str = (
        EXPECTED_GAIN_SCORE_SEMANTICS
    )


def _finite_scalar(
    value,
    *,
    name: str,
) -> float:
    result = float(
        value
    )

    if not math.isfinite(
        result
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def build_uniform_azimuth_codebook(
    *,
    beam_count: int,
    support_min_azimuth_rad: float,
    support_max_azimuth_rad: float,
) -> UniformAzimuthCodebook:
    """
    Construct non-overlapping azimuth decision cells.

    Important:
      these are probability-accounting cells.

      They are NOT yet the frozen optical physical
      beam pattern / half-power width.

    Physical gain patterns are a separate object and may
    overlap across neighbouring beams.
    """

    beam_count = int(
        beam_count
    )

    if (
        beam_count
        not in
        SUPPORTED_BEAM_COUNTS
    ):
        raise ValueError(
            "beam_count must be one of "
            "16, 32 or 64."
        )

    minimum = _finite_scalar(
        support_min_azimuth_rad,
        name="support_min_azimuth_rad",
    )

    maximum = _finite_scalar(
        support_max_azimuth_rad,
        name="support_max_azimuth_rad",
    )

    if maximum <= minimum:
        raise ValueError(
            "Azimuth support must have "
            "positive width."
        )

    support_width = (
        maximum
        -
        minimum
    )

    if (
        support_width
        >
        (
            2.0
            *
            math.pi
            +
            1e-12
        )
    ):
        raise ValueError(
            "Azimuth support cannot exceed 2*pi."
        )

    width = (
        support_width
        /
        float(
            beam_count
        )
    )

    cells = []

    for index in range(
        beam_count
    ):
        lower = (
            minimum
            +
            float(
                index
            )
            *
            width
        )

        upper = (
            minimum
            +
            float(
                index
                +
                1
            )
            *
            width
        )

        center = (
            0.5
            *
            (
                lower
                +
                upper
            )
        )

        cells.append(
            BeamDecisionCell(
                index=(
                    index
                ),

                center_azimuth_rad=(
                    float(
                        center
                    )
                ),

                lower_azimuth_rad=(
                    float(
                        lower
                    )
                ),

                upper_azimuth_rad=(
                    float(
                        upper
                    )
                ),
            )
        )

    return UniformAzimuthCodebook(
        beam_count=(
            beam_count
        ),

        support_min_azimuth_rad=(
            minimum
        ),

        support_max_azimuth_rad=(
            maximum
        ),

        decision_width_rad=(
            float(
                width
            )
        ),

        cells=tuple(
            cells
        ),
    )


def beam_probability_mass_from_samples(
    *,
    azimuth_samples_rad,
    codebook: UniformAzimuthCodebook,
) -> BeamProbabilityMass:
    """
    Monte-Carlo estimate of:

        P_b = integral_{Theta_b} p(theta_RX) d theta

    Samples outside the declared codebook support are never
    silently discarded. Their probability mass is reported
    explicitly as outside_support_mass.

    Boundary convention:
      [lower, upper) for every beam except the final beam,
      which also contains support_max exactly.
    """

    samples = np.asarray(
        azimuth_samples_rad,
        dtype=np.float64,
    )

    if (
        samples.ndim != 1
        or
        samples.size <= 0
    ):
        raise ValueError(
            "azimuth_samples_rad must be "
            "a non-empty 1-D array."
        )

    if not np.all(
        np.isfinite(
            samples
        )
    ):
        raise ValueError(
            "Azimuth samples must be finite."
        )

    minimum = (
        codebook
        .support_min_azimuth_rad
    )

    maximum = (
        codebook
        .support_max_azimuth_rad
    )

    width = (
        codebook
        .decision_width_rad
    )

    inside = (
        (
            samples
            >=
            minimum
        )
        &
        (
            samples
            <=
            maximum
        )
    )

    inside_samples = samples[
        inside
    ]

    counts = np.zeros(
        (
            codebook.beam_count,
        ),
        dtype=np.int64,
    )

    if inside_samples.size:

        raw_indices = np.floor(
            (
                inside_samples
                -
                minimum
            )
            /
            width
        ).astype(
            np.int64
        )

        raw_indices = np.clip(
            raw_indices,
            0,
            codebook.beam_count
            -
            1,
        )

        counts += np.bincount(
            raw_indices,
            minlength=(
                codebook.beam_count
            ),
        )[
            :codebook.beam_count
        ]

    count = int(
        samples.size
    )

    masses = (
        counts.astype(
            np.float64
        )
        /
        float(
            count
        )
    )

    inside_mass = float(
        np.sum(
            masses
        )
    )

    outside_mass = (
        1.0
        -
        inside_mass
    )

    if abs(
        outside_mass
    ) < 1e-15:
        outside_mass = 0.0

    if (
        inside_mass
        <
        -1e-12
        or
        inside_mass
        >
        1.0
        +
        1e-12
    ):
        raise RuntimeError(
            "Invalid inside-support "
            "probability mass."
        )

    if (
        outside_mass
        <
        -1e-12
        or
        outside_mass
        >
        1.0
        +
        1e-12
    ):
        raise RuntimeError(
            "Invalid outside-support "
            "probability mass."
        )

    return BeamProbabilityMass(
        masses=tuple(
            float(
                value
            )
            for value in masses
        ),

        inside_support_mass=(
            float(
                inside_mass
            )
        ),

        outside_support_mass=(
            float(
                max(
                    0.0,
                    outside_mass
                )
            )
        ),

        sample_count=(
            count
        ),
    )


def expected_gain_scores_from_samples(
    *,
    azimuth_samples_rad,
    codebook: UniformAzimuthCodebook,
    gain_evaluator: Callable[
        [
            BeamDecisionCell,
            np.ndarray,
        ],
        np.ndarray,
    ],
) -> ExpectedBeamGainScores:
    """
    Monte-Carlo estimate of:

        S_b = E_p [ G_b(theta_RX) ]

    This is deliberately distinct from P_b.

    The supplied physical gain evaluator may assign
    non-zero gain to angles outside a beam's probability
    decision cell, allowing overlapping physical beams.
    """

    samples = np.asarray(
        azimuth_samples_rad,
        dtype=np.float64,
    )

    if (
        samples.ndim != 1
        or
        samples.size <= 0
    ):
        raise ValueError(
            "azimuth_samples_rad must be "
            "a non-empty 1-D array."
        )

    if not np.all(
        np.isfinite(
            samples
        )
    ):
        raise ValueError(
            "Azimuth samples must be finite."
        )

    scores = []

    for cell in codebook.cells:

        values = np.asarray(
            gain_evaluator(
                cell,
                samples,
            ),
            dtype=np.float64,
        )

        if values.shape != samples.shape:
            raise ValueError(
                "gain_evaluator must return "
                "one gain value per sample."
            )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise ValueError(
                "Beam gain values must be finite."
            )

        if np.any(
            values
            <
            0.0
        ):
            raise ValueError(
                "Beam gain values cannot be negative."
            )

        scores.append(
            float(
                np.mean(
                    values
                )
            )
        )

    return ExpectedBeamGainScores(
        scores=tuple(
            scores
        ),

        sample_count=int(
            samples.size
        ),
    )



# ============================================================
# BLOCK53_COMPLETE_DECISION_PARTITION_V1
#
# Structural repair:
# The primary probability representation is a complete
# non-overlapping decision partition. The bounded physical
# beam sector is retained for optical gain evaluation.
#
# Equivalently, the first/last decision cells extend outward.
# No posterior mass is discarded or renormalized.
# ============================================================

def complete_partition_azimuth_samples(
    *,
    azimuth_samples_rad,
    codebook,
):
    import numpy as _np

    values = _np.asarray(
        azimuth_samples_rad,
        dtype=_np.float64,
    )

    if not _np.all(_np.isfinite(values)):
        raise ValueError(
            "Azimuth samples must be finite."
        )

    lo = float(
        codebook.support_min_azimuth_rad
    )
    hi = float(
        codebook.support_max_azimuth_rad
    )

    if not lo < hi:
        raise ValueError(
            "Invalid codebook support."
        )

    # Edge-cell extension:
    # samples are assigned to the nearest boundary decision
    # region, while their original angles remain untouched for
    # physical gain/SNR/BER evaluation.
    return _np.clip(
        values,
        lo,
        hi,
    )


def complete_partition_beam_probability_mass_from_samples(
    *,
    azimuth_samples_rad,
    codebook,
):
    clipped = complete_partition_azimuth_samples(
        azimuth_samples_rad=azimuth_samples_rad,
        codebook=codebook,
    )

    result = beam_probability_mass_from_samples(
        azimuth_samples_rad=clipped,
        codebook=codebook,
    )

    if abs(
        float(result.outside_support_mass)
    ) > 1e-12:
        raise RuntimeError(
            "Complete probability partition produced "
            "non-zero outside-support mass."
        )

    if abs(
        float(result.inside_support_mass) - 1.0
    ) > 1e-12:
        raise RuntimeError(
            "Complete probability partition does not "
            "sum to one."
        )

    return result


def complete_partition_decision_azimuth(
    *,
    azimuth_rad,
    codebook,
):
    import math as _math

    value = float(azimuth_rad)

    if not _math.isfinite(value):
        raise ValueError(
            "Decision azimuth must be finite."
        )

    return min(
        max(
            value,
            float(
                codebook.support_min_azimuth_rad
            ),
        ),
        float(
            codebook.support_max_azimuth_rad
        ),
    )
