from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
    UniformAzimuthCodebook,
)


FROZEN_COVERAGE_TARGETS = (
    0.90,
    0.95,
    0.975,
    0.99,
)

FROZEN_NOMINAL_COVERAGE = (
    0.95
)

ADAPTIVE_SELECTION_RULE = (
    "smallest_descending_P_b_prefix_"
    "whose_cumulative_mass_reaches_"
    "requested_coverage"
)

ADAPTIVE_TIE_BREAK = (
    "lower_beam_index"
)

OUTSIDE_SUPPORT_POLICY = (
    "outside_support_mass_is_explicit_"
    "and_cannot_be_claimed_as_covered_"
    "by_in_support_beams"
)

KMAX_POLICY_STATUS = (
    "SUPPORTED_NOT_NUMERICALLY_FROZEN_IN_PART1"
)


@dataclass(
    frozen=True
)
class AdaptiveTopKSelection:
    beam_indices: tuple[
        int,
        ...
    ]

    requested_coverage: float

    selected_in_support_mass: float

    inside_support_mass: float

    outside_support_mass: float

    achieved_requested_coverage: bool

    k: int

    kmax: int | None

    kmax_limited: bool

    unattainable_due_to_outside_support: bool

    ranking_source: str = (
        "descending_P_b"
    )

    tie_break: str = (
        ADAPTIVE_TIE_BREAK
    )


def _validate_probability(
    probability: BeamProbabilityMass,
    codebook: UniformAzimuthCodebook,
):
    masses = np.asarray(
        probability.masses,
        dtype=np.float64,
    )

    if masses.shape != (
        codebook.beam_count,
    ):
        raise ValueError(
            "Probability vector does not "
            "match frozen codebook size."
        )

    if not np.all(
        np.isfinite(
            masses
        )
    ):
        raise ValueError(
            "Beam probability masses "
            "must be finite."
        )

    if np.any(
        masses
        <
        -1e-15
    ):
        raise ValueError(
            "Beam probability masses "
            "cannot be negative."
        )

    outside = float(
        probability
        .outside_support_mass
    )

    inside = float(
        probability
        .inside_support_mass
    )

    if (
        not math.isfinite(
            inside
        )
        or
        not math.isfinite(
            outside
        )
    ):
        raise ValueError(
            "Support probability masses "
            "must be finite."
        )

    if (
        inside
        <
        -1e-12
        or
        outside
        <
        -1e-12
    ):
        raise ValueError(
            "Support probability masses "
            "cannot be negative."
        )

    if abs(
        (
            float(
                np.sum(
                    masses
                )
            )
            +
            outside
        )
        -
        1.0
    ) > 1e-9:
        raise ValueError(
            "P_b plus outside-support "
            "mass must sum to one."
        )

    if abs(
        float(
            np.sum(
                masses
            )
        )
        -
        inside
    ) > 1e-9:
        raise ValueError(
            "inside_support_mass does not "
            "equal sum(P_b)."
        )

    return masses


def _validate_coverage(
    coverage,
):
    value = float(
        coverage
    )

    if (
        not math.isfinite(
            value
        )
        or
        value
        <=
        0.0
        or
        value
        >
        1.0
    ):
        raise ValueError(
            "requested_coverage must "
            "be in (0,1]."
        )

    return value


def _validate_kmax(
    kmax,
    *,
    beam_count,
):
    if kmax is None:
        return None

    value = int(
        kmax
    )

    if (
        value
        <=
        0
        or
        value
        >
        int(
            beam_count
        )
    ):
        raise ValueError(
            "kmax must lie in "
            "[1, beam_count]."
        )

    return value


def descending_probability_ranking(
    *,
    probability: BeamProbabilityMass,
    codebook: UniformAzimuthCodebook,
):
    """
    Deterministic ranking:

      1. larger P_b first
      2. lower beam index for ties
    """

    masses = _validate_probability(
        probability,
        codebook,
    )

    return tuple(
        sorted(
            range(
                codebook.beam_count
            ),
            key=lambda index:
                (
                    -float(
                        masses[
                            index
                        ]
                    ),
                    int(
                        index
                    ),
                ),
        )
    )


def adaptive_topk_probability_mass(
    *,
    probability: BeamProbabilityMass,
    codebook: UniformAzimuthCodebook,
    requested_coverage: float,
    kmax: int | None = None,
) -> AdaptiveTopKSelection:
    """
    Frozen Block5.5 core adaptive Top-K rule.

    Select the smallest prefix of beams ranked by descending
    P_b whose cumulative in-support posterior mass reaches
    requested_coverage.

    Important:
      outside-support mass is not silently renormalized away.

    Therefore if:

        inside_support_mass < requested_coverage

    no in-support beam set can honestly claim requested
    coverage. In that case all available beams up to Kmax
    are returned and achieved_requested_coverage=False.

    Kmax is supported here structurally, but its numerical
    value is not frozen in Block5.5 Part1.
    """

    coverage = _validate_coverage(
        requested_coverage
    )

    masses = _validate_probability(
        probability,
        codebook,
    )

    maximum_k = _validate_kmax(
        kmax,
        beam_count=(
            codebook.beam_count
        ),
    )

    effective_limit = (
        codebook.beam_count
        if maximum_k is None
        else maximum_k
    )

    ranking = (
        descending_probability_ranking(
            probability=(
                probability
            ),

            codebook=(
                codebook
            ),
        )
    )

    selected = []

    cumulative = 0.0

    achieved = False

    for index in ranking[
        :effective_limit
    ]:

        selected.append(
            int(
                index
            )
        )

        cumulative += float(
            masses[
                index
            ]
        )

        if (
            cumulative
            +
            1e-15
            >=
            coverage
        ):
            achieved = True
            break

    inside_mass = float(
        probability
        .inside_support_mass
    )

    outside_mass = float(
        probability
        .outside_support_mass
    )

    unattainable_due_to_outside = (
        inside_mass
        +
        1e-15
        <
        coverage
    )

    if unattainable_due_to_outside:
        achieved = False

    kmax_limited = (
        maximum_k is not None
        and
        len(
            selected
        )
        >=
        maximum_k
        and
        not achieved
    )

    return AdaptiveTopKSelection(
        beam_indices=tuple(
            selected
        ),

        requested_coverage=(
            coverage
        ),

        selected_in_support_mass=(
            float(
                cumulative
            )
        ),

        inside_support_mass=(
            inside_mass
        ),

        outside_support_mass=(
            outside_mass
        ),

        achieved_requested_coverage=(
            bool(
                achieved
            )
        ),

        k=len(
            selected
        ),

        kmax=(
            maximum_k
        ),

        kmax_limited=(
            bool(
                kmax_limited
            )
        ),

        unattainable_due_to_outside_support=(
            bool(
                unattainable_due_to_outside
            )
        ),
    )


def nominal_adaptive_topk(
    *,
    probability: BeamProbabilityMass,
    codebook: UniformAzimuthCodebook,
    kmax: int | None = None,
):
    return adaptive_topk_probability_mass(
        probability=(
            probability
        ),

        codebook=(
            codebook
        ),

        requested_coverage=(
            FROZEN_NOMINAL_COVERAGE
        ),

        kmax=(
            kmax
        ),
    )


def all_frozen_coverage_targets(
    *,
    probability: BeamProbabilityMass,
    codebook: UniformAzimuthCodebook,
    kmax: int | None = None,
):
    return {
        coverage:
            adaptive_topk_probability_mass(
                probability=(
                    probability
                ),

                codebook=(
                    codebook
                ),

                requested_coverage=(
                    coverage
                ),

                kmax=(
                    kmax
                ),
            )
        for coverage in (
            FROZEN_COVERAGE_TARGETS
        )
    }
