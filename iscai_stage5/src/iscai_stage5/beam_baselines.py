from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
    UniformAzimuthCodebook,
)

from iscai_stage5.beam_directional_gain import (
    normalized_gaussian_directional_gain,
)


FROZEN_BASELINE_NAMES = (
    "exhaustive_sweep",
    "previous_beam_persistence",
    "geometry_nearest",
    "fixed_top1_probability",
    "fixed_top3_probability",
    "fixed_top5_probability",
    "oracle_best_gain_eval_only",
)

CONTROLLER_BASELINE_NAMES = (
    "exhaustive_sweep",
    "previous_beam_persistence",
    "geometry_nearest",
    "fixed_top1_probability",
    "fixed_top3_probability",
    "fixed_top5_probability",
)

EVALUATOR_ONLY_BASELINE_NAMES = (
    "oracle_best_gain_eval_only",
)

FIXED_TOP_K_VALUES = (
    1,
    3,
    5,
)

FIXED_TOP_K_RANKING = (
    "descending_posterior_decision_cell_mass_P_b"
)

GEOMETRY_NEAREST_SEMANTICS = (
    "nearest_frozen_codebook_center_to_"
    "causal_predicted_receiver_azimuth"
)

PERSISTENCE_SEMANTICS = (
    "reuse_previous_selected_beam_if_available_"
    "else_geometry_nearest_causal_fallback"
)

ORACLE_SEMANTICS = (
    "evaluation_only_best_constructed_"
    "directional_gain_at_realized_receiver_azimuth"
)


@dataclass(
    frozen=True
)
class BeamSelection:
    policy_name: str

    beam_indices: tuple[
        int,
        ...
    ]

    ranking_source: str

    outside_support: bool = False

    evaluation_only: bool = False

    fallback_used: bool = False

    @property
    def probing_count(
        self,
    ) -> int:
        return len(
            self.beam_indices
        )


def _validate_codebook(
    codebook: UniformAzimuthCodebook,
):
    if (
        codebook.beam_count
        <=
        0
    ):
        raise ValueError(
            "Codebook must contain beams."
        )

    if len(
        codebook.cells
    ) != codebook.beam_count:
        raise ValueError(
            "Codebook beam-count/cell-count mismatch."
        )


def _finite_angle(
    value,
    *,
    name,
):
    angle = float(
        value
    )

    if not math.isfinite(
        angle
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return angle


def _stable_descending_indices(
    values,
):
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        array.ndim
        !=
        1
    ):
        raise ValueError(
            "Ranking values must be 1-D."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "Ranking values must be finite."
        )

    return tuple(
        sorted(
            range(
                array.size
            ),
            key=lambda index:
                (
                    -float(
                        array[
                            index
                        ]
                    ),
                    int(
                        index
                    ),
                ),
        )
    )


def exhaustive_sweep(
    *,
    codebook: UniformAzimuthCodebook,
) -> BeamSelection:
    """
    Probe every available beam.

    This is the maximum-overhead classical reference and
    contains no probabilistic ranking.
    """

    _validate_codebook(
        codebook
    )

    return BeamSelection(
        policy_name=(
            "exhaustive_sweep"
        ),

        beam_indices=tuple(
            range(
                codebook.beam_count
            )
        ),

        ranking_source=(
            "all_codebook_beams"
        ),
    )


def fixed_top_k_probability(
    *,
    probability: BeamProbabilityMass,
    codebook: UniformAzimuthCodebook,
    k: int,
) -> BeamSelection:
    """
    Fixed Top-K baseline using frozen posterior decision-cell
    probability mass P_b.

    Tie-breaking is deterministic:
        larger P_b first,
        then lower beam index.
    """

    _validate_codebook(
        codebook
    )

    k = int(
        k
    )

    if (
        k
        not in
        FIXED_TOP_K_VALUES
    ):
        raise ValueError(
            "Fixed Top-K baseline only supports "
            "K in {1,3,5}."
        )

    if (
        k
        >
        codebook.beam_count
    ):
        raise ValueError(
            "K exceeds codebook size."
        )

    masses = np.asarray(
        probability.masses,
        dtype=np.float64,
    )

    if masses.shape != (
        codebook.beam_count,
    ):
        raise ValueError(
            "Probability vector does not "
            "match codebook size."
        )

    if not np.all(
        np.isfinite(
            masses
        )
    ):
        raise ValueError(
            "Beam probabilities must be finite."
        )

    if np.any(
        masses
        <
        -1e-15
    ):
        raise ValueError(
            "Beam probabilities cannot be negative."
        )

    ranking = (
        _stable_descending_indices(
            masses
        )
    )

    return BeamSelection(
        policy_name=(
            f"fixed_top{k}_probability"
        ),

        beam_indices=tuple(
            ranking[
                :k
            ]
        ),

        ranking_source=(
            FIXED_TOP_K_RANKING
        ),

        outside_support=(
            float(
                probability
                .outside_support_mass
            )
            >
            0.0
        ),
    )


def geometry_nearest_beam(
    *,
    predicted_azimuth_rad,
    codebook: UniformAzimuthCodebook,
) -> BeamSelection:
    """
    Classical geometry baseline.

    Select the beam whose center is closest to the causal
    predicted receiver azimuth.

    The frozen support is narrow and does not cross +/-pi,
    so ordinary angular difference is used.

    If the predicted angle is outside codebook support, the
    nearest edge beam is returned and outside_support=True.
    """

    _validate_codebook(
        codebook
    )

    angle = _finite_angle(
        predicted_azimuth_rad,
        name="predicted_azimuth_rad",
    )

    outside = (
        angle
        <
        codebook.support_min_azimuth_rad
        or
        angle
        >
        codebook.support_max_azimuth_rad
    )

    best = min(
        codebook.cells,
        key=lambda cell:
            (
                abs(
                    angle
                    -
                    float(
                        cell.center_azimuth_rad
                    )
                ),
                int(
                    cell.index
                ),
            ),
    )

    return BeamSelection(
        policy_name=(
            "geometry_nearest"
        ),

        beam_indices=(
            int(
                best.index
            ),
        ),

        ranking_source=(
            GEOMETRY_NEAREST_SEMANTICS
        ),

        outside_support=(
            bool(
                outside
            )
        ),
    )


def previous_beam_persistence(
    *,
    previous_beam_index,
    codebook: UniformAzimuthCodebook,
    fallback_predicted_azimuth_rad=None,
) -> BeamSelection:
    """
    Causal previous-beam persistence baseline.

    If a valid previous beam exists:
        reuse it exactly.

    At initialization / loss of previous state:
        use causal geometry-nearest fallback.

    No future receiver position or oracle label enters this
    controller baseline.
    """

    _validate_codebook(
        codebook
    )

    if previous_beam_index is not None:

        index = int(
            previous_beam_index
        )

        if (
            index
            <
            0
            or
            index
            >=
            codebook.beam_count
        ):
            raise ValueError(
                "previous_beam_index is outside "
                "the frozen codebook."
            )

        return BeamSelection(
            policy_name=(
                "previous_beam_persistence"
            ),

            beam_indices=(
                index,
            ),

            ranking_source=(
                PERSISTENCE_SEMANTICS
            ),

            fallback_used=(
                False
            ),
        )

    if fallback_predicted_azimuth_rad is None:
        raise ValueError(
            "Persistence initialization requires "
            "fallback_predicted_azimuth_rad."
        )

    fallback = geometry_nearest_beam(
        predicted_azimuth_rad=(
            fallback_predicted_azimuth_rad
        ),

        codebook=(
            codebook
        ),
    )

    return BeamSelection(
        policy_name=(
            "previous_beam_persistence"
        ),

        beam_indices=(
            fallback.beam_indices
        ),

        ranking_source=(
            PERSISTENCE_SEMANTICS
        ),

        outside_support=(
            fallback.outside_support
        ),

        fallback_used=(
            True
        ),
    )


def oracle_best_gain_beam(
    *,
    realized_azimuth_rad,
    codebook: UniformAzimuthCodebook,
) -> BeamSelection:
    """
    Evaluation-only oracle.

    It chooses the available codebook beam with maximum
    constructed Block5.3 directional score at the realized
    receiver azimuth.

    This is:
        - evaluator-only,
        - geometry-derived,
        - NOT a real WOMD communication label,
        - NOT available to the controller.

    The oracle remains defined outside support because the
    physical score surrogate has non-zero tails; the result
    is explicitly flagged outside_support=True.
    """

    _validate_codebook(
        codebook
    )

    angle = _finite_angle(
        realized_azimuth_rad,
        name="realized_azimuth_rad",
    )

    outside = (
        angle
        <
        codebook.support_min_azimuth_rad
        or
        angle
        >
        codebook.support_max_azimuth_rad
    )

    gains = []

    values = np.asarray(
        [
            angle
        ],
        dtype=np.float64,
    )

    for cell in codebook.cells:

        gain = float(
            normalized_gaussian_directional_gain(
                cell,
                values,
            )[
                0
            ]
        )

        gains.append(
            gain
        )

    ranking = (
        _stable_descending_indices(
            gains
        )
    )

    return BeamSelection(
        policy_name=(
            "oracle_best_gain_eval_only"
        ),

        beam_indices=(
            int(
                ranking[
                    0
                ]
            ),
        ),

        ranking_source=(
            ORACLE_SEMANTICS
        ),

        outside_support=(
            bool(
                outside
            )
        ),

        evaluation_only=(
            True
        ),
    )


def frozen_baseline_registry():
    return {
        "controller_baselines":
            CONTROLLER_BASELINE_NAMES,

        "evaluation_only_baselines":
            EVALUATOR_ONLY_BASELINE_NAMES,

        "fixed_top_k_values":
            FIXED_TOP_K_VALUES,

        "fixed_top_k_ranking":
            FIXED_TOP_K_RANKING,

        "geometry_nearest":
            GEOMETRY_NEAREST_SEMANTICS,

        "persistence":
            PERSISTENCE_SEMANTICS,

        "oracle":
            ORACLE_SEMANTICS,

        "adaptive_TopK_included":
            False,

        "blockage_aware_included":
            False,
    }
