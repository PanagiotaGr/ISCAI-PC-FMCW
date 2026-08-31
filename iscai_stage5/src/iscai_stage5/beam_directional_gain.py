from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from iscai_stage5.beam_codebook import (
    BeamDecisionCell,
    BeamProbabilityMass,
    ExpectedBeamGainScores,
    UniformAzimuthCodebook,
    beam_probability_mass_from_samples,
    build_uniform_azimuth_codebook,
    expected_gain_scores_from_samples,
)


FROZEN_SUPPORT_HALF_ANGLE_DEG = (
    12.0
)

FROZEN_SUPPORT_MIN_DEG = (
    -12.0
)

FROZEN_SUPPORT_MAX_DEG = (
    12.0
)

FROZEN_SUPPORT_MIN_RAD = (
    math.radians(
        FROZEN_SUPPORT_MIN_DEG
    )
)

FROZEN_SUPPORT_MAX_RAD = (
    math.radians(
        FROZEN_SUPPORT_MAX_DEG
    )
)

FROZEN_SUPPORT_PROVENANCE = (
    "constructed_Stage5_reuse_of_"
    "PartA_explicit_visualization_"
    "headlamp_half_FOV_assumption"
)

FROZEN_SUPPORT_IS_MEASURED = (
    False
)

FROZEN_SUPPORT_IS_PAPER_GIVEN = (
    False
)

FROZEN_GAIN_MODEL = (
    "constructed_normalized_Gaussian_"
    "main_lobe_HPBW_equals_"
    "decision_cell_width"
)

FROZEN_GAIN_ROLE = (
    "codebook_expected_gain_scoring_only"
)

FROZEN_GAIN_IS_PARTA_OPTICAL_LINK_GAIN = (
    False
)

FROZEN_GAIN_IS_MEASURED = (
    False
)

FROZEN_HPBW_RULE = (
    "HPBW_equals_uniform_decision_cell_width"
)


@dataclass(
    frozen=True
)
class FrozenBeamScores:
    probability: BeamProbabilityMass

    expected_gain: ExpectedBeamGainScores

    beam_count: int

    support_min_deg: float

    support_max_deg: float

    decision_width_deg: float

    gain_model: str


def build_frozen_directional_codebook(
    beam_count: int,
) -> UniformAzimuthCodebook:
    """
    Stage5 frozen azimuth codebook.

    Support:
        [-12 deg, +12 deg].

    Provenance:
        constructed Stage5 assumption reusing the explicit
        Part-A visualization half-FOV assumption.

    It is NOT claimed to be:
        - a WOMD measurement,
        - a Part-A paper-given communication FOV,
        - a measured optical beamwidth.
    """

    return build_uniform_azimuth_codebook(
        beam_count=int(
            beam_count
        ),

        support_min_azimuth_rad=(
            FROZEN_SUPPORT_MIN_RAD
        ),

        support_max_azimuth_rad=(
            FROZEN_SUPPORT_MAX_RAD
        ),
    )


def normalized_gaussian_directional_gain(
    cell: BeamDecisionCell,
    azimuth_samples_rad,
):
    """
    Constructed normalized directional beam-score pattern:

        G_b(theta)
          = exp[
              -4 ln(2)
              * ((theta-theta_b)/HPBW_b)^2
            ]

    with:

        HPBW_b = decision-cell width.

    Therefore:

        G_b(theta_b) = 1

    and at either decision-cell boundary:

        G_b = 0.5.

    Adjacent physical score patterns therefore overlap.

    This function is NOT the final Part-A optical link
    gain G_opt used for received-power/SNR/BER evaluation.
    """

    values = np.asarray(
        azimuth_samples_rad,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "Azimuth samples must be finite."
        )

    hpbw = (
        float(
            cell.upper_azimuth_rad
        )
        -
        float(
            cell.lower_azimuth_rad
        )
    )

    if (
        not math.isfinite(
            hpbw
        )
        or
        hpbw <= 0.0
    ):
        raise ValueError(
            "Beam HPBW must be positive."
        )

    delta = (
        values
        -
        float(
            cell.center_azimuth_rad
        )
    )

    gain = np.exp(
        (
            -4.0
            *
            math.log(
                2.0
            )
        )
        *
        np.square(
            delta
            /
            hpbw
        )
    )

    return gain.astype(
        np.float64,
        copy=False,
    )


def frozen_beam_scores_from_samples(
    *,
    azimuth_samples_rad,
    beam_count: int,
) -> FrozenBeamScores:
    """
    Compute both frozen Stage5 beam quantities:

        P_b = posterior mass in decision cell
        S_b = E_p[G_b(theta)]

    They remain distinct quantities.
    """

    codebook = (
        build_frozen_directional_codebook(
            beam_count
        )
    )

    probability = (
        beam_probability_mass_from_samples(
            azimuth_samples_rad=(
                azimuth_samples_rad
            ),

            codebook=(
                codebook
            ),
        )
    )

    expected_gain = (
        expected_gain_scores_from_samples(
            azimuth_samples_rad=(
                azimuth_samples_rad
            ),

            codebook=(
                codebook
            ),

            gain_evaluator=(
                normalized_gaussian_directional_gain
            ),
        )
    )

    return FrozenBeamScores(
        probability=(
            probability
        ),

        expected_gain=(
            expected_gain
        ),

        beam_count=(
            int(
                beam_count
            )
        ),

        support_min_deg=(
            FROZEN_SUPPORT_MIN_DEG
        ),

        support_max_deg=(
            FROZEN_SUPPORT_MAX_DEG
        ),

        decision_width_deg=(
            (
                FROZEN_SUPPORT_MAX_DEG
                -
                FROZEN_SUPPORT_MIN_DEG
            )
            /
            float(
                beam_count
            )
        ),

        gain_model=(
            FROZEN_GAIN_MODEL
        ),
    )
