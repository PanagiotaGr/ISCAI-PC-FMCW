from __future__ import annotations

from dataclasses import dataclass
import math


IMM_MODE_NAMES = (
    "CV",
    "CA",
    "CTRV",
)


Vector3 = tuple[
    float,
    float,
    float,
]


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True)
class IMMProbabilities:
    """
    IMM model probabilities in frozen order:

        [CV, CA, CTRV]
    """

    cv: float
    ca: float
    ctrv: float

    def as_tuple(
        self,
    ) -> Vector3:
        return (
            self.cv,
            self.ca,
            self.ctrv,
        )

    def __post_init__(
        self,
    ) -> None:

        values = self.as_tuple()

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "IMM probabilities must be finite."
            )

        if any(
            value < 0.0
            for value in values
        ):
            raise ValueError(
                "IMM probabilities cannot be negative."
            )

        if abs(
            sum(values) - 1.0
        ) > 1e-12:
            raise ValueError(
                "IMM probabilities must sum to one."
            )


@dataclass(frozen=True)
class IMMMixingResult:
    """
    Result of the IMM interaction/mixing probability step.

    predicted_probabilities:
        mu_j^-

    conditional_mixing:
        mu_{i|j}

    Rows i:
        previous mode

    Columns j:
        destination mode
    """

    predicted_probabilities: (
        IMMProbabilities
    )

    conditional_mixing: Matrix3


def _validate_transition_matrix(
    matrix: Matrix3,
) -> None:

    if len(matrix) != 3:
        raise ValueError(
            "IMM transition matrix must be 3x3."
        )

    for row in matrix:

        if len(row) != 3:
            raise ValueError(
                "IMM transition matrix must be 3x3."
            )

        if not all(
            math.isfinite(value)
            for value in row
        ):
            raise ValueError(
                "IMM transition probabilities "
                "must be finite."
            )

        if any(
            value < 0.0
            for value in row
        ):
            raise ValueError(
                "IMM transition probabilities "
                "cannot be negative."
            )

        if abs(
            sum(row) - 1.0
        ) > 1e-12:
            raise ValueError(
                "Each IMM transition row "
                "must sum to one."
            )


def imm_mixing_probabilities(
    *,
    probabilities: IMMProbabilities,
    transition_matrix: Matrix3,
) -> IMMMixingResult:
    """
    Standard IMM interaction probability step.

    transition_matrix[i][j]:

        P(mode_k = j | mode_{k-1} = i)
    """

    _validate_transition_matrix(
        transition_matrix
    )

    mu = probabilities.as_tuple()

    predicted = []

    for destination in range(3):

        value = sum(
            mu[source]
            *
            transition_matrix[
                source
            ][
                destination
            ]
            for source in range(3)
        )

        predicted.append(
            value
        )


    if any(
        value <= 0.0
        for value in predicted
    ):
        raise ValueError(
            "IMM destination probability is zero; "
            "conditional mixing is undefined."
        )


    conditional = []

    for source in range(3):

        row = []

        for destination in range(3):

            value = (
                transition_matrix[
                    source
                ][
                    destination
                ]
                *
                mu[source]
                /
                predicted[
                    destination
                ]
            )

            row.append(
                value
            )

        conditional.append(
            tuple(row)
        )


    # Each destination column must sum to one.
    for destination in range(3):

        column_sum = sum(
            conditional[source][
                destination
            ]
            for source in range(3)
        )

        if abs(
            column_sum - 1.0
        ) > 1e-12:
            raise RuntimeError(
                "IMM conditional mixing "
                "column does not sum to one."
            )


    predicted_probabilities = (
        IMMProbabilities(
            cv=predicted[0],
            ca=predicted[1],
            ctrv=predicted[2],
        )
    )


    return IMMMixingResult(
        predicted_probabilities=(
            predicted_probabilities
        ),
        conditional_mixing=tuple(
            conditional
        ),
    )


def update_imm_probabilities(
    *,
    predicted_probabilities:
        IMMProbabilities,

    measurement_likelihoods: Vector3,
) -> IMMProbabilities:
    """
    Bayesian IMM mode-probability correction.

        mu_j =
            Lambda_j * mu_j^-
            ------------------
            sum_k Lambda_k * mu_k^-
    """

    if not all(
        math.isfinite(value)
        for value
        in measurement_likelihoods
    ):
        raise ValueError(
            "IMM likelihoods must be finite."
        )

    if any(
        value < 0.0
        for value
        in measurement_likelihoods
    ):
        raise ValueError(
            "IMM likelihoods cannot be negative."
        )


    predicted = (
        predicted_probabilities
        .as_tuple()
    )


    unnormalized = tuple(
        predicted[index]
        *
        measurement_likelihoods[index]
        for index in range(3)
    )


    normalizer = sum(
        unnormalized
    )


    if normalizer <= 0.0:
        raise ValueError(
            "IMM probability update has "
            "zero normalization constant."
        )


    normalized = tuple(
        value / normalizer
        for value
        in unnormalized
    )


    return IMMProbabilities(
        cv=normalized[0],
        ca=normalized[1],
        ctrv=normalized[2],
    )
