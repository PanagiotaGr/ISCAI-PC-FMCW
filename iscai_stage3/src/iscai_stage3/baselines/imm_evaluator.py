from __future__ import annotations

from dataclasses import dataclass

from iscai_stage3.baselines.imm_innovation import (
    IMMInnovation,
    build_innovation_record,
)

from iscai_stage3.baselines.imm_likelihood import (
    gaussian_log_likelihood,
)


Measurement = tuple[
    float,
    float,
    float,
    float,
]


@dataclass(frozen=True)
class IMMModeMeasurement:

    mode_name: str

    predicted_measurement: Measurement



def evaluate_mode(
    *,
    mode: IMMModeMeasurement,
    measured: Measurement,
    covariance,
) -> IMMInnovation:
    """
    Evaluate one IMM mode against the same
    causal measurement.

    No future trajectory access.
    """

    log_likelihood = (
        gaussian_log_likelihood(
            innovation=tuple(
                measured[i]
                -
                mode.predicted_measurement[i]
                for i in range(4)
            ),
            innovation_covariance=covariance,
        )
    )

    return build_innovation_record(
        mode_name=mode.mode_name,
        predicted_measurement=(
            mode.predicted_measurement
        ),
        measured=measured,
        covariance_trace=sum(
            covariance[i][i]
            for i in range(
                len(covariance)
            )
        ),
        log_likelihood=log_likelihood,
    )
