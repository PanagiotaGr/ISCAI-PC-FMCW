from __future__ import annotations

from dataclasses import dataclass
import math


PART_A_GIT_COMMIT = (
    "44d62e3478e3818d1757b00971890f844cb032f7"
)

PART_A_NOTEBOOK_SHA256 = (
    "b5a80a6d3441de6d571db4f65b4a43ed4052cc2b3ccba935ad31b5dd51316ef3"
)


@dataclass(frozen=True)
class PartAReferenceParameters:
    """
    Frozen numerical parameters reproduced from the
    Part-A notebook.

    No Stage-2 tuning belongs in this dataclass.
    """

    c_mps: float = 299_792_458.0

    carrier_frequency_hz: float = 193.4e12
    bandwidth_hz: float = 10.0e9
    chirp_duration_s: float = 10.0e-6
    data_rate_bps: float = 1.0e9

    n_fast: int = 131_072
    m_chirps: int = 64

    def __post_init__(self) -> None:
        positive = (
            self.c_mps,
            self.carrier_frequency_hz,
            self.bandwidth_hz,
            self.chirp_duration_s,
            self.data_rate_bps,
        )

        if not all(
            math.isfinite(value)
            and value > 0.0
            for value in positive
        ):
            raise ValueError(
                "Part-A physical parameters must be "
                "finite and positive."
            )

        if self.n_fast <= 0:
            raise ValueError(
                "n_fast must be positive."
            )

        if self.m_chirps <= 0:
            raise ValueError(
                "m_chirps must be positive."
            )

    @property
    def chirp_slope_hz_per_s(
        self,
    ) -> float:
        return (
            self.bandwidth_hz
            / self.chirp_duration_s
        )

    @property
    def wavelength_m(
        self,
    ) -> float:
        return (
            self.c_mps
            / self.carrier_frequency_hz
        )

    @property
    def symbol_duration_s(
        self,
    ) -> float:
        return 1.0 / self.data_rate_bps

    @property
    def prf_hz(
        self,
    ) -> float:
        return 1.0 / self.chirp_duration_s

    @property
    def slow_time_rate_hz(
        self,
    ) -> float:
        # In the frozen notebook this equals PRF.
        return self.prf_hz

    @property
    def range_resolution_m(
        self,
    ) -> float:
        return (
            self.c_mps
            / (2.0 * self.bandwidth_hz)
        )


FROZEN_PART_A = PartAReferenceParameters()
