from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrajectorySample:
    past: torch.Tensor
    future: torch.Tensor


class TrajectoryDataset(Dataset):

    def __init__(
        self,
        samples: list[TrajectorySample],
    ) -> None:

        self.samples = samples


    def __len__(self) -> int:

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index: int,
    ):

        sample = self.samples[
            index
        ]

        return (
            sample.past,
            sample.future,
        )
