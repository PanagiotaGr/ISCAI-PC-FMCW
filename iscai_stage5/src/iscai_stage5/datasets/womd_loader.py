from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset


class WOMDTrajectoryDataset(Dataset):

    def __init__(
        self,
        path: str,
    ):

        with open(path, "r") as f:
            data = json.load(f)


        self.samples = data


    def __len__(self):

        return len(self.samples)


    def __getitem__(
        self,
        index,
    ):

        item = self.samples[index]


        past = torch.tensor(
            item["past"],
            dtype=torch.float32,
        )


        future = torch.tensor(
            item["future"],
            dtype=torch.float32,
        )


        return (
            past,
            future,
        )
