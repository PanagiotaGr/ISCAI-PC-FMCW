import torch
from torch import nn


class GRUPredictor(nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden=64,
        future=10,
    ):
        super().__init__()

        self.future=future

        self.gru = nn.GRU(
            input_size,
            hidden,
            batch_first=True
        )

        self.head = nn.Linear(
            hidden,
            future*2
        )


    def forward(self,x):

        _,h = self.gru(x)

        out = self.head(
            h[-1]
        )

        return out.reshape(
            -1,
            self.future,
            2
        )
