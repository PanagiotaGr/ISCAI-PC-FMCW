import torch
from torch import nn


class GRUPredictor(nn.Module):

    def __init__(
        self,
        input_size=8,
        hidden=128,
        layers=1,
        dropout=0.0,
        future=10
    ):

        super().__init__()

        self.future = future


        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.head = nn.Linear(
            hidden,
            future*2
        )


    def forward(self,x):

        _,h = self.gru(x)


        out = self.dropout(
            h[-1]
        )


        out = self.head(
            out
        )


        return out.reshape(
            -1,
            self.future,
            2
        )
