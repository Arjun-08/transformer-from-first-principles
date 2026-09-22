import math
import torch


class SinusoidalPositionalEncoding:
    def __init__(self, max_length, d_model):
        self.max_length = max_length
        self.d_model = d_model

        print(
            f"Creating positional encoding: "
            f"[{max_length}, {d_model}]"
        )

        position = torch.arange(
            max_length,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            ) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_length, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def __call__(self, x):
        sequence_length = x.size(1)
        return x + self.pe[:, :sequence_length, :].to(x.device)
