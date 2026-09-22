import torch
import torch.nn as nn


class ManualLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)

        variance = (
            (x - mean) ** 2
        ).mean(dim=-1, keepdim=True)

        normalized = (
            x - mean
        ) / torch.sqrt(variance + self.eps)

        return self.gamma * normalized + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        print(
            f"Creating Feed Forward Network: "
            f"{d_model} -> {d_ff} -> {d_model}"
        )

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))
