import torch.nn as nn

from src.attention import MultiHeadSelfAttention
from src.layers import ManualLayerNorm, FeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads
        )

        self.norm1 = ManualLayerNorm(d_model)

        self.ffn = FeedForward(
            d_model,
            d_ff
        )

        self.norm2 = ManualLayerNorm(d_model)

    def forward(self, x):
        attention_output = self.attention(x)

        x = self.norm1(
            x + attention_output
        )

        ffn_output = self.ffn(x)

        x = self.norm2(
            x + ffn_output
        )

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff
    ):
        super().__init__()

        print("\nCreating Transformer Encoder")
        print(f"Number of layers: {num_layers}")

        layers = []

        for i in range(num_layers):
            print(f"Creating encoder layer {i + 1}")

            layers.append(
                TransformerEncoderLayer(
                    d_model,
                    num_heads,
                    d_ff
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
