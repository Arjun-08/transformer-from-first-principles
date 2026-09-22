import torch
import torch.nn as nn

from src.positional_encoding import SinusoidalPositionalEncoding
from src.transformer import TransformerEncoder


class ManualEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        print(
            f"Creating embedding matrix: "
            f"[{vocab_size}, {d_model}]"
        )

        self.weight = nn.Parameter(
            torch.randn(vocab_size, d_model) * 0.02
        )

    def forward(self, token_ids):
        return self.weight[token_ids]


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_length,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_classes=2
    ):
        super().__init__()

        print("\n")
        print("=" * 70)
        print("BUILDING TINY TRANSFORMER")
        print("=" * 70)

        self.d_model = d_model

        self.embedding = ManualEmbedding(
            vocab_size,
            d_model
        )

        self.position = SinusoidalPositionalEncoding(
            max_length,
            d_model
        )

        self.encoder = TransformerEncoder(
            num_layers,
            d_model,
            num_heads,
            d_ff
        )

        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

        print("\nClassifier:")
        print(f"  {d_model} -> {num_classes}")
        print("=" * 70)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = self.position(x)
        x = self.encoder(x)

        cls_representation = x[:, 0, :]

        return self.classifier(
            cls_representation
        )
