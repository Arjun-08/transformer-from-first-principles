import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        print("Creating Multi-Head Attention")
        print(f"  d_model   = {d_model}")
        print(f"  heads     = {num_heads}")
        print(f"  head_dim  = {self.head_dim}")

        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        self.last_attention = None

    def split_heads(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size = x.size(0)
        seq_len = x.size(2)

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            seq_len,
            self.d_model
        )

    def forward(self, x):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        attention = torch.softmax(scores, dim=-1)
        self.last_attention = attention.detach()

        context = torch.matmul(attention, V)
        context = self.combine_heads(context)

        return self.W_o(context)
