
"""Simple temporal module placeholder (e.g., TCN/GRU)."""

import torch
import torch.nn as nn

class SimpleTemporalModule(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x_seq):
        # x_seq: (T, N, F) -> treat N as batch for simplicity
        T, N, F = x_seq.shape
        x_seq = x_seq.permute(1, 0, 2)  # (N, T, F)
        out, _ = self.gru(x_seq)
        return out[:, -1, :]  # (N, hidden_dim)
