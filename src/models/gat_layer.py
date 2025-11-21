
"""Placeholder Graph Attention Layer (GAT) for STGL-SAFE.

Replace this with a full PyTorch implementation if desired.
"""

import torch
import torch.nn as nn

class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: (N, F), adj: (N, N)
        h = self.lin(x)
        # simple masked average as placeholder
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        h_agg = adj @ h / deg
        return h_agg
