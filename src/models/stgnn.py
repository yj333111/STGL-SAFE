
"""STGNN skeleton for the STGL-SAFE framework."""

import torch
import torch.nn as nn

from .gat_layer import SimpleGATLayer
from .temporal_module import SimpleTemporalModule
from .uncertainty_head import UncertaintyHead

class STGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gat = SimpleGATLayer(in_dim, hidden_dim)
        self.temporal = SimpleTemporalModule(hidden_dim, hidden_dim)
        self.head = UncertaintyHead(hidden_dim, out_dim=1)

    def forward(self, x_seq, adj):
        # x_seq: (T, N, F)
        T, N, F = x_seq.shape
        # apply spatial operator at each time step
        x_spatial = []
        for t in range(T):
            x_t = x_seq[t]            # (N, F)
            x_g = self.gat(x_t, adj)  # (N, H)
            x_spatial.append(x_g)
        x_spatial = torch.stack(x_spatial, dim=0)  # (T, N, H)
        # temporal aggregation
        x_temporal = self.temporal(x_spatial)      # (N, H)
        mean, logvar = self.head(x_temporal)
        return mean, logvar
