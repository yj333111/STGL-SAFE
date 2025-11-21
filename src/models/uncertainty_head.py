
"""Output head producing mean and variance (heteroscedastic regression)."""

import torch
import torch.nn as nn

class UncertaintyHead(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.mean_head = nn.Linear(in_dim, out_dim)
        self.logvar_head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mean, logvar
