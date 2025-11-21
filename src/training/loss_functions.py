
"""Loss functions for heteroscedastic Gaussian likelihood (Eq.(23))."""

import torch

def heteroscedastic_gaussian_nll(y_true, mean, logvar):
    """Negative log-likelihood for Gaussian with input-dependent variance."""
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (logvar + (y_true - mean) ** 2 * inv_var)
    return nll.mean()
