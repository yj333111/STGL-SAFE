
"""Normalization utilities."""

import numpy as np

def min_max_normalize(x, eps=1e-8):
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + eps)
