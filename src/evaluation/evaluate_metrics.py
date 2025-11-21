
"""Evaluation metrics: RMSE, IoU, ABD (simplified)."""

import numpy as np

def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

def iou(pred_mask, true_mask):
    inter = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return inter / max(union, 1)

def abd(pred_boundary, true_boundary):
    """Average boundary deviation (simple L2 difference)."""
    return float(np.mean(np.abs(pred_boundary - true_boundary)))
