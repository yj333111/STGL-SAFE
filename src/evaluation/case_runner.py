
"""Simple case runner to illustrate multi-run evaluation."""

import numpy as np
from .evaluate_metrics import rmse, iou, abd

def run_dummy_case(runs=5):
    results = []
    for r in range(runs):
        # placeholder random metrics
        pred = np.random.rand(100)
        target = np.random.rand(100)
        rmse_val = rmse(pred, target)
        iou_val = iou(pred > 0.5, target > 0.5)
        abd_val = abd(pred, target)
        results.append((rmse_val, iou_val, abd_val))

    results = np.array(results)
    mean = results.mean(axis=0)
    std = results.std(axis=0)
    print("RMSE mean ± std:", mean[0], std[0])
    print("IoU  mean ± std:", mean[1], std[1])
    print("ABD  mean ± std:", mean[2], std[2])
