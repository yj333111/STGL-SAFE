
"""Trajectory cost function (Eq.(24) style) – simplified."""

import numpy as np

def trajectory_cost(risk, energy, fluctuation, alpha=1.0, beta=0.1, gamma=0.1):
    return alpha * risk + beta * energy + gamma * fluctuation
