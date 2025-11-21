
"""Main entry point for STGL-SAFE example code.

This script shows how the different components can be connected.
"""

import os
import argparse
import numpy as np

from preprocessing.data_loader import load_sample_environment
from preprocessing.normalize import min_max_normalize
from preprocessing.build_graph import build_grid_adjacency
from training.train_stgnn import train_dummy
from utils.seed import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    set_seed(2025)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_fire_data')
    env = load_sample_environment(data_dir)

    # Simple normalization
    fire = min_max_normalize(env['fire_intensity'])
    T, H, W = fire.shape
    N = H * W

    # Build adjacency
    adj = build_grid_adjacency(H, W, eight_neighborhood=True)

    # Build dummy sequence of node features from fire tensor only
    x_seq = fire.reshape(T, N, 1)
    # Dummy target
    y = np.mean(x_seq, axis=0, keepdims=False)  # (N, 1) after later expansion
    y = y.reshape(N, 1)

    if args.train:
        train_dummy(adj, x_seq, y, epochs=3, lr=1e-3)

    if args.eval:
        print("Evaluation logic can be added here (e.g., loading a trained model).")

if __name__ == '__main__':
    main()
