
"""Spatiotemporal graph construction.

This is a simple reference implementation that takes a regular grid
and builds an adjacency matrix using 4- or 8-neighborhoods.
"""

import numpy as np

def build_grid_adjacency(H, W, eight_neighborhood=True):
    N = H * W
    A = np.zeros((N, N), dtype=np.float32)

    def idx(i, j):
        return i * W + j

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if eight_neighborhood:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(H):
        for j in range(W):
            u = idx(i, j)
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    v = idx(ni, nj)
                    A[u, v] = 1.0
    return A
