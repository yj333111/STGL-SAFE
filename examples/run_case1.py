
"""Example script for Case 1 (single fire source with uniform wind)."""

# This is a simple placeholder that could be extended to reproduce Fig. 4 style plots.

from src.utils.seed import set_seed
from src.utils.visualize import plot_fire_intensity
from src.preprocessing.data_loader import load_sample_environment
import os

if __name__ == '__main__':
    set_seed(2025)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_fire_data')
    env = load_sample_environment(data_dir)
    # Just plot the first time step as a placeholder
    plot_fire_intensity(env['fire_intensity'][0], title="Case 1: sample fire intensity at t0")
