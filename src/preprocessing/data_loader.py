
"""Data loading utilities for STGL-SAFE.

This module loads sample fire environment tensors and prepares them
for spatiotemporal graph construction.
"""

import numpy as np
import os

def load_sample_environment(data_dir):
    """Load sample fire environment tensors from `data_dir`.

    Returns
    -------
    dict
        Dictionary with keys: 'fire_intensity', 'smoke_concentration',
        'wind_field', 'terrain_dem'.
    """
    env = {}
    env['fire_intensity'] = np.load(os.path.join(data_dir, 'fire_intensity.npy'))
    env['smoke_concentration'] = np.load(os.path.join(data_dir, 'smoke_concentration.npy'))
    env['wind_field'] = np.load(os.path.join(data_dir, 'wind_field.npy'))
    env['terrain_dem'] = np.load(os.path.join(data_dir, 'terrain_dem.npy'))
    return env
