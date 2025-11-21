
"""Simple visualization utilities (placeholder)."""

import matplotlib.pyplot as plt

def plot_fire_intensity(field, title="Fire intensity"):
    plt.figure()
    plt.imshow(field, origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
