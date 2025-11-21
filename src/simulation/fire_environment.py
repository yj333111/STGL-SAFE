
"""Simplified fire environment interface (Section 3.1)."""

import numpy as np

class FireEnvironment:
    def __init__(self, fire_intensity, smoke_concentration, wind_field):
        self.fire_intensity = fire_intensity
        self.smoke_concentration = smoke_concentration
        self.wind_field = wind_field

    def get_state_at(self, t_idx):
        """Return environment slice at time index t_idx."""
        return dict(
            fire=self.fire_intensity[t_idx],
            smoke=self.smoke_concentration[t_idx],
            wind=self.wind_field[t_idx],
        )
