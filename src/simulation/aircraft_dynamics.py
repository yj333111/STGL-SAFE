
"""Skeleton for rescue aircraft dynamics (Section 3.2.1)."""

import numpy as np

class SimpleAircraftModel:
    def __init__(self, mass=1500.0):
        self.mass = mass

    def step(self, state, control, disturbance, dt):
        """Very simple placeholder integrator.

        Parameters
        ----------
        state : np.ndarray
            Placeholder state vector.
        control : np.ndarray
            Control input.
        disturbance : np.ndarray
            External disturbance (e.g., from fire environment).
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Next state (placeholder).
        """
        return state + dt * (control + disturbance) / max(self.mass, 1.0)
