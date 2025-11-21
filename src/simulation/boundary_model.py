
"""Safety operational boundary modeling (Section 3.2.3) – simplified."""

import numpy as np

def safety_margin(dynamic_term, environment_term, sensor_term):
    """Placeholder for Eq.(22) style safety margin.

    Returns positive values for safe states, negative for unsafe.
    """
    return dynamic_term - (environment_term + sensor_term)

def is_safe(margin):
    return margin >= 0.0
