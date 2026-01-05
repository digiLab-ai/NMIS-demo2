
import numpy as np
from constants import G

def screw_jack_step(
    load_kg: float,
    screw_pitch_m: float,
    efficiency: float,
    rpm: float,
    dt: float,
):
    """
    Single simulation step of a screw jack.
    Returns height increment and required torque.
    """

    load_N = load_kg * G

    # Angular speed (rad/s)
    omega = 2 * np.pi * rpm / 60.0

    # Linear velocity (m/s)
    v = screw_pitch_m * rpm / 60.0

    # Output power
    P_out = load_N * v

    # Input power including losses
    P_in = P_out / max(efficiency, 1e-3)

    # Torque required
    torque = P_in / max(omega, 1e-3)

    # Height increment
    dh = v * dt

    return dh, torque
