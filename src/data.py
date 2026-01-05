
import numpy as np
import pandas as pd

from constants import RANGES, NOISE_STD
from model import screw_jack_step

def generate_dataset(n_runs: int = 50, dt: float = 1.0) -> pd.DataFrame:
    rows = []

    for run_id in range(n_runs):
        load_kg = np.random.uniform(*RANGES["load_kg"])
        screw_pitch_m = np.random.uniform(*RANGES["screw_pitch_m"])
        efficiency = np.random.uniform(*RANGES["efficiency"])
        rpm = np.random.uniform(*RANGES["rpm"])

        height = 0.0
        time = 0.0

        while time <= RANGES["time_s"][1]:
            dh, torque = screw_jack_step(
                load_kg,
                screw_pitch_m,
                efficiency,
                rpm,
                dt
            )

            # Inject noise so GP does not fit perfectly
            height_noisy = height + np.random.normal(0, NOISE_STD["height_m"])
            torque_noisy = torque + np.random.normal(0, NOISE_STD["torque_Nm"])

            rows.append({
                "run_id": run_id,
                "time_s": time,
                "load_kg": load_kg,
                "screw_pitch_m": screw_pitch_m,
                "efficiency": efficiency,
                "rpm": rpm,
                "height_m": height_noisy,
                "torque_Nm": torque_noisy,
            })

            height += dh
            time += dt

    return pd.DataFrame(rows)
