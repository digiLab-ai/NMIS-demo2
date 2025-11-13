import numpy as np

ALPHA_BASE = {"H2": 1.12, "D2": 1.04, "T2": 1.00}

RANGES = {
    "P_kPa": (60.0, 300.0),
    "T_K": (18.0, 35.0),
}

SPECIES = ["H2", "D2", "T2"]

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)
