import numpy as np

ALPHA_BASE = {"H2": 1.12, "D2": 1.04, "T2": 1.00}

RANGES = {
    "P_kPa": (60.0, 300.0),
    "T_K": (18.0, 35.0),
}

FEED_TOTAL_MOLES = (50.0, 200.0)

SPECIES = ["H2", "D2", "T2"]
SPECIES_LABELS = {"H2": "H", "D2": "D", "T2": "T"}
SPECIES_COLORS = {
    "H2": "#C0F1EC",
    "D2": "#BBe7B2",
    "T2": "#8FBFBB",
}

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)
