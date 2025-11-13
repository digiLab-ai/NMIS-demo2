import numpy as np
import pandas as pd
from scipy.stats import qmc
from .constants import RANGES

def lhs_unit(n_samples: int, dim: int, seed: int = 42) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    return sampler.random(n=n_samples)

def scale_to_range(u: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (high - low) * u

def lhs_inputs(n_samples: int, seed: int = 42) -> pd.DataFrame:
    u = lhs_unit(n_samples, dim=2, seed=seed)
    cols = ["P_kPa", "T_K"]
    X = {}
    for i, c in enumerate(cols):
        lo, hi = RANGES[c]
        X[c] = scale_to_range(u[:, i], lo, hi)

    rng = np.random.default_rng(seed)
    feed = rng.dirichlet(alpha=[1.6, 1.3, 1.1], size=n_samples)
    X["x_H2"] = feed[:, 0]
    X["x_D2"] = feed[:, 1]
    X["x_T2"] = feed[:, 2]
    return pd.DataFrame(X)
