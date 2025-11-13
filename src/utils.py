import numpy as np

def msll(y_true: np.ndarray, y_mu: np.ndarray, y_std: np.ndarray, eps: float = 1e-12) -> float:
    var = np.clip(y_std**2, eps, None)
    return float(np.mean(0.5*np.log(2*np.pi*var) + 0.5*((y_true - y_mu)**2)/var))
