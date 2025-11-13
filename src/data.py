from .sampler import lhs_inputs
from .model import simulate_outputs

def generate_dataset(n_samples: int = 300, seed: int = 42, return_details: bool = False):
    X = lhs_inputs(n_samples=n_samples, seed=seed)
    Y, details = simulate_outputs(X, seed=seed)
    if return_details:
        return X, Y, details
    return X, Y
