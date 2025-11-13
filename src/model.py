import numpy as np
import pandas as pd
from .constants import ALPHA_BASE, RANGES, SPECIES

def separation_strength(P_kPa: float, T_K: float) -> float:
    """Smooth response function combining pressure (log) and temperature (scaled) effects."""
    p_min, p_max = RANGES["P_kPa"]
    t_min, t_max = RANGES["T_K"]
    p_norm = (np.log(P_kPa) - np.log(np.sqrt(p_min * p_max))) / np.log(p_max / p_min)
    t_norm = (T_K - 0.5 * (t_min + t_max)) / (t_max - t_min)
    S = 1.0 + 0.45 * np.tanh(-p_norm) + 0.35 * np.tanh(-t_norm)
    return float(np.clip(S, 0.4, 1.6))

def effective_alpha(S: float, feed: np.ndarray) -> dict:
    """Blend baseline selectivity with feed-driven enrichment biases."""
    base = {k: (v ** S) for k, v in ALPHA_BASE.items()}
    centered_feed = feed - 1.0 / len(feed)
    bias = np.clip(1.0 + 0.5 * centered_feed, 0.6, 1.4)
    alpha = {}
    for idx, species in enumerate(SPECIES):
        alpha[species] = base[species] * bias[idx]
    return alpha

def product_fractions(feed: np.ndarray, alpha: dict, noise_scale: float = 0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    xH, xD, xT = feed
    w_top = np.array([xH * alpha["H2"], xD * alpha["D2"], xT * alpha["T2"]], dtype=float)
    w_bot = np.array([xH / alpha["H2"], xD / alpha["D2"], xT / alpha["T2"]], dtype=float)
    y_top = w_top / np.sum(w_top)
    y_bot = w_bot / np.sum(w_bot)

    if noise_scale > 0:
        eta_top = rng.normal(0.0, noise_scale, size=3)
        eta_bot = rng.normal(0.0, noise_scale, size=3)
        y_top = np.clip(y_top * (1.0 + eta_top), 1e-9, None); y_top /= np.sum(y_top)
        y_bot = np.clip(y_bot * (1.0 + eta_bot), 1e-9, None); y_bot /= np.sum(y_bot)
    return y_top, y_bot

def simulate_outputs(df_inputs: pd.DataFrame, seed: int = 42):
    delta_rows = []
    detail_rows = []
    for _, r in df_inputs.iterrows():
        feed = np.array([r.feed_H, r.feed_D, r.feed_T], dtype=float)
        S = separation_strength(r.P_kPa, r.T_K)
        alpha = effective_alpha(S, feed)
        y_top, _ = product_fractions(feed, alpha, noise_scale=0.0)
        delta = y_top - feed
        delta_rows.append({
            "delta_H": delta[0],
            "delta_D": delta[1],
            "delta_T": delta[2],
        })
        detail_rows.append({
            "feed_H": feed[0],
            "feed_D": feed[1],
            "feed_T": feed[2],
            "y_top_H2": y_top[0],
            "y_top_D2": y_top[1],
            "y_top_T2": y_top[2],
            "S_eff": S,
        })
    return pd.DataFrame(delta_rows), pd.DataFrame(detail_rows)
