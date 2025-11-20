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

def effective_alpha(S: float, feed_frac: np.ndarray) -> dict:
    """Blend baseline selectivity with feed-driven enrichment biases."""
    base = {k: (v ** S) for k, v in ALPHA_BASE.items()}
    centered_feed = feed_frac - 1.0 / len(feed_frac)
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
    rng = np.random.default_rng(seed)
    for _, r in df_inputs.iterrows():
        feed = np.array([r.feed_H, r.feed_D, r.feed_T], dtype=float)
        total_moles = np.sum(feed)
        if total_moles <= 0:
            continue
        feed_frac = feed / total_moles

        S = separation_strength(r.P_kPa, r.T_K) 
        alpha = effective_alpha(S, feed_frac)
        noise_scale = 0.01
        y_top, _ = product_fractions(feed_frac, alpha, noise_scale=noise_scale, rng=rng)
        raw_top_moles = y_top * total_moles
        drawdown = rng.uniform(0.01, 0.08, size=3)
        top_capacity = feed * np.clip(1.0 - drawdown, 0.0, 1.0)
        top_moles = np.minimum(raw_top_moles, top_capacity)
        top_total = np.sum(top_moles)
        if top_total > 0:
            top_frac = top_moles / top_total
        else:
            top_frac = np.zeros_like(top_moles)
        delta = top_moles - feed
        delta_rows.append({
            "delta_H": delta[0],
            "delta_D": delta[1],
            "delta_T": delta[2],
        })
        detail_rows.append({
            "feed_H": feed[0],
            "feed_D": feed[1],
            "feed_T": feed[2],
            "top_H": top_moles[0],
            "top_D": top_moles[1],
            "top_T": top_moles[2],
            "top_frac_H": top_frac[0],
            "top_frac_D": top_frac[1],
            "top_frac_T": top_frac[2],
            "S_eff": S,
        })
    return pd.DataFrame(delta_rows), pd.DataFrame(detail_rows)
