import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str = "", color: str = "#16425B"):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(y_true, y_pred, s=20, alpha=0.7)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    if title:
        ax.set_title(title)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf

def line_with_uncertainty(idx: np.ndarray, y_true: np.ndarray, y_mu: np.ndarray, y_std: np.ndarray, title: str = "", color="#16D5C2"):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(idx, y_true, marker="o", linestyle="", label="Ground truth")
    ax.plot(idx, y_mu, marker="x", linestyle="-", label="Prediction")
    ax.fill_between(idx, y_mu - 1.96*y_std, y_mu + 1.96*y_std, alpha=0.25, label="±1.96σ")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    ax.legend()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf

def parity_grid_with_errorbars(series: dict, outputs: list, nrows: int = 2, ncols: int = 3):
    """Plot multiple parity plots with 95% CI error bars on a shared grid."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = axes.flatten()
    for idx, output in enumerate(outputs):
        ax = axes[idx]
        y_true, y_pred, y_std = series[output]
        yerr = 1.96 * np.clip(y_std, 0.0, None)
        ax.errorbar(
            y_true,
            y_pred,
            yerr=yerr,
            fmt="o",
            markersize=3,
            alpha=0.8,
            ecolor="#9DA9A0",
            elinewidth=0.8,
            capsize=1.5,
        )
        lims = [
            min(np.min(y_true), np.min(y_pred - yerr)),
            max(np.max(y_true), np.max(y_pred + yerr)),
        ]
        ax.plot(lims, lims, linestyle="--", color="#555555", linewidth=1)
        ax.set_title(output)
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    for j in range(len(outputs), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf
