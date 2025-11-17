import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from pathlib import Path
import sys

# Ensure the repository root is importable so `src` works even when run from app/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import BRAND, CONFIG
from src.data import generate_dataset
from src.utils import msll
import src.constants as const

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

SPECIES_COLORS = const.SPECIES_COLORS
SPECIES_LABELS = const.SPECIES_LABELS
RANGES = const.RANGES
FEED_TOTAL_MOLES = getattr(const, "FEED_TOTAL_MOLES", (50.0, 200.0))

st.set_page_config(page_title="Cryo Isotope GP Emulator", page_icon="❄️", layout="wide")

ASSETS_DIR = ROOT_DIR / "assets"
HERO_PATH = ASSETS_DIR / "CryoDistil.png"
SIDEBAR_LOGO = ASSETS_DIR / "digilab.png"

def set_base_font():
    st.markdown(
        """
        <style>
        html, body, .stApp {
            font-family: "Helvetica Neue", sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_base_font()


st.title("❄️ Cryogenic Isotope Distillation Simulator")
st.caption("")
# Hero image (scaled down to ~33% width)
if HERO_PATH.exists():
    st.image(str(HERO_PATH), caption="Cryogenic distillation emulator overview", width=600)

with st.expander("What does this generator do? (click to expand)"):
    st.markdown(
        """
**Purpose.** Generate a synthetic ("fake-CFD") dataset for a cryogenic **hydrogen isotope** distillation segment:
inputs → product mole shifts. Use the CSVs to train GP emulators elsewhere, then upload their predictions here for validation.

**Inputs (X).**
- `P_kPa` — Column pressure (kPa) | range: 60–300
- `T_K` — Column temperature (K) | range: 18–35
- `feed_H`, `feed_D`, `feed_T` — Feed moles (not normalised)

**Outputs (Y).**
- `delta_*` — Change in top product moles vs feed (`top - feed`) for H/D/T

**How it works.**
- Specify the desired number of samples and random seed.
- The app will generate a Latin hypercube sample (LHS) of the inputs within specified ranges.
- The mock-CFD simulation will run for each sample input and generate the top and bottom isotope mole splits.
- The user specifies the train/validation split fraction.
- The app displays previews of the generated input/output datasets and allows CSV downloads.
- These CSV downloads can be used to train a model in the Uncertainty Engine.
- The model can be used on the validation inputs.
- The validation tab can be used to compare the ground truth (validation outputs, `val_Y.csv`) vs the model predictions and uncertainties.
        """
    )

# Sidebar
if SIDEBAR_LOGO.exists():
    st.sidebar.image(str(SIDEBAR_LOGO), use_container_width=True)

st.sidebar.header("Settings")
n_samples = st.sidebar.number_input("Number of samples", min_value=1, max_value=1000, value=CONFIG.default_samples, step=50)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=CONFIG.seed, step=1)
train_frac = st.sidebar.slider("Train fraction", min_value=0.5, max_value=0.95, value=float(CONFIG.train_fraction), step=0.05)

# Generate
X, Y, details = generate_dataset(n_samples=int(n_samples), seed=int(seed), return_details=True)

# Train/validation split (deterministic for given seed)
rng = np.random.default_rng(int(seed))
indices = rng.permutation(len(X))
if len(X) < 2:
    train_idx = indices
    val_idx = indices
else:
    n_train = int(round(train_frac * len(X)))
    n_train = min(max(n_train, 1), len(X) - 1)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

def take_rows(df, idx):
    return df.iloc[idx].reset_index(drop=True)

X_train, X_val = take_rows(X, train_idx), take_rows(X, val_idx)
Y_train, Y_val = take_rows(Y, train_idx), take_rows(Y, val_idx)

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# Tabs
tab_data, tab_validate, tab_vis = st.tabs(["Data & Downloads", "Validation", "Visualisation"])

with tab_data:
    st.subheader("Dataset previews")
    st.caption(f"Train samples: **{len(X_train)}** • Validation samples: **{len(X_val)}** (seeded shuffle)")
    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown("**Inputs (P, T, feed)**")
        with st.expander("Show first 20 rows", expanded=False):
            st.dataframe(X.head(20), use_container_width=True)
        st.download_button("Download train inputs (train_X.csv)", csv_bytes(X_train), file_name="train_X.csv")
        st.download_button("Download validation inputs (val_X.csv)", csv_bytes(X_val), file_name="val_X.csv")
    with col_out:
        st.markdown("**Outputs (delta moles)**")
        with st.expander("Show first 20 rows", expanded=False):
            st.dataframe(Y.head(20), use_container_width=True)
        st.download_button("Download train outputs (train_Y.csv)", csv_bytes(Y_train), file_name="train_Y.csv")
        st.download_button("Download validation outputs (val_Y.csv)", csv_bytes(Y_val), file_name="val_Y.csv")

    st.markdown("---")
    st.subheader("Per-sample composition explorer")
    idx = st.slider("Sample index", 0, len(Y) - 1, 0, 1)
    row_delta = Y.iloc[idx]
    row_detail = details.iloc[idx]
    row_input = X.iloc[idx]
    feed_labels = [SPECIES_LABELS["H2"], SPECIES_LABELS["D2"], SPECIES_LABELS["T2"]]
    feed_colors = [SPECIES_COLORS["H2"], SPECIES_COLORS["D2"], SPECIES_COLORS["T2"]]

    c_feed, c_top, c_delta = st.columns(3)
    with c_feed:
        feed_values = np.array([row_detail.feed_H, row_detail.feed_D, row_detail.feed_T], dtype=float)
        feed_share = feed_values / feed_values.sum() if feed_values.sum() > 0 else np.ones(3) / 3
        fig_feed = px.pie(
            values=feed_share,
            names=feed_labels,
            hole=0.35,
            title="Feed composition (share)",
        )
        fig_feed.update_traces(marker=dict(colors=feed_colors))
        fig_feed.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_feed, use_container_width=True)
        st.markdown(
            f"**Pressure:** {row_input.P_kPa:.1f} kPa<br>"
            f"**Temperature:** {row_input.T_K:.1f} K<br>"
            f"**Feed moles (H/D/T):** {feed_values[0]:.1f} / {feed_values[1]:.1f} / {feed_values[2]:.1f}",
            unsafe_allow_html=True,
        )
    with c_top:
        top_values = np.array([row_detail.top_H, row_detail.top_D, row_detail.top_T], dtype=float)
        top_share = top_values / top_values.sum() if top_values.sum() > 0 else np.ones(3) / 3
        fig_top = px.pie(
            values=top_share,
            names=feed_labels,
            hole=0.35,
            title="Top composition (share)",
        )
        fig_top.update_traces(marker=dict(colors=feed_colors))
        fig_top.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_top, use_container_width=True)
        st.caption(f"Top moles (H/D/T): {top_values[0]:.1f} / {top_values[1]:.1f} / {top_values[2]:.1f}")
    with c_delta:
        fig_delta = px.bar(
            x=feed_labels,
            y=[row_delta.delta_H, row_delta.delta_D, row_delta.delta_T],
            title="Delta (top - feed)",
            color=feed_labels,
            color_discrete_map={label: color for label, color in zip(feed_labels, feed_colors)},
        )
        fig_delta.update_layout(yaxis_title="Mole change", xaxis_title="")
        st.plotly_chart(fig_delta, use_container_width=True)

with tab_validate:
    st.subheader("Upload GP results for validation")
    st.markdown(
        """
Upload **three CSV files** generated elsewhere:
1) **Ground truth** (`val_Y.csv`)  
2) **Predictions** (`pred_Y.csv`)
3) **Uncertainty** (`std_Y.csv`)

        """
    )
    true_file = st.file_uploader("Ground truth CSV", type=["csv"], key="true")
    pred_file = st.file_uploader("Prediction CSV", type=["csv"], key="pred")
    std_file  = st.file_uploader("Uncertainty CSV (std or var)", type=["csv"], key="std")

    if true_file and pred_file and std_file:
        Yt = pd.read_csv(true_file)
        Yp = pd.read_csv(pred_file)
        Yu = pd.read_csv(std_file)

        outputs = ["delta_H","delta_D","delta_T"]
        candidates = [c for c in outputs if c in Yt.columns and c in Yp.columns]
        if not candidates:
            st.error("No common output columns found between ground truth and prediction CSVs.")
        else:
            st.markdown("**Confidence interval(s) to display**")
            ci_defs = [
                ("68% CI", 1.0, 1.2, 2, "≈±1σ"),
                ("95% CI", 1.96, 2.0, 3, "Default (≈±1.96σ)"),
                ("99% CI", 2.576, 2.6, 4, "≈±2.58σ"),
            ]
            ci_cols = st.columns(len(ci_defs))
            selected_cis = []
            for ci_idx, (label, z_score, thickness, width, hint) in enumerate(ci_defs):
                with ci_cols[ci_idx]:
                    checked = st.checkbox(
                        label,
                        value=(label == "95% CI"),
                        key=f"ci_{label.replace('%','')}",
                        help=hint,
                    )
                if checked:
                    selected_cis.append((label, z_score, thickness, width))
            if not selected_cis:
                # Fallback to 95% CI if user unchecks everything
                selected_cis.append(ci_defs[1][:4])
            numeric_cols = Yu.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                st.error("No numeric columns in uncertainty CSV."); st.stop()

            metrics = []
            series = {}
            fallback_std = Yu[numeric_cols[0]].to_numpy().astype(float)

            for col in candidates:
                y_true = Yt[col].to_numpy().astype(float)
                y_pred = Yp[col].to_numpy().astype(float)
                if col in Yu.columns:
                    y_std = Yu[col].to_numpy().astype(float)
                else:
                    y_std = fallback_std.copy()

                n = min(len(y_true), len(y_pred), len(y_std))
                y_true, y_pred, y_std = y_true[:n], y_pred[:n], y_std[:n]

                metrics.append(
                    {
                        "Output": col,
                        "R²": r2_score(y_true, y_pred),
                        "MSLL": msll(y_true, y_pred, y_std),
                    }
                )
                series[col] = (y_true, y_pred, y_std)

            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            fig = make_subplots(rows=1, cols=len(candidates), subplot_titles=candidates)
            for idx, col in enumerate(candidates, start=1):
                y_true, y_pred, y_std = series[col]
                diag_min = float(np.min(np.concatenate([y_true, y_pred])))
                diag_max = float(np.max(np.concatenate([y_true, y_pred])))
                pad = 0.05 * (diag_max - diag_min) if diag_max > diag_min else 0.05
                diag_min -= pad
                diag_max += pad
                fig.add_trace(
                    go.Scatter(
                        x=y_true,
                        y=y_pred,
                        mode="markers",
                        marker=dict(size=7, color=BRAND.INDIGO, opacity=0.85),
                        showlegend=False,
                    ),
                    row=1,
                    col=idx,
                )
                for ci_label, z_score, thickness, width in selected_cis:
                    yerr = z_score * np.clip(y_std, 0.0, None)
                    fig.add_trace(
                        go.Scatter(
                            x=y_true,
                            y=y_pred,
                            mode="markers",
                            marker=dict(size=0.1, color="rgba(0,0,0,0)"),
                            error_y=dict(
                                type="data",
                                array=yerr,
                                thickness=thickness,
                                width=width,
                                color=BRAND.INDIGO,
                            ),
                            showlegend=(idx == 1),
                            name=ci_label,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=idx,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=[diag_min, diag_max],
                        y=[diag_min, diag_max],
                        mode="lines",
                        line=dict(color=BRAND.LIGHT_GREY, dash="dash"),
                        showlegend=False,
                    ),
                    row=1,
                    col=idx,
                )
                fig.update_xaxes(title_text="Ground truth", range=[diag_min, diag_max], row=1, col=idx)
                fig.update_yaxes(title_text="Prediction", range=[diag_min, diag_max], scaleanchor=f"x{idx}", scaleratio=1, row=1, col=idx)
            fig.update_layout(height=450, margin=dict(t=60, l=20, r=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for all three CSVs...")

with tab_vis:
    st.subheader("Input mesh grid generator")
    st.markdown(
        f"""
        **Parameter bounds**

        - Pressure: {RANGES["P_kPa"][0]}–{RANGES["P_kPa"][1]} kPa
        - Temperature: {RANGES["T_K"][0]}–{RANGES["T_K"][1]} K
        - Feed moles: H, D, T ∈ [0, {FEED_TOTAL_MOLES[1]}] mol (independent axes)
        """
    )
    p_steps = t_steps = feed_steps = 12
    def build_mesh():
        p_vals = np.linspace(RANGES["P_kPa"][0], RANGES["P_kPa"][1], p_steps)
        t_vals = np.linspace(RANGES["T_K"][0], RANGES["T_K"][1], t_steps)
        feed_vals = np.linspace(0.0, FEED_TOTAL_MOLES[1], feed_steps)
        rows = []
        for P in p_vals:
            for T in t_vals:
                for fH in feed_vals:
                    for fD in feed_vals:
                        for fT in feed_vals:
                            rows.append({"P_kPa": P, "T_K": T, "feed_H": fH, "feed_D": fD, "feed_T": fT})
        return pd.DataFrame(rows)

    mesh_df = build_mesh()
    st.caption(f"Mesh contains **{len(mesh_df):,}** combinations.")
    st.dataframe(mesh_df.head(50), use_container_width=True)
    if len(mesh_df) == 0:
        st.warning("No valid combinations with current settings. Increase the feed range or lower the minimum component.")
    else:
        mesh_df_float16 = mesh_df.astype(np.float16)
        st.download_button(
            "Download mesh grid (CSV)",
            mesh_df_float16.to_csv(index=False).encode("utf-8"),
            file_name="mesh_grid_X.csv",
        )
