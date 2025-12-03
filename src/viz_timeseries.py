"""
viz_zipmonth_tabmtl.py

Diagnostics and visualization for the ZIP×month latent index model.

This module converts the scalar evaluation summaries produced by the
multi-horizon quantile MLP into a compact set of figures that mirror
the typical "Results" section of a NeurIPS paper:

  • cross-fold accuracy (MAE, WAPE),
  • explained variance (R²),
  • robustness to outliers (MdAPE, pct10),
  • probabilistic calibration and sharpness (p90–p10 coverage vs width),
  • horizon-wise summary (H1 vs H2) on the final holdout.

All functions are pure: they take metric dicts as input and optionally
save figures to disk, but never depend on global training state.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers: convert raw dicts → tidy DataFrames
# ---------------------------------------------------------------------

def _metrics_to_frame(
    name: str,
    h1: Dict[str, float],
    h2: Dict[str, float]
) -> pd.DataFrame:
    """
    Convert a pair of metric dicts (H1, H2) into a single tidy DataFrame.

    Parameters
    ----------
    name : str
        Label for the evaluation split (e.g. 'holdout', 'fold1').
    h1, h2 : dict
        Metric dictionaries for horizon 1 and horizon 2 respectively.

    Returns
    -------
    df : pd.DataFrame
        Columns: ['split', 'horizon', 'mae', 'r2', 'wape', 'mdape',
                  'pct10', 'p90_p10_cover', 'rel_width'].
    """
    rows = []
    for horizon_name, metrics in [("H1", h1), ("H2", h2)]:
        if metrics is None:
            continue
        row = {"split": name, "horizon": horizon_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def build_eval_frames(
    holdout_h1: Dict[str, float],
    holdout_h2: Dict[str, float],
    rolling_results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Aggregate holdout and rolling-backtest metrics into a single tidy table.

    Parameters
    ----------
    holdout_h1, holdout_h2 : dict
        Metric summaries for the final holdout month(s), for H1 and H2.
    rolling_results : list of dict
        Output of rolling_backtest(...). Each element has keys:
        {'fold': int, 'H1': dict or None, 'H2': dict or None}.

    Returns
    -------
    df_all : pd.DataFrame
        One row per (split, horizon) pair, where "split" includes both
        named folds ('fold1', 'fold2', ...) and 'holdout'.
    """
    frames = []

    # Final holdout
    frames.append(_metrics_to_frame("holdout", holdout_h1, holdout_h2))

    # Rolling folds
    for r in rolling_results:
        split_name = f"fold{r['fold']}"
        frames.append(_metrics_to_frame(split_name, r["H1"], r["H2"]))

    df_all = pd.concat(frames, ignore_index=True)
    # Enforce a stable categorical ordering: folds in chronological order, then holdout
    order = sorted([s for s in df_all["split"].unique() if s.startswith("fold")],
                   key=lambda x: int(x.replace("fold", "")))
    if "holdout" in df_all["split"].unique():
        order.append("holdout")
    df_all["split"] = pd.Categorical(df_all["split"], categories=order, ordered=True)
    return df_all


# ---------------------------------------------------------------------
# Figure 1: MAE and WAPE by split & horizon
# ---------------------------------------------------------------------

def plot_mae_wape_bars(
    df_all: pd.DataFrame,
    savepath: Optional[str] = None
) -> plt.Figure:
    """
    Bar plots of absolute error (MAE) and relative error (WAPE)
    for each temporal split and horizon.

    This figure corresponds to the standard "overall accuracy" panel in
    a NeurIPS results section.

    Parameters
    ----------
    df_all : pd.DataFrame
        Output of build_eval_frames(...).
    savepath : str, optional
        If provided, path to save the figure as a PNG/PDF.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    splits = df_all["split"].cat.categories
    horizons = ["H1", "H2"]

    x = np.arange(len(splits))
    width = 0.35

    # Prepare series
    def get_series(metric: str, horizon: str) -> np.ndarray:
        sub = df_all[df_all["horizon"] == horizon].set_index("split").reindex(splits)
        return sub[metric].values.astype(float)

    mae_h1 = get_series("mae", "H1")
    mae_h2 = get_series("mae", "H2")
    wape_h1 = get_series("wape", "H1")
    wape_h2 = get_series("wape", "H2")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("Overall Accuracy Across Temporal Splits", fontsize=14)

    ax = axes[0]
    ax.bar(x - width / 2, mae_h1, width, label="H1 (MAE)")
    ax.bar(x + width / 2, mae_h2, width, label="H2 (MAE)")
    ax.set_ylabel("MAE (dollars)")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    ax.bar(x - width / 2, wape_h1 * 100.0, width, label="H1 (WAPE)")
    ax.bar(x + width / 2, wape_h2 * 100.0, width, label="H2 (WAPE)")
    ax.set_ylabel("WAPE (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in splits])
    ax.legend()
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------
# Figure 2: R² by split & horizon
# ---------------------------------------------------------------------

def plot_r2_lines(
    df_all: pd.DataFrame,
    savepath: Optional[str] = None
) -> plt.Figure:
    """
    Line plot of explained variance (R²) across folds and holdout.

    Parameters
    ----------
    df_all : pd.DataFrame
        Output of build_eval_frames(...).
    savepath : str, optional
        If provided, path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    splits = df_all["split"].cat.categories
    x = np.arange(len(splits))

    def get_series(horizon: str) -> np.ndarray:
        sub = df_all[df_all["horizon"] == horizon].set_index("split").reindex(splits)
        return sub["r2"].values.astype(float)

    r2_h1 = get_series("H1")
    r2_h2 = get_series("H2")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, r2_h1, marker="o", label="H1")
    ax.plot(x, r2_h2, marker="s", label="H2")
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in splits])
    ax.set_ylabel("$R^2$")
    ax.set_title("Explained Variance Across Temporal Splits")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------
# Figure 3: Robustness metrics – MdAPE and pct10
# ---------------------------------------------------------------------

def plot_robustness_bars(
    df_all: pd.DataFrame,
    savepath: Optional[str] = None
) -> plt.Figure:
    """
    Two-panel bar plot for MdAPE and pct10.

    MdAPE reflects robustness to outliers; pct10 is the fraction of
    predictions within ±10% relative error.

    Parameters
    ----------
    df_all : pd.DataFrame
        Output of build_eval_frames(...).
    savepath : str, optional
        If provided, path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    splits = df_all["split"].cat.categories
    x = np.arange(len(splits))
    width = 0.35

    def get_series(metric: str, horizon: str) -> np.ndarray:
        sub = df_all[df_all["horizon"] == horizon].set_index("split").reindex(splits)
        return sub[metric].values.astype(float)

    mdape_h1 = get_series("mdape", "H1") * 100.0
    mdape_h2 = get_series("mdape", "H2") * 100.0
    pct10_h1 = get_series("pct10", "H1") * 100.0
    pct10_h2 = get_series("pct10", "H2") * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("Robustness to Outliers and Tight-Error Mass", fontsize=14)

    ax = axes[0]
    ax.bar(x - width / 2, mdape_h1, width, label="H1")
    ax.bar(x + width / 2, mdape_h2, width, label="H2")
    ax.set_ylabel("MdAPE (%)")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    ax.bar(x - width / 2, pct10_h1, width, label="H1")
    ax.bar(x + width / 2, pct10_h2, width, label="H2")
    ax.set_ylabel("Pct. within ±10%")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in splits])
    ax.legend()
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------
# Figure 4: Interval calibration vs sharpness
# ---------------------------------------------------------------------

def plot_interval_calibration(
    df_all: pd.DataFrame,
    savepath: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot of p90–p10 coverage vs relative interval width.

    Each point corresponds to a (split, horizon) pair. Ideally, the
    model lies near the upper-left region: high coverage with narrow
    intervals.

    Parameters
    ----------
    df_all : pd.DataFrame
        Output of build_eval_frames(...).
    savepath : str, optional
        If provided, path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))

    for horizon, marker in [("H1", "o"), ("H2", "s")]:
        sub = df_all[df_all["horizon"] == horizon].copy()
        ax.scatter(
            sub["rel_width"],
            sub["p90_p10_cover"],
            label=horizon,
            marker=marker,
            alpha=0.8,
        )
        # annotate with split names for interpretability
        for _, row in sub.iterrows():
            ax.annotate(str(row["split"]), (row["rel_width"], row["p90_p10_cover"]),
                        textcoords="offset points", xytext=(3, 3), fontsize=7)

    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Relative Interval Width (|p90 − p10| / |median|)")
    ax.set_ylabel("Empirical Coverage  P( y ∈ [p10,p90] )")
    ax.set_title("Calibration vs Sharpness (by Split and Horizon)")
    ax.grid(alpha=0.2)
    ax.legend(title="Horizon")
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------
# Figure 5: Horizon comparison radar – holdout only
# ---------------------------------------------------------------------

def plot_holdout_radar(
    holdout_h1: Dict[str, float],
    holdout_h2: Dict[str, float],
    savepath: Optional[str] = None
) -> plt.Figure:
    """
    Radar (spider) chart comparing H1 vs H2 on the final holdout.

    We invert "lower is better" metrics so that a larger radius always
    corresponds to better performance.

    Parameters
    ----------
    holdout_h1, holdout_h2 : dict
        Metric summaries for the final holdout.
    savepath : str, optional
        If provided, path to save figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Metrics to summarize
    metrics = ["mae", "wape", "r2", "mdape", "pct10", "p90_p10_cover", "rel_width"]

    def transform_for_display(mdict: Dict[str, float]) -> List[float]:
        vals = []
        for m in metrics:
            v = float(mdict[m])
            # For error/width metrics, invert so larger = better
            if m in ["mae", "wape", "mdape", "rel_width"]:
                vals.append(-v)
            else:
                vals.append(v)
        return vals

    v1 = np.array(transform_for_display(holdout_h1))
    v2 = np.array(transform_for_display(holdout_h2))

    # Normalize to [0,1] across both horizons for visual comparability
    stacked = np.vstack([v1, v2])
    v_min = stacked.min(axis=0)
    v_max = stacked.max(axis=0)
    denom = np.where(v_max - v_min > 0, v_max - v_min, 1.0)
    v1_n = (v1 - v_min) / denom
    v2_n = (v2 - v_min) / denom

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    # close polygon
    angles = np.concatenate([angles, angles[:1]])
    v1_plot = np.concatenate([v1_n, v1_n[:1]])
    v2_plot = np.concatenate([v2_n, v2_n[:1]])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(angles, v1_plot, label="H1")
    ax.fill(angles, v1_plot, alpha=0.15)
    ax.plot(angles, v2_plot, label="H2")
    ax.fill(angles, v2_plot, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Final Holdout Summary: H1 vs H2", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig
