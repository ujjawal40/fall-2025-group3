from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import FIG_DIR


def build_eval_frame_from_run(
    h1: Dict[str, Any],
    h2: Dict[str, Any],
    rolling_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for horizon, metrics in [("H1", h1), ("H2", h2)]:
        rows.append(
            {
                "kind": "holdout",
                "fold": 0,
                "horizon": horizon,
                **metrics,
            }
        )

    for res in rolling_results:
        f = res["fold"]
        for horizon in ["H1", "H2"]:
            m = res[horizon]
            if m is None:
                continue
            rows.append(
                {
                    "kind": "rolling",
                    "fold": f,
                    "horizon": horizon,
                    **m,
                }
            )

    df = pd.DataFrame(rows)
    return df


def _save_fig(fig, fname: str):
    out_path = FIG_DIR / fname
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {out_path}")


def plot_mae_wape_bars(df_eval: pd.DataFrame):
    agg = (
        df_eval.groupby(["kind", "horizon"])[["mae", "wape"]]
        .mean()
        .reset_index()
    )

    horizons = ["H1", "H2"]
    kinds = ["holdout", "rolling"]
    x = np.arange(len(horizons))
    width = 0.35

    for metric, ylabel, fname in [
        ("mae", "MAE (dollars)", "fig_mae_bar.png"),
        ("wape", "WAPE", "fig_wape_bar.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, kind in enumerate(kinds):
            vals = [
                agg[(agg["kind"] == kind) & (agg["horizon"] == h)][metric].values[0]
                for h in horizons
            ]
            ax.bar(x + (i - 0.5) * width, vals, width, label=kind.capitalize())

        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.upper()} – holdout vs rolling")
        ax.legend()
        _save_fig(fig, fname)


def plot_r2_lines(df_eval: pd.DataFrame):
    df_roll = df_eval[df_eval["kind"] == "rolling"].copy()
    df_hold = df_eval[df_eval["kind"] == "holdout"].copy()

    if df_roll.empty:
        print("[viz] No rolling rows for R² plot; skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    folds = sorted(df_roll["fold"].unique())

    for horizon in ["H1", "H2"]:
        sub = df_roll[df_roll["horizon"] == horizon]
        if sub.empty:
            continue

        ax.plot(sub["fold"], sub["r2"], marker="o", label=f"{horizon} (rolling)")

        r2_h = float(df_hold[df_hold["horizon"] == horizon]["r2"].values[0])
        ax.hlines(
            r2_h,
            folds[0],
            folds[-1],
            linestyles="dashed",
            label=f"{horizon} holdout",
        )

    ax.set_xlabel("Fold (rolling window)")
    ax.set_ylabel("R²")
    ax.set_title("R² across rolling folds vs holdout")
    ax.set_xticks(folds)
    ax.set_ylim(0.9, 1.01)
    ax.legend()
    _save_fig(fig, "fig_r2_rolling.png")


def plot_robustness_bars(df_eval: pd.DataFrame):
    df_roll = df_eval[df_eval["kind"] == "rolling"].copy()
    df_hold = df_eval[df_eval["kind"] == "holdout"].copy()

    if df_roll.empty:
        print("[viz] No rolling rows for robustness bar plots; skipping.")
        return

    folds = sorted(df_roll["fold"].unique())
    x = np.arange(len(folds))
    width = 0.5

    for horizon in ["H1", "H2"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sub = df_roll[df_roll["horizon"] == horizon]
        if sub.empty:
            plt.close(fig)
            continue

        wapes = [sub[sub["fold"] == f]["wape"].values[0] for f in folds]
        ax.bar(x, wapes, width)

        wape_h = float(df_hold[df_hold["horizon"] == horizon]["wape"].values[0])
        ax.hlines(
            wape_h,
            -0.5,
            len(folds) - 0.5,
            linestyles="dashed",
            label="Holdout WAPE",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"F{f}" for f in folds])
        ax.set_ylabel("WAPE")
        ax.set_title(f"WAPE per rolling fold ({horizon})")
        ax.legend()
        _save_fig(fig, f"fig_wape_robust_{horizon}.png")


def plot_interval_calibration(df_eval: pd.DataFrame):
    agg = (
        df_eval.groupby(["kind", "horizon"])[["p90_p10_cover", "rel_width"]]
        .mean()
        .reset_index()
    )

    metrics_info = [
        ("p90_p10_cover", "P(y in [p10, p90])", (0.8, 1.01), "fig_pi_cover.png"),
        ("rel_width", "Mean interval width / |median|", (0.0, 1.2), "fig_pi_width.png"),
    ]

    horizons = ["H1", "H2"]
    kinds = ["holdout", "rolling"]
    x = np.arange(len(horizons))
    width = 0.35

    for metric, ylabel, ylim, fname in metrics_info:
        fig, ax = plt.subplots(figsize=(6, 4))

        for i, kind in enumerate(kinds):
            vals = [
                agg[(agg["kind"] == kind) & (agg["horizon"] == h)][metric].values[0]
                for h in horizons
            ]
            ax.bar(x + (i - 0.5) * width, vals, width, label=kind.capitalize())

        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(metric)
        ax.legend()
        _save_fig(fig, fname)


def plot_holdout_radar(h1: Dict[str, Any], h2: Dict[str, Any]):
    metrics = ["mae", "wape", "mdape", "pct10", "p90_p10_cover", "rel_width"]
    labels = [
        "MAE",
        "WAPE",
        "MdAPE",
        "Pct |err|<10%",
        "PI cover",
        "Rel. width",
    ]

    better_high = {"pct10", "p90_p10_cover"}

    data = np.array([[h1[m] for m in metrics], [h2[m] for m in metrics]])

    norm = np.zeros_like(data, dtype=float)
    for j, m in enumerate(metrics):
        col = data[:, j]
        mn, mx = col.min(), col.max()
        if mx == mn:
            norm[:, j] = 0.5
            continue
        if m in better_high:
            norm[:, j] = (col - mn) / (mx - mn)
        else:
            norm[:, j] = 1.0 - (col - mn) / (mx - mn)

    norm = np.concatenate([norm, norm[:, :1]], axis=1)
    angles = np.linspace(0, 2 * np.pi, len(metrics) + 1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, name in enumerate(["H1", "H2"]):
        ax.plot(angles, norm[i], marker="o", label=name)
        ax.fill(angles, norm[i], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Relative performance (H1 vs H2, holdout)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    _save_fig(fig, "fig_holdout_radar.png")
