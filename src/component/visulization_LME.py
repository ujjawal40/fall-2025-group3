# src/component/visualization.py
from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt, os
from typing import Dict, List
from .config import FIG_DIR
from .utils_LME import ensure_dirs

def save_em_plots(history: Dict[str, List[float]]):
    ensure_dirs(FIG_DIR)
    if history.get("em_losses"):
        plt.figure(figsize=(6,4))
        plt.plot(history["em_losses"], marker="o")
        plt.title("EM iteration loss (inner-train energy)")
        plt.xlabel("EM iteration"); plt.ylabel("loss"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{FIG_DIR}/em_loss.png", dpi=150)
    if history.get("mstep_losses_per_iter"):
        plt.figure(figsize=(6,4))
        for i, losses in enumerate(history["mstep_losses_per_iter"]):
            plt.plot(losses, label=f"EM {i+1}")
        plt.title("M-step (NN) losses per EM (inner-train)")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{FIG_DIR}/mstep_losses.png", dpi=150)

def save_test_plots(y_te_log, y_pred_te_log, target_name: str):
    ensure_dirs(FIG_DIR)
    clip = 20.0
    y_true = np.exp(np.clip(y_te_log, -clip, clip))
    y_pred = np.exp(np.clip(y_pred_te_log, -clip, clip))
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=4)
    mn, mx = y_true.min(), y_true.max()
    plt.plot([mn, mx], [mn, mx], color="red")
    plt.title(f"Predicted vs True ({target_name}, TEST)")
    plt.xlabel("True"); plt.ylabel("Pred"); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/pred_vs_true_test.png", dpi=150)

    abs_rel = np.abs(y_pred - y_true) / (y_true + 1e-12)
    plt.figure(figsize=(6,4))
    plt.hist(abs_rel, bins=50)
    plt.title(f"Absolute Relative Error ({target_name}, TEST)")
    plt.xlabel("abs_rel"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/abs_rel_hist_test.png", dpi=150)

def write_results_md(target_name: str, tr_metrics: Dict[str,float], te_metrics: Dict[str,float], out_path="results/results_summary.md"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Latent Manifold Estimation — Results Summary\n\n")
        f.write(f"**Target:** {target_name}\n\n")
        def block(title, m):
            f.write(f"## {title}\n\n| Metric | Value |\n|---|---|\n")
            f.write(f"| < 5% within | {m['within_5']*100:.2f}% |\n")
            f.write(f"| < 10% within | {m['within_10']*100:.2f}% |\n")
            f.write(f"| < 15% within | {m['within_15']*100:.2f}% |\n")
            f.write(f"| Median abs rel | {m['median_abs_rel']:.4f} |\n\n")
        block("Train Metrics", tr_metrics)
        block("Test Metrics", te_metrics)
        f.write("## Figures\n\n")
        f.write("- `figs/em_loss.png`\n- `figs/mstep_losses.png`\n- `figs/pred_vs_true_test.png`\n- `figs/abs_rel_hist_test.png`\n")
