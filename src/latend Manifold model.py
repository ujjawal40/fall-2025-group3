# src/lme_main.py
from __future__ import annotations
import numpy as np, torch

from component.config_LME import HParams, FIG_DIR, RESULTS_DIR
from component.utils_LME import ensure_dirs, set_seeds, to_ppsqft_from_logs
from component.data_io_LME import load_with_preprocessor
from component.splits_LME import build_surface_and_splits, make_inner_spatial_val
from component.surface import KernelSurface, LaplacianOp
from component.model_LME import IntrinsicPriceNet
from component.trainer_LME import LMETrainer
from component.metrics_LME import price_metrics_from_logs, pretty_table, save_metrics_csv
from component.visulization_LME import save_em_plots, save_test_plots, write_results_md


def main():
    # ----------------- Setup -----------------
    hp = HParams()
    set_seeds(hp.random_state)
    ensure_dirs(RESULTS_DIR, FIG_DIR)

    # ----------------- 1) Load -----------------
    X_std_all, y_log_price_all, X_raw_all, extras = load_with_preprocessor(hp)
    print(f"[data] parametric X shape: {X_std_all.shape}, y shape: {y_log_price_all.shape}")

    # ----------------- 2) Split + Surface -----------------
    (X_tr, X_te, Xr_tr, Xr_te, y_tr_lp, y_te_lp,
     S_tr_std, S_te_std, S_tr_deg, S_te_deg,
     train_mask, test_mask, meta) = build_surface_and_splits(
        X_std_all, X_raw_all, y_log_price_all, extras, hp
    )

    # Align SQFT (if present) to 'valid' mask before slicing
    valid = meta["valid_mask"]
    SQFT_all = extras.get("SQFT_RAW", None)
    if SQFT_all is not None:
        SQFT_all = SQFT_all[valid]

    # ----------------- 3) Choose target -----------------
    if SQFT_all is not None:
        SQFT_tr, SQFT_te = SQFT_all[train_mask], SQFT_all[test_mask]
        y_tr = (y_tr_lp - np.log(np.clip(SQFT_tr, 1.0, 1e12))).astype(np.float32)
        y_te = (y_te_lp - np.log(np.clip(SQFT_te, 1.0, 1e12))).astype(np.float32)

        # sample weights by PPSQFT quantiles
        ppsqft_tr = to_ppsqft_from_logs(y_tr_lp, SQFT_tr)
        q50, q90 = np.quantile(ppsqft_tr, [0.5, 0.9])
        w_tr_full = np.ones_like(ppsqft_tr, dtype=np.float32)
        w_tr_full[ppsqft_tr >= q50] *= 1.2
        w_tr_full[ppsqft_tr >= q90] *= 1.6

        target_name = "PPSQFT"
    else:
        print("[warn] SQFT missing/unusable; using LOG_PRICE target.")
        y_tr, y_te = y_tr_lp.astype(np.float32), y_te_lp.astype(np.float32)
        w_tr_full = np.ones_like(y_tr, dtype=np.float32)
        target_name = "LOG_PRICE"

    # ----------------- 4) Inner spatial VAL -----------------
    tr_inner_mask, val_mask = make_inner_spatial_val(
        S_tr_deg, frac=hp.inner_val_frac, seed=hp.random_state
    )
    X_tr_in, X_val = X_tr[tr_inner_mask], X_tr[val_mask]
    y_tr_in, y_val = y_tr[tr_inner_mask], y_tr[val_mask]
    w_tr_in = w_tr_full[tr_inner_mask]
    S_tr_in_std, S_val_std = S_tr_std[tr_inner_mask], S_tr_std[val_mask]

    # ----------------- 5) Model + Trainer -----------------
    model = IntrinsicPriceNet(in_dim=X_tr.shape[1], hidden=hp.hidden_layers, dropout_prob=hp.dropout_prob)
    surf_in = KernelSurface(S_tr_in_std, K=hp.K, q=hp.q)
    lap_in  = LaplacianOp(surf_in, K_lap=hp.K_lap or hp.K, q=hp.q) if hp.lap_lambda > 0 else None

    trainer = LMETrainer(
        X_tr_in, y_tr_in, w_tr_in,
        S_tr_in_std, S_val_std,
        model, surf_in, lap_in, hp
    )

    # ----------------- 6) Fit -----------------
    history = trainer.fit(X_val=X_val, y_val=y_val)

    # ----------------- 7) Single E-step on FULL TRAIN -----------------
    surf_full = KernelSurface(S_tr_std, K=hp.K, q=hp.q)
    lap_full  = LaplacianOp(surf_full, K_lap=hp.K_lap or hp.K, q=hp.q) if hp.lap_lambda > 0 else None

    tmp = LMETrainer(
        X_tr, y_tr, w_tr_full,
        S_tr_std, S_tr_std[:1],  # dummy VAL surface
        model, surf_full, lap_full, hp
    )
    tmp.U_list = surf_full.build_U_list()
    tmp._update_D()

    # ----------------- Predictions & Metrics -----------------
    y_pred_tr = tmp.predict(X_tr, S_tr_std).astype(np.float64)
    y_pred_te = tmp.predict(X_te, S_te_std).astype(np.float64)

    tr_metrics = price_metrics_from_logs(y_tr, y_pred_tr)
    te_metrics = price_metrics_from_logs(y_te, y_pred_te)

    print(pretty_table(f"Paper-style metrics (TRAIN, {target_name})", tr_metrics))
    print(pretty_table(f"Paper-style metrics (TEST, {target_name})",  te_metrics))

    # ----------------- 8) Outputs (plots, md, CSV) -----------------
    save_em_plots(history)
    save_test_plots(y_te, y_pred_te, target_name)
    write_results_md(target_name, tr_metrics, te_metrics)

    # Save CSVs: per-run file + cumulative log
    extra = {
        "em_final_loss": history["em_losses"][-1] if history.get("em_losses") else None,
        "em_best_loss": min(history["em_losses"]) if history.get("em_losses") else None,
    }
    csv_path = save_metrics_csv(
        hp=hp,
        target_name=target_name,
        tr_metrics=tr_metrics,
        te_metrics=te_metrics,
        extra=extra
    )
    print(f"[results] metrics saved to: {csv_path}")
    print(f"[results] cumulative log:  {RESULTS_DIR}/lme_eval_metrics_log.csv")


if __name__ == "__main__":
    main()
