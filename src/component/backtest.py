from typing import Dict, List

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .config import BATCH_SIZE, TARGETS, QUANTILES, EMB_DIM_CAP, HIDDEN, LAYERS, DROPOUT
from .datasets import TabularDataset
from .evaluation import eval_split
from .model import MultiTaskQuantileNet
from .training import train_one


def rolling_backtest(
    pdf: pd.DataFrame,
    X_cols_num: List[str],
    X_cols_cat: List[str],
    cat_maps: Dict[str, Dict[str, int]],
    n_folds: int = 3,
    fold_len_days: int = 60,
):
    """
    Rolling-window backtest in time:
      - On each fold, train on history up to 'train_end'
      - Evaluate on [hold_start … hold_end]
    """
    labels_ok = pdf["Y_H1"].notna() & pdf["Y_H2"].notna() & pdf["IDX"].notna()
    tcol = "DAY_FOR_SPLIT" if "DAY_FOR_SPLIT" in pdf.columns else "YM"

    days = pd.to_datetime(pdf.loc[labels_ok, tcol])
    tmax = days.max()

    folds = []
    for i in range(n_folds):
        hold_end = tmax - pd.Timedelta(days=i * fold_len_days)
        hold_start = hold_end - pd.Timedelta(days=fold_len_days - 1)
        train_end = hold_start - pd.Timedelta(days=1)
        folds.append((train_end, hold_start, hold_end))

    results = []
    for k, (train_end, hold_start, hold_end) in enumerate(folds[::-1], 1):
        print(
            f"\n[Fold {k}] train ≤ {train_end.date()} | "
            f"holdout=[{hold_start.date()} … {hold_end.date()}]"
        )

        trn_mask = labels_ok & (pdf[tcol] <= train_end)
        hld_mask = labels_ok & (pdf[tcol] >= hold_start) & (pdf[tcol] <= hold_end)

        print(
            f"[Fold {k}] train rows (labeled)={int(trn_mask.sum()):,} | "
            f"holdout rows (labeled)={int(hld_mask.sum()):,}"
        )

        if hld_mask.sum() == 0:
            print(f"[Fold {k}] No labeled holdout rows – skipping this fold.")
            results.append(dict(fold=k, H1=None, H2=None))
            continue

        ds_trn = TabularDataset(pdf, trn_mask, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col="W_H1")
        ds_hld = TabularDataset(pdf, hld_mask, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col=None)

        model = MultiTaskQuantileNet(
            num_dim=len(X_cols_num),
            cat_cardinals=[len(cat_maps[c]) for c in X_cols_cat],
            emb_cap=EMB_DIM_CAP,
            hidden=HIDDEN,
            layers=LAYERS,
            dropout=DROPOUT,
            n_targets=len(TARGETS),
            n_quants=len(QUANTILES),
        )

        model = train_one(model, ds_trn, ds_trn, QUANTILES)

        dl_h = DataLoader(ds_hld, batch_size=BATCH_SIZE, shuffle=False)
        h1_f = eval_split(model, dl_h, QUANTILES, 0)
        h2_f = eval_split(model, dl_h, QUANTILES, 1)
        results.append(dict(fold=k, H1=h1_f, H2=h2_f))

    return results
