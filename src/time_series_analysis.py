import json
import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch

from component.config import (
    RUN_DIR,
    MIN_START_DATE,
    HOLDOUT_DAYS,
    VAL_FRACTION_OF_TRAIN_TIME,
    CATEGORICAL_COLS,
    NON_FEATURE_KEYS,
    LABEL_COLS,
    FUTUREISH_COLS,
    TARGETS,
    QUANTILES,
    BATCH_SIZE,
    EMB_DIM_CAP,
    HIDDEN,
    LAYERS,
    DROPOUT,
)
from component.data_io import load_raw_and_build_features
from component.datasets import TabularDataset
from component.model import MultiTaskQuantileNet
from component.training import train_one
from component.evaluation import eval_split, suppression_report
from component.backtest import rolling_backtest
from component.visualization import (
    build_eval_frame_from_run,
    plot_mae_wape_bars,
    plot_r2_lines,
    plot_robustness_bars,
    plot_interval_calibration,
    plot_holdout_radar,
)

# ============================================
# LOAD DATA + FEATURE MATRIX
# ============================================
combined_events, feat_df = load_raw_and_build_features()

# ============================================
# PULL feat_df INTO TRAINING DF, SPLIT, WINSORIZE
# (this mirrors your original notebook cell)
# ============================================
START_DATE = MIN_START_DATE

pdf = feat_df.copy()

if "EVT_IS_RENTAL" in pdf.columns:
    pdf = pdf.drop(columns=["EVT_IS_RENTAL"])

date_col = "DAY_FOR_SPLIT" if "DAY_FOR_SPLIT" in pdf.columns else "YM"
pdf[date_col] = pd.to_datetime(pdf[date_col], errors="coerce")

mask_start = pdf[date_col] >= pd.to_datetime(START_DATE)

fat_text = [c for c in ("URL", "STREETADDRESS", "DESCRIPTION") if c in pdf.columns]
base = pdf.loc[mask_start, [c for c in pdf.columns if c not in fat_text]].copy()

diag_min = base[date_col].min()
diag_max = base[date_col].max()
print(f"[CELL11:DIAG] {date_col} range: {diag_min} .. {diag_max} | rows={len(base):,}")

pdf = base.copy()

for c in ("YM", "DAY_FOR_SPLIT"):
    if c in pdf.columns:
        pdf[c] = pd.to_datetime(pdf[c], errors="coerce")

tcol = "DAY_FOR_SPLIT" if "DAY_FOR_SPLIT" in pdf.columns else "YM"
max_evt_day = pd.to_datetime(pdf[tcol]).max()
max_evt_day = pd.Timestamp(max_evt_day)
effective_train_end = max_evt_day - pd.Timedelta(days=HOLDOUT_DAYS)
holdout_start = effective_train_end + pd.Timedelta(days=1)
print(
    f"[CELL11:SPLIT] max_day={max_evt_day.date()} | train_end={effective_train_end.date()} "
    f"| holdout=[{holdout_start.date()} … {max_evt_day.date()}]"
)

if "DAY_FOR_SPLIT" in pdf.columns:
    trn_mask = pdf["DAY_FOR_SPLIT"] <= effective_train_end - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    hld_mask = ~trn_mask
else:
    trn_mask = pdf["YM"] <= effective_train_end
    hld_mask = pdf["YM"] > effective_train_end

df_trn = pdf.loc[trn_mask].copy()


def _winsor_train_only(df_trn_small: pd.DataFrame, df_full: pd.DataFrame, ycol: str, k: float) -> pd.DataFrame:
    key = ["STATE_MODE", "YM"] if "STATE_MODE" in df_full.columns else ["YM"]
    fences = {}
    for gk, g in df_trn_small.groupby(key, dropna=False):
        s = pd.to_numeric(g[ycol], errors="coerce")
        if s.notna().sum() == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        fences[gk] = (q1 - k * iqr, q3 + k * iqr)

    def clamp(row):
        gk = tuple(row[k] if k in df_full.columns else np.nan for k in key)
        lo, hi = fences.get(gk, (-np.inf, np.inf))
        v = row[ycol]
        return np.clip(v, lo, hi) if pd.notna(v) else v

    df_full[ycol + "_WZ"] = df_full.apply(clamp, axis=1)
    return df_full


for ycol, k in [("Y_H1", 1.5), ("Y_H2", 3.0)]:
    if ycol in pdf.columns:
        cols_small = ["YM", ycol]
        if {"STATE_MODE", "YM"}.issubset(df_trn.columns):
            cols_small = ["STATE_MODE", "YM", ycol]
        pdf = _winsor_train_only(df_trn[cols_small], pdf, ycol, k)

all_cols = set(pdf.columns)
drop_never = NON_FEATURE_KEYS | LABEL_COLS | FUTUREISH_COLS
cand_feats = sorted([c for c in all_cols if c not in drop_never])

for c in cand_feats:
    if c in CATEGORICAL_COLS:
        continue
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

pdf["W_H1"] = 1.0
if "N_SOLD" in pdf.columns:
    pdf.loc[trn_mask & pdf["N_SOLD"].notna(), "W_H1"] = 1.0

if "W_H1_COMBINED" in pdf.columns:
    pdf["W_H1_COMBINED"] = (
        pd.to_numeric(pdf["W_H1_COMBINED"], errors="coerce")
        .clip(0.2, 1.0)
        .fillna(1.0)
    )
    pdf["W_H1"] = pdf["W_H1"] * pdf["W_H1_COMBINED"]

if "DAY_FOR_SPLIT" in pdf.columns:
    trange = pdf.loc[trn_mask, "DAY_FOR_SPLIT"]
else:
    trange = pdf.loc[trn_mask, "YM"]

t0, t1 = pd.to_datetime(trange.min()), pd.to_datetime(trange.max())
cut = t0 + (t1 - t0) * (1 - VAL_FRACTION_OF_TRAIN_TIME)

if "DAY_FOR_SPLIT" in pdf.columns:
    trn_in = (pdf["DAY_FOR_SPLIT"] <= cut) & trn_mask
    val_in_tmp = (pdf["DAY_FOR_SPLIT"] > cut) & trn_mask
else:
    trn_in = (pdf["YM"] <= cut) & trn_mask
    val_in_tmp = (pdf["YM"] > cut) & trn_mask

cat_maps = {}
for c in [c for c in CATEGORICAL_COLS if c in pdf.columns]:
    vals = pd.Index(pdf.loc[trn_mask, c].astype("string").fillna("<NA>").unique())
    cat_maps[c] = {v: i + 1 for i, v in enumerate(vals)}
    for split_mask in [trn_in, val_in_tmp, hld_mask]:
        pdf.loc[split_mask, c] = pdf.loc[split_mask, c].astype("string").fillna("<NA>")

num_cols = [c for c in cand_feats if c not in CATEGORICAL_COLS]
scaler = StandardScaler()
if len(num_cols) > 0 and np.sum(trn_in) > 0:
    scaler.fit(pdf.loc[trn_in, num_cols])
    for split_mask in [trn_in, val_in_tmp, hld_mask]:
        pdf.loc[split_mask, num_cols] = scaler.transform(pdf.loc[split_mask, num_cols])

X_cols_num = num_cols
X_cols_cat = [c for c in CATEGORICAL_COLS if c in pdf.columns]

print(
    f"[CELL11:DONE] rows={len(pdf):,} | F_num={len(X_cols_num)} | "
    f"F_cat={len(X_cols_cat)} | totalF={len(X_cols_num)+len(X_cols_cat)} "
    f"| dropped_text={len(fat_text)}"
)

# ============================================
# FIT NATIONAL MODEL, EVAL HOLDOUT, SAVE
# (exact same splitting logic as original)
# ============================================
pdf["YM"] = pd.to_datetime(pdf["YM"])
if "DAY_FOR_SPLIT" in pdf.columns:
    pdf["DAY_FOR_SPLIT"] = pd.to_datetime(pdf["DAY_FOR_SPLIT"])
else:
    pdf["DAY_FOR_SPLIT"] = pdf["YM"]

has_labels = pdf["Y_H1"].notna() & pdf["Y_H2"].notna() & pdf["IDX"].notna()

last_ym = pdf.loc[has_labels, "YM"].max()
m0 = last_ym
m1 = last_ym - relativedelta(months=1)
m2 = last_ym - relativedelta(months=2)
m3 = last_ym - relativedelta(months=3)

hld_mask = has_labels & (pdf["YM"].isin([m0, m1]))
val_in = has_labels & (pdf["YM"].isin([m2, m3]))
trn_in = has_labels & (pdf["YM"] < m3)

if trn_in.sum() < 0.25 * has_labels.sum():
    m0 = last_ym
    m1 = last_ym - relativedelta(months=1)
    hld_mask = has_labels & (pdf["YM"] == m0)
    val_in = has_labels & (pdf["YM"] == m1)
    trn_in = has_labels & (pdf["YM"] < m1)

n_trn = int(trn_in.sum())
n_val = int(val_in.sum())
n_hld = int(hld_mask.sum())
steps_trn = int(np.ceil(n_trn / BATCH_SIZE)) if n_trn else 0
steps_val = int(np.ceil(n_val / BATCH_SIZE)) if n_val else 0
print(
    f"[CELL11:DATA] train={n_trn:,} ({steps_trn} steps/epoch) | "
    f"val={n_val:,} ({steps_val} steps) | holdout={n_hld:,}"
)

cat_cardinals = [len(cat_maps[c]) for c in X_cols_cat]

ds_trn = TabularDataset(pdf, trn_in, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col="W_H1")
ds_val = TabularDataset(pdf, val_in, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col="W_H1")
ds_hld = TabularDataset(pdf, hld_mask, X_cols_num, X_cols_cat, cat_maps, TARGETS, weights_col=None)

dl_trn = DataLoader(ds_trn, batch_size=BATCH_SIZE, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

print(
    f"[CELL11:DIMS] F_num={len(X_cols_num)} | F_cat={len(X_cols_cat)} | "
    f"train/val/hld={len(ds_trn)}/{len(ds_val)}/{len(ds_hld)}"
)

model = MultiTaskQuantileNet(
    num_dim=len(X_cols_num),
    cat_cardinals=cat_cardinals,
    emb_cap=EMB_DIM_CAP,
    hidden=HIDDEN,
    layers=LAYERS,
    dropout=DROPOUT,
    n_targets=len(TARGETS),
    n_quants=len(QUANTILES),
)

model = train_one(model, ds_trn, ds_val, QUANTILES)

dl_hld = DataLoader(ds_hld, batch_size=BATCH_SIZE, shuffle=False)
h1 = eval_split(model, dl_hld, QUANTILES, head_ix=0)
h2 = eval_split(model, dl_hld, QUANTILES, head_ix=1)

print("\n=== HOLDOUT METRICS ===")
print("H1:", h1)
print("H2:", h2)

print("\n=== SUPPRESSION (confidence gating) ===")
print("H1:", suppression_report(model, ds_hld, QUANTILES, head_ix=0))
print("H2:", suppression_report(model, ds_hld, QUANTILES, head_ix=1))

ts = time.strftime("%Y%m%d-%H%M%S")
runname = "zipmonth_tabmtl_quantile"

train_max_ym = pd.to_datetime(pdf.loc[trn_in, "YM"]).max() if n_trn else pd.NaT
val_ym_range = pd.to_datetime(pdf.loc[val_in, "YM"]).sort_values().unique() if n_val else []
hld_ym_range = pd.to_datetime(pdf.loc[hld_mask, "YM"]).sort_values().unique() if n_hld else []

split_meta = dict(
    last_ym=str(last_ym.date()) if pd.notna(last_ym) else None,
    train_max_ym=str(train_max_ym.date()) if pd.notna(train_max_ym) else None,
    val_months=[str(pd.to_datetime(x).date()) for x in val_ym_range],
    holdout_months=[str(pd.to_datetime(x).date()) for x in hld_ym_range],
    sizes=dict(train=n_trn, val=n_val, holdout=n_hld),
)

torch.save(
    {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": dict(
            HIDDEN=HIDDEN,
            LAYERS=LAYERS,
            DROPOUT=DROPOUT,
            QUANTILES=QUANTILES,
            CATEGORICAL_COLS=X_cols_cat,
            NUMERIC_COLS=X_cols_num,
        ),
        "cat_maps": cat_maps,
        "scaler_mean": scaler.mean_.tolist() if len(num_cols) > 0 else [],
        "scaler_scale": scaler.scale_.tolist() if len(num_cols) > 0 else [],
        "split": split_meta,
        "metrics_holdout": {"H1": h1, "H2": h2},
    },
    RUN_DIR / f"{ts}__{runname}.pt",
)

with open(RUN_DIR / f"{ts}__{runname}.json", "w") as f:
    json.dump(
        {
            "metrics_holdout": {"H1": h1, "H2": h2},
            "split": split_meta,
            "shapes": dict(
                train=len(ds_trn),
                val=len(ds_val),
                holdout=len(ds_hld),
                F_num=len(X_cols_num),
                F_cat=len(X_cols_cat),
            ),
        },
        f,
        indent=2,
    )

print("Saved:", RUN_DIR / f"{ts}__{runname}.pt")

n_trn = len(ds_trn)
n_val = len(ds_val)
n_hld = len(ds_hld)
steps_trn = int(np.ceil(n_trn / BATCH_SIZE)) if n_trn else 0
steps_val = int(np.ceil(n_val / BATCH_SIZE)) if n_val else 0
print(
    f"[CELL11:DATA] train={n_trn:,} ({steps_trn} steps/epoch) | "
    f"val={n_val:,} ({steps_val} steps) | holdout={n_hld:,}"
)

# ============================================
# ROLLING BACKTEST + VIZ
# ============================================
rolling_results = rolling_backtest(pdf, X_cols_num, X_cols_cat, cat_maps, n_folds=3, fold_len_days=60)
print("\nRolling backtest results:", rolling_results)

eval_df = build_eval_frame_from_run(h1, h2, rolling_results)
print("\n[viz] eval_df head:")
print(eval_df.head())

# ---- SAVE EVAL DF AS CSV IN results/ ----
from component.config import RESULTS_DIR
eval_csv_path = RESULTS_DIR / f"{ts}__zipmonth_eval_metrics.csv"
eval_df.to_csv(eval_csv_path, index=False)
print(f"[eval] saved metrics CSV → {eval_csv_path}")

# ---- PLOTS ----
plot_mae_wape_bars(eval_df)
plot_r2_lines(eval_df)
plot_robustness_bars(eval_df)
plot_interval_calibration(eval_df)
plot_holdout_radar(h1, h2)
