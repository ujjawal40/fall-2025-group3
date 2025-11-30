# src/component/data_io.py
from __future__ import annotations
import os, numpy as np
from typing import Dict, Any, Tuple
from .config_LME import HParams
from data_preprocessor_LME import DataPreprocessor  # ← reuse your class

def load_with_preprocessor(hp: HParams):
    # Locate CSV (user places it under src/sub_sample.csv as per README)
    for p in ["src/data/sub_sample.csv", "data/src/sub_sample.csv",
              "data/sub_sample.csv", "src/sub_sample.csv", "sub_sample.csv"]:
        if os.path.exists(p):
            csv_path = p; break
    else:
        raise FileNotFoundError("Could not find sub_sample.csv. See README for Box link.")

    pre = DataPreprocessor(dataset_path=csv_path)
    raw_df = pre.load_data()
    clean_df = pre.clean_and_engineer(raw_df)
    if hp.max_rows is not None and len(clean_df) > hp.max_rows:
        clean_df = clean_df.sample(n=hp.max_rows, random_state=hp.random_state)

    X_raw, y_log_price, feature_names, extras = pre.prepare_features(
        clean_df, target="LOG_PRICE", clip_ppsqft_quantile=0.995
    )

    # Try to detect SQFT
    sqft_idx, SQFT_raw = None, None
    sqft_names = {"SQFT","SQUARE_FEET","LIVING_AREA","TOTAL_SQFT","FINISHED_SQ_FT","AREA_SQFT"}
    fn = np.array(extras.get("feature_names", feature_names))
    for nm in fn:
        if str(nm).upper() in sqft_names:
            sqft_idx = int(np.where(fn == nm)[0][0]); break
    if sqft_idx is not None:
        SQFT_raw = X_raw[:, sqft_idx].astype(np.float64)
        extras["SQFT_RAW"] = SQFT_raw

    # Standardize
    x_mean = X_raw.mean(axis=0, keepdims=True)
    x_std  = X_raw.std(axis=0, keepdims=True) + 1e-9
    X_std  = (X_raw - x_mean) / x_std

    extras["x_mean"] = x_mean; extras["x_std"] = x_std; extras["feature_names"] = feature_names
    return X_std.astype(np.float32), y_log_price.astype(np.float32), X_raw.astype(np.float32), extras
