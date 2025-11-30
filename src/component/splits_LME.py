# src/component/splits.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.neighbors import KDTree
from .utils_LME import standardize_by_train

def spatial_grid_split(S_deg: np.ndarray, test_size=0.2, seed=42):
    n = S_deg.shape[0]
    lat, lon = S_deg[:,0], S_deg[:,1]
    lat_bins = np.linspace(lat.min(), lat.max(), 101)
    lon_bins = np.linspace(lon.min(), lon.max(), 101)
    lat_id = np.digitize(lat, lat_bins) - 1
    lon_id = np.digitize(lon, lon_bins) - 1
    cell = lat_id * 100 + lon_id
    uniq = np.unique(cell)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    cut = int(test_size * len(uniq))
    test_cells = set(uniq[:cut])
    train_mask = ~np.isin(cell, list(test_cells))
    test_mask  = ~train_mask
    if train_mask.sum()==0 or test_mask.sum()==0:
        idx = rng.permutation(n); cut = int((1-test_size)*n)
        train_mask = np.zeros(n, bool); train_mask[idx[:cut]] = True
        test_mask  = ~train_mask
    return train_mask, test_mask

def make_inner_spatial_val(S_tr_deg: np.ndarray, frac: float, seed: int):
    lat, lon = S_tr_deg[:,0], S_tr_deg[:,1]
    lat_bins = np.linspace(lat.min(), lat.max(), 61)
    lon_bins = np.linspace(lon.min(), lon.max(), 61)
    lat_id = np.digitize(lat, lat_bins) - 1
    lon_id = np.digitize(lon, lon_bins) - 1
    cell = lat_id * 60 + lon_id
    uniq = np.unique(cell)
    rng = np.random.RandomState(seed+123)
    rng.shuffle(uniq)
    cut = max(1, int(frac * len(uniq)))
    val_cells = set(uniq[:cut])
    val_mask = np.isin(cell, list(val_cells))
    train_inner = ~val_mask
    return train_inner, val_mask

def latlon_to_xy_meters(latlon: np.ndarray, lat0_rad: float) -> np.ndarray:
    R = 6_371_000.0
    lat = np.deg2rad(latlon[:,0]); lon = np.deg2rad(latlon[:,1])
    x = R * (lon - lon.mean()) * np.cos(lat0_rad)
    y = R * (lat - lat0_rad)
    return np.c_[y, x].astype(np.float32)

def build_surface_and_splits(X_std, X_raw, y_log_price, extras, hp):
    spatial = extras.get("spatial")
    spatial_cols = extras.get("spatial_cols", [])
    if spatial is None or spatial.size==0 or "LATITUDE" not in spatial_cols or "LONGITUDE" not in spatial_cols:
        raise RuntimeError("LATITUDE/LONGITUDE not in extras['spatial'].")

    lat_idx = spatial_cols.index("LATITUDE")
    lon_idx = spatial_cols.index("LONGITUDE")
    S_all_deg = spatial[:, [lat_idx, lon_idx]].astype(np.float32)

    valid = ~np.isnan(S_all_deg).any(axis=1)
    X_std = X_std[valid]; X_raw = X_raw[valid]
    y_log_price = y_log_price[valid]; S_all_deg = S_all_deg[valid]

    train_mask, test_mask = spatial_grid_split(S_all_deg, hp.test_size, hp.random_state)

    X_tr, X_te = X_std[train_mask], X_std[test_mask]
    Xr_tr, Xr_te = X_raw[train_mask], X_raw[test_mask]
    y_tr_lp, y_te_lp = y_log_price[train_mask], y_log_price[test_mask]
    S_tr_deg, S_te_deg = S_all_deg[train_mask], S_all_deg[test_mask]

    lat0_rad = float(np.deg2rad(S_tr_deg[:,0]).mean())
    S_tr_m = latlon_to_xy_meters(S_tr_deg, lat0_rad)
    S_te_m = latlon_to_xy_meters(S_te_deg, lat0_rad)
    S_tr_std, S_te_std, s_mean, s_std = standardize_by_train(S_tr_m, S_te_m)

    meta = {"valid_mask": valid, "lat0_rad": lat0_rad, "s_mean": s_mean,
            "s_std": s_std, "spatial_cols": ["LATITUDE","LONGITUDE"],
            "S_tr_deg": S_tr_deg, "S_te_deg": S_te_deg}
    return (X_tr, X_te, Xr_tr, Xr_te, y_tr_lp, y_te_lp,
            S_tr_std, S_te_std, S_tr_deg, S_te_deg, train_mask, test_mask, meta)
