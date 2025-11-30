# src/component/utils.py
from __future__ import annotations
import os, numpy as np, torch
from typing import Tuple
import numpy as np

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def standardize_by_train(X_tr: np.ndarray, X_te: np.ndarray):
    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-9
    return (X_tr - mean) / std, (X_te - mean) / std, mean, std

def to_ppsqft_from_logs(y_log_price: np.ndarray, sqft: np.ndarray) -> np.ndarray:
    price = np.exp(np.clip(y_log_price, -20.0, 20.0))
    sqft_safe = np.clip(sqft.astype(np.float64), 1.0, 1e12)
    return price / sqft_safe
