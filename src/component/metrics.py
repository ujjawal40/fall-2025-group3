import numpy as np


def wape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return np.nan if denom == 0 else np.abs(y_true - y_pred).sum() / denom


def mdape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    pct = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))
    return np.nanmedian(pct)


def pct_within(y_true, y_pred, pct=0.10):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = np.abs(y_pred - y_true) <= (pct * np.abs(y_true))
    return float(np.mean(ok)) if len(ok) else np.nan
