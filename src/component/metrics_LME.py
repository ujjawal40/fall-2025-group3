# src/component/metrics.py
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Dict, Any, Optional
from .config_LME import RESULTS_DIR, HParams
import numpy as np
import pandas as pd
import os
import csv

def price_metrics_from_logs(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> Dict[str, float]:
    clip = 20.0
    y_true = np.exp(np.clip(y_log_true, -clip, clip))
    y_pred = np.exp(np.clip(y_log_pred, -clip, clip))
    abs_rel = np.abs(y_pred - y_true) / (y_true + 1e-12)
    return {
        "within_5": float((abs_rel < 0.05).mean()),
        "within_10": float((abs_rel < 0.10).mean()),
        "within_15": float((abs_rel < 0.15).mean()),
        "median_abs_rel": float(np.median(abs_rel)),
    }

def pretty_table(title: str, metrics: Dict[str, float]) -> str:
    rows = [
        ("< 5% within",  f"{metrics['within_5']*100:6.2f}%"),
        ("< 10% within", f"{metrics['within_10']*100:6.2f}%"),
        ("< 15% within", f"{metrics['within_15']*100:6.2f}%"),
        ("Median abs rel", f"{metrics['median_abs_rel']:.4f}"),
    ]
    width = max(len(r[0]) for r in rows) + 2
    s = [f"\n{title}\n" + "-" * len(title)]
    s += [f"{k:<{width}} {v}" for k, v in rows]
    return "\n".join(s)

def _flat_metrics(prefix: str, m: Dict[str, float]) -> Dict[str, Any]:
    return {
        f"{prefix}_within_5":  m.get("within_5", 0.0),
        f"{prefix}_within_10": m.get("within_10", 0.0),
        f"{prefix}_within_15": m.get("within_15", 0.0),
        f"{prefix}_median_abs_rel": m.get("median_abs_rel", 0.0),
    }

def _hp_to_dict(hp: HParams) -> Dict[str, Any]:
    # turns dataclass into a flat dict with strings where needed
    d = asdict(hp) if is_dataclass(hp) else dict(hp)
    # make hidden layers readable in CSV
    if "hidden_layers" in d and isinstance(d["hidden_layers"], tuple):
        d["hidden_layers"] = "-".join(str(x) for x in d["hidden_layers"])
    return d

def save_metrics_csv(
    hp: HParams,
    target_name: str,
    tr_metrics: Dict[str, float],
    te_metrics: Dict[str, float],
    *,
    extra: Optional[Dict[str, Any]] = None,
    results_dir: str = RESULTS_DIR,
    basename: str = "lme_eval_metrics"
) -> str:
    """
    Writes two CSVs:
      1) results/<timestamp>_<basename>.csv        (one row for this run)
      2) results/<basename>_log.csv (appended)     (cumulative log)
    Returns the path of the per-run CSV.
    """
    os.makedirs(results_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    row: Dict[str, Any] = {
        "run_id": ts,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target": target_name,
    }
    row.update(_hp_to_dict(hp))
    row.update(_flat_metrics("train", tr_metrics))
    row.update(_flat_metrics("test", te_metrics))
    if extra:
        row.update(extra)

    per_run_path = os.path.join(results_dir, f"{ts}_{basename}.csv")
    log_path     = os.path.join(results_dir, f"{basename}_log.csv")

    # write per-run file (with header)
    _write_csv_with_header(per_run_path, [row])

    # append into cumulative log (create with header if missing)
    _append_csv_with_header(log_path, row)

    return per_run_path

def _write_csv_with_header(path: str, rows: list[Dict[str, Any]]) -> None:
    fieldnames = _ordered_fieldnames(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        row.update(extra)

    per_run_path = os.path.join(results_dir, f"{ts}_{basename}.csv")
    log_path     = os.path.join(results_dir, f"{basename}_log.csv")

    # write per-run file (with header)
    _write_csv_with_header(per_run_path, [row])

    # append into cumulative log (create with header if missing)
    _append_csv_with_header(log_path, row)

    return per_run_path

def _write_csv_with_header(path: str, rows: list[Dict[str, Any]]) -> None:
    fieldnames = _ordered_fieldnames(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _append_csv_with_header(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    fieldnames = _ordered_fieldnames(row)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()

        for r in rows:
            w.writerow(r)

def _append_csv_with_header(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.exists(path)
    fieldnames = _ordered_fieldnames(row)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

def _ordered_fieldnames(row: Dict[str, Any]) -> list[str]:
    # put identifiers first, then hp fields, then metrics
    ids = ["run_id", "timestamp", "target"]
    hp_keys = [
        "random_state","test_size","inner_val_frac","max_rows",
        "em_iters","warmup_epochs","mstep_epochs","patience",
        "batch_size","lr","weight_decay",
        "K","q","K_lap","lap_lambda","reg_r",
        "hidden_layers","dropout_prob","device",
        "cg_rel_tol","cg_max_iter","cg_patience","verbose"
    ]
    metrics = [
        "train_within_5","train_within_10","train_within_15","train_median_abs_rel",
        "test_within_5","test_within_10","test_within_15","test_median_abs_rel"
    ]
    # keep whatever exists and preserve order above
    ordered = [k for k in ids if k in row]
    ordered += [k for k in hp_keys if k in row]
    ordered += [k for k in metrics if k in row]
    # any extras at the end
    ordered += [k for k in row.keys() if k not in set(ordered)]
    return ordered
