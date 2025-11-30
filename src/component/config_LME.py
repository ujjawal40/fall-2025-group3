# src/component/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class HParams:
    # Data / splits
    test_size: float = 0.20
    inner_val_frac: float = 0.12
    random_state: int = 42
    max_rows: Optional[int] = None

    # EM loop
    em_iters: int = 6
    warmup_epochs: int = 8
    mstep_epochs: int = 12
    patience: int = 10

    # Optimizer
    batch_size: int = 512
    lr: float = 5e-4
    weight_decay: float = 5e-4

    # Surface / neighbors
    K: int = 40
    q: float = 1.0
    K_lap: Optional[int] = None
    lap_lambda: float = 0.02

    # Regularization on D (ridge)
    reg_r: float = 5e-2

    # Intrinsic model
    hidden_layers: Tuple[int, ...] = (256, 128, 64, 32)
    dropout_prob: float = 0.25

    # Compute
    device: str = "cpu"

    # CG stopping
    cg_rel_tol: float = 1e-5
    cg_max_iter: int = 300
    cg_patience: int = 10

    # Logging
    verbose: bool = True

RESULTS_DIR = "results"
FIG_DIR = f"{RESULTS_DIR}/figs"
