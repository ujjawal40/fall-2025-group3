import os
import random
from pathlib import Path

import numpy as np
import torch

# ---------------- PATHS ----------------
SRC_DIR = Path(__file__).resolve().parent.parent  # .../src
PROJECT_ROOT = SRC_DIR.parent                     # repo root

RESULTS_DIR = PROJECT_ROOT / "results"
RUN_DIR = RESULTS_DIR / "runlog"
FIG_DIR = RESULTS_DIR / "figs"

for p in [RESULTS_DIR, RUN_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Raw CSV path (local sub_sample export) – relative to src/
RAW_CSV_PATH = SRC_DIR / "sub_sample.csv"

# ---------------- DATA / INDEX CONFIG ----------------
MIN_START_DATE     = "2022-01-01"
HOLDOUT_DAYS       = 60
MIN_SOLD_PER_ZIP_M = 20
MIN_LIST_PER_ZIP_M = 40
RANDOM_SEED        = 42

# ---------------- DEVICE / SEEDS ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() // 2))

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------- MODEL / TRAINING ----------------
TARGETS = ["Y_H1", "Y_H2"]
QUANTILES = [0.1, 0.5, 0.9]

PINBALL_WEIGHT      = 1.0
L1_MEDIAN_WEIGHT    = 0.5
RELIABILITY_C       = 2.5
VAL_FRACTION_OF_TRAIN_TIME = 1 / 6

EPOCHS   = 60
PATIENCE = 8
BATCH_SIZE = 2048

HIDDEN  = 384
LAYERS  = 3
DROPOUT = 0.15
EMB_DIM_CAP = 64

SUPPRESS_WIDTH_PCT = 0.15
MIN_LEVEL          = 1.0

# ---------------- FEATURE SETS ----------------
CATEGORICAL_COLS = [
    "H3_R6", "H3_R7", "H3_R8", "H3_R9",
    "STATE_MODE", "COUNTY_MODE", "ZIPCODE"
]

NON_FEATURE_KEYS = {"ZIPCODE", "YM", "STATE_MODE", "COUNTY_MODE", "DAY_FOR_SPLIT"}
LABEL_COLS = {"Y_H1", "Y_H2", "IDX_FUTURE_H1", "IDX_FUTURE_H2"}
FUTUREISH_COLS = {"BASE_DLOG_H1", "BASE_DLOG_H2", "_BASE_FWD1", "_BASE_FWD2", "_BASE_NOW"}
