# 🏠 Neural House Price Forecasting  
### ZIP×Month MultiTaskQuantileNet Pipeline + Latent Manifold Estimation (LME)

This repository contains **two complementary algorithms** for residential property valuation:

1. **ZIP×Month MultiTaskQuantileNet Pipeline**  
   A production-style, end-to-end system that:
   - Flattens raw `PRICEHISTORY` into event rows  
   - Builds rich temporal / spatial / market features at the **ZIP×Month** level  
   - Trains a **multi-task, multi-quantile MLP** (`MultiTaskQuantileNet`)  
   - Evaluates performance with rolling **backtests** across multiple temporal folds  

2. **Latent Manifold Estimation (LME)**  
   An experimental model that decomposes prices into:
   - **Intrinsic value** (parametric neural net on house features)  
   - **Spatial desirability** (non-parametric latent surface over geography)  
   learned with an EM-style loop.

Both algorithms live under `src/` and write results into `results/`.

---

## 📂 Repository Structure

```bash
fall-2025-group3/
│
├── src/
│   ├── component/
│   │   # ZIP×Month MultiTaskQuantileNet pipeline
│   │   ├── backtest.py            # Rolling backtest (multiple temporal folds)
│   │   ├── config.py              # Paths, hyperparameters, constants (RUN_DIR, RESULTS_DIR, etc.)
│   │   ├── data_io.py             # Load raw CSV, build combined_events + feature matrix
│   │   ├── datasets.py            # TabularDataset + dataframe prep (numeric/cat splits, scaler)
│   │   ├── evaluation.py          # eval_split, suppression_report, dlog→level conversion
│   │   ├── events.py              # CombinedEventsBuilder: flatten PRICEHISTORY → event rows
│   │   ├── features.py            # ZipIndexFeatureizer: temporal, lags, macro, momentum, etc.
│   │   ├── geo_tiling.py          # Approximate H3-style tiling per ZIP×Month
│   │   ├── metrics.py             # WAPE, MdAPE, coverage, helper metrics
│   │   ├── model.py               # MultiTaskQuantileNet (multi-task, multi-quantile MLP)
│   │   ├── training.py            # train_one loop (pinball + L1, early stopping, weighting)
│   │   ├── utils.py               # Shared helpers (downcast_df, safe parsing, etc.)
│   │   ├── visualization.py       # Plots: MAE/WAPE bars, R² lines, radar, calibration, robustness
│   │   ├── zip_index.py           # Helpers for ZIP×Month index construction
│   │   #
│   │   # Latent Manifold Estimation (LME) – all files ending in `_LME.py`
│   │   ├── config_LME.py          # Paths + hyperparameters specific to LME
│   │   ├── data_io_LME.py         # LME loader / preprocessing
│   │   ├── data_preprocessor_LME.py
│   │   ├── metrics_LME.py
│   │   ├── model_LME.py           # IntrinsicPriceNet + LME model pieces
│   │   ├── splits_LME.py          # Spatial splits / helpers for LME
│   │   ├── trainer_LME.py         # EM-style training loop (E: D, M: W)
│   │   ├── utils_LME.py           # LME-specific utilities
│   │   ├── visualization_LME.py   # LME plots + markdown-like summaries
│   │   ├── surface.py             # KernelSurface, LaplacianOp (used by LME)
│   │   #
│   │   # Internal scratch / prototype files (not part of the main pipelines)
│   │   ├── class_one.py
│   │   ├── class_two.py
│   │   ├── utils_one.py
│   │   └── utils_two.py
│   │
│   ├── tests/                     # (Optional) unit / smoke tests
│   ├── README.md                  # Code-level README for src/ (see below)
│   ├── latend Manifold model.py   # LME experiment entry point
│   └── time_series_analysis.py    # MAIN script: ZIP×Month pipeline end-to-end
│
├── results/
│   ├── runlog/                    # Saved checkpoints + JSON metadata
│   ├── figs/                      # Generated figures from visualization.py / visualization_LME.py
│   └── ...                        # CSV metrics, comparison text, PDFs, etc.
│
└── README.md                      # This file
```

## Download the data 
https://gwu.box.com/s/c38fp0sbxkcy2dwl31jnqbgaklso6136

##⚙️ Quick Start (common setup for BOTH algorithms)

**1) Clone the repo**

```bash
git clone https://github.com/ujjawal40/fall-2025-group3.git
cd fall-2025-group3
```
**2) Put the data in src/**
```bash
# make sure the file exists at this exact path:
ls -l src/sub_sample.csv
```

**3) Create & activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**4) Install dependencies (requirements are inside src/)**
```bash
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

##After this, you can run either (or both) algorithms.

**🚀 Component 1 — ZIP×Month MultiTaskQuantileNet Pipeline**

-Goal: End-to-end forecasting pipeline that:

-Flattens PRICEHISTORY into event rows

-Builds rich ZIP×Month features (lags, indices, macro, momentum, etc.)

-Trains a multi-task, multi-quantile MLP (MultiTaskQuantileNet)

-Evaluates performance via rolling backtests

🔧 Entry point

From the repo root:

```bash
cd src
python time_series_analysis.py
```
**What this script does (high level)**

-Loads config from component/config.py (paths, hyperparameters, run IDs).

-Reads src/sub_sample.csv via component/data_io.py.

-Uses component/events.py to flatten nested PRICEHISTORY into one row per event.

-Uses component/geo_tiling.py, features.py, and zip_index.py to build ZIP×Month indices and temporal / market features.

-Prepares tabular datasets + scalers in datasets.py.

-Defines and trains MultiTaskQuantileNet from model.py using training.py:

-Multi-task (e.g., horizons H1/H2)

-Multi-quantile (p10 / p50 / p90, etc.)

-Pinball loss + optional L1 + early stopping.

-Runs rolling backtests with backtest.py.

-Evaluates metrics (metrics.py, evaluation.py) and saves plots (visualization.py) under ../results/.

**🌈 Component 2 — Latent Manifold Estimation (LME)**

Goal: Separate:

Intrinsic value – neural net on house features, and

Spatial desirability – a smooth latent surface over geography,

learned via an EM-style loop.

🔧 Entry point

From the repo root:

```bash
cd src
python "latend Manifold model.py"
```

**What this script does (high level)**

-Loads LME config from component/config_LME.py.

-Reads the same src/sub_sample.csv, using data_io_LME.py / data_preprocessor_LME.py.

-Builds an intrinsic feature matrix for properties.

-Defines the intrinsic value network and related components in model_LME.py.

-Uses surface.py to construct a kernel-based spatial surface with Laplacian regularization.

-Runs the EM-style loop in trainer_LME.py:

-E-step: estimate spatial desirability field D(x, y) given current network weights.

-M-step: update intrinsic network weights W conditioned on D.

-Computes LME metrics (metrics_LME.py) and produces desirability / diagnostic plots (visualization_LME.py).

-Writes outputs to an LME-specific subdirectory inside ../results/.

