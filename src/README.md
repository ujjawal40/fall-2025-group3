# 🏠 Enhancing Residential Property Valuation via Latent Manifold Estimation (LME)

Neural intrinsic value **+** non-parametric spatial desirability learned with an EM-style loop.  
This repo exposes a clean, reproducible **LME** pipeline located in `src/` with reusable modules under `src/component/`.

---

## Table of Contents
- [Directory](#directory)
- [Data](#data)
- [Quick Start](#quick-start)
- [What the Pipeline Does](#what-the-pipeline-does)
- [Configuration (Optional)](#configuration-optional)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Environment](#environment)

---

## Directory
fall-2025-group3/
├─ src/
│ ├─ lme_main.py # main entry
│ ├─ requirements.txt # deps (kept in src/)
│ └─ component/
│ ├─ config_LME.py # HParams + paths (RESULTS_DIR, FIG_DIR)
│ ├─ utils_LME.py # seeds, standardize-by-train, helpers
│ ├─ data_io_LME.py # load + preprocess
│ ├─ splits_LME.py # spatial splits + lat/lon→meters
│ ├─ surface.py # KernelSurface, LaplacianOp
│ ├─ model_LME.py # IntrinsicPriceNet (parametric part)
│ ├─ trainer_LME.py # EM loop (E: D, M: W)
│ ├─ metrics_LME.py # paper-style metrics + CSV writer
│ └─ visulization_LME.py # plots + results markdown
└─ results/ # created on first run


---

## Data

Download the dataset (approved for README) and place it **exactly** at:

src/sub_sample.csv

Source (GW Box):  
https://gwu.box.com/s/c38fp0sbxkcy2dwl31jnqbgaklso6136

> The loader also tries `src/data/sub_sample.csv`, `data/src/sub_sample.csv`, `data/sub_sample.csv`, but **`src/sub_sample.csv`** is the simplest.

---

## Quick Start

**1) Clone**
```bash
git clone https://github.com/ujjawal40/fall-2025-group3.git
cd fall-2025-group3
```

