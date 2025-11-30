Enhancing Residential Property Valuation via Latent Manifold Estimation (LME)

A clean, reproducible pipeline to estimate residential prices using a Latent Manifold Estimation approach. The model separates an intrinsic, parametric component (home attributes) from a spatial desirability surface learned over neighborhoods, then alternates E/M steps to refine both.

Scope: this README covers only the LME pipeline you refactored (the src/ + component/ code).

What you get

One-command run to train and evaluate the LME model

Automatic artifacts in results/:

results_summary.md (markdown report)

results/figs/*.png (EM loss, M-step loss, scatter, error histogram)

results/lme_eval_metrics.csv (append-only per-run metrics)

Repo layout (relevant parts)
.
├─ src/
│  ├─ lme_main.py                 # main entry point
│  ├─ requirements.txt            # Python deps (kept in src per your setup)
│  └─ component/
│     ├─ config_LME.py            # HParams + paths (RESULTS_DIR, FIG_DIR)
│     ├─ utils_LME.py             # seeds, dir utils, helpers
│     ├─ data_io_LME.py           # loading + preprocessing
│     ├─ splits_LME.py            # spatial splits & standardization
│     ├─ surface.py               # KernelSurface + LaplacianOp
│     ├─ model_LME.py             # IntrinsicPriceNet (parametric model)
│     ├─ trainer_LME.py           # EM loop + desirability solver
│     ├─ metrics_LME.py           # metrics + CSV writer
│     └─ visulization_LME.py      # plots + results markdown
└─ results/                        # created on first run

Data

Download the dataset from GW Box:
https://gwu.box.com/s/c38fp0sbxkcy2dwl31jnqbgaklso6136

Place the file as:

Recommended (simple): src/sub_sample.csv

(The loader also auto-searches: src/data/sub_sample.csv, data/src/sub_sample.csv, data/sub_sample.csv.)

Quickstart (works on macOS/Linux/EC2)
# 1) Clone
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

# 2) Put data in place
#    └─ ensure the dataset is available at:  src/sub_sample.csv

# 3) Create a clean Python env (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# 4) Install deps (requirements.txt is kept in src/)
python3 -m pip install --upgrade pip
python3 -m pip install -r src/requirements.txt

# 5) Run the LME pipeline
cd src
python3 lme_main.py


That’s it. The run prints “paper-style metrics” to the console and drops all artifacts in ../results/.

What the script does (chronology)

Load & preprocess (component.data_io_LME.load_with_preprocessor)

Reads src/sub_sample.csv

Cleans/engineers features, builds X, y, and extras (includes spatial columns, feature names, etc.)

Spatial split & surface (component.splits_LME.build_surface_and_splits)

Spatial train/test split using grid cells (prevents leakage across nearby homes)

Projects lat/lon to meters, standardizes, builds a KD-Tree surface for neighborhood interpolation

Optional graph Laplacian for smoothness

Target choice

If SQFT exists: target = log(PPSQFT); else fallback to log(Price)

Applies simple sample-weighting based on PPSQFT quantiles

Inner spatial VAL (make_inner_spatial_val)

Carves a spatial validation set inside train for early-stopping the M-step

Model & EM (trainer_LME.LMETrainer)

Pretrain parametric net on inner-train

E-step: solve D (desirabilities) with conjugate-gradient using KD-tree weights (+ optional Laplacian)

M-step: fit parametric net to residuals y − h(D) with early-stop on inner spatial VAL

Iterate E/M for em_iters

Finalize & evaluate

Recompute D on full train once with the learned W

Predict on train/test; compute paper-style metrics

Save plots & markdown & append metrics row to results/lme_eval_metrics.csv

Configure (optional)

Edit defaults in src/component/config_LME.py:

HParams: K, q, em_iters, warmup_epochs, mstep_epochs, lap_lambda, reg_r, batch_size, lr, etc.

Output paths: RESULTS_DIR = "results", FIG_DIR = f"{RESULTS_DIR}/figs"

No CLI flags yet—just modify and re-run.

Output artifacts (after a successful run)

results/results_summary.md – readable summary with the two paper-style tables (TRAIN/TEST)

results/lme_eval_metrics.csv – appends one row per run:

timestamp, run_id, target, within_5, within_10, within_15, median_abs_rel

results/figs/:

em_loss.png – EM energy vs iteration

mstep_losses.png – M-step loss curves per EM iteration

pred_vs_true_test.png – scatter (TEST)

abs_rel_hist_test.png – absolute relative error distribution (TEST)

Environment notes

Confirmed with Python 3.9 (Amazon Linux 2023 / EC2).

CPU is fine; GPU not required.

If you use system Python, prefer a venv to avoid conflicts.

Troubleshooting

Can’t find data: ensure src/sub_sample.csv exists (exact name).

Import errors: run from repo root, then cd src → python3 lme_main.py.
Keep the component imports as in the template.

Permissions on EC2: chmod 600 <your.pem> before SSH/SCP.

Plots/metrics missing: check write permissions; results/ is created automatically.
