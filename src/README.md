Enhancing Residential Property Valuation via Latent Manifold Estimation (LME)

Neural intrinsic value (home attributes) + non-parametric spatial desirability (KD-Tree neighbors).
EM-style training, spatial splits to avoid leakage, and paper-style accuracy metrics.

.
├─ src/
│  ├─ lme_main.py
│  ├─ requirements.txt
│  └─ component/
│     ├─ config_LME.py
│     ├─ utils_LME.py
│     ├─ data_io_LME.py
│     ├─ splits_LME.py
│     ├─ surface.py
│     ├─ model_LME.py
│     ├─ trainer_LME.py
│     ├─ metrics_LME.py
│     └─ visulization_LME.py
└─ results/            # created on first run

1) Setup Environment
# Clone the repo
git clone https://github.com/ujjawal40/fall-2025-group3.git
cd fall-2025-group3

# (Recommended) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate           # Windows PowerShell:  .venv\Scripts\Activate.ps1

# Install Python dependencies (requirements are kept under src/)
python3 -m pip install --upgrade pip
python3 -m pip install -r src/requirements.txt

2) Get the Data
# Download from GW Box (public to your professor)
https://gwu.box.com/s/c38fp0sbxkcy2dwl31jnqbgaklso6136

# Place the CSV here (exact name is flexible, but simplest is):
#   src/sub_sample.csv
#
# Auto-fallback search paths if you prefer:
#   src/data/sub_sample.csv
#   data/src/sub_sample.csv
#   data/sub_sample.csv

3) Train & Evaluate (LME)
# From repo root:
cd src
python3 lme_main.py

4) View Results
ls ../results



