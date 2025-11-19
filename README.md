🏠 House-Price Forecasting via Latent Manifold Estimation (LME)


Neural-network intrinsic value + non-parametric spatial desirability (KD-Tree neighbors) with an EM-style trainer. Includes spatial train/val/test splits to avoid leakage and paper-style accuracy metrics.
   ```
📂 Repository Overview
fall-2025-group3/
├─ reports/
│  ├─ Latex_report/
│  ├─ Markdown_Report/
│  ├─ Word_Report/
│  ├─ Latent Manifold Model Report.pdf
│  └─ ZIP_Month Model.pdf
├─ results/
│  ├─ Comparison_metrics.txt
│  └─ Reports_and_Results_from_Capstone.pdf
├─ src/
│  ├─ component/
│  ├─ figs/                      # figures exported by the scripts
│  ├─ runlog/                    # optional logs
│  ├─ data_preprocessor_LME.py   # feature cleaning/engineering
│  ├─ latend Manifold model.py   # main LME script (note the spaces)
│  ├─ Time series analysis.py
│  └─ sub_sample.csv             # ← place the data file here (not tracked)
├─ README.md
└─ LICENSE

🚀 Quick Start
1) Clone and create a virtual environment
git clone <YOUR-REPO-URL>.git
cd fall-2025-group3

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install numpy pandas scikit-learn torch matplotlib tqdm

2) Put the data in place

Download sub_sample.csv (GWU Box) and put it here:
# verify file is present
ls -lh src/sub_sample.csv

3) Run the LME model

The filename has spaces—quote it:

python "src/latend Manifold model.py"

