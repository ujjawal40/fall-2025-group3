## Instruction for code folder

- Shared classes and functions should live in the top-level `src/` directory.
- Main loop code needs to be in the main loop code folder.
- All codes need to have docstrings and hints.
- The regression runners (`glm_model.py`, `xgboost_model.py`) depend on `scikit-learn`,
  `xgboost`, `matplotlib`, `numpy`, and `pandas`. Ensure these packages are installed in
  the interpreter you use to execute the scripts.
- Both regression runners can operate without arguments by falling back to the diabetes
  regression dataset bundled with `scikit-learn`. Provide a CSV path and target column if
  you want to run against your own data.
- When working with the 100k-row Zillow export, start with a subsample to avoid running
  out of memory on local machines: `--max-rows 10000` or lower keeps preprocessing and
  histogram boosting within a few gigabytes of RAM. The scripts sample **before** one-hot
  encoding so the feature matrix is reduced accordingly.
- All preprocessing flows now clip price-per-square-foot outliers at roughly the 99.5th
  percentile before taking logarithms. This keeps downstream models numerically stable.
  Pass `clip_ppsqft_quantile=None` to `DataPreprocessor.prepare_features` if you need the
  raw distribution without trimming.
- The GLM runner leaves one-hot encoding disabled by default to mirror the neural
  network workflow. Pass `--one-hot` if you need categorical dummies and have enough
  memory; otherwise the compact feature set is safer for dense linear models.
- Plots are displayed interactively by default. Use `--output-dir results/plots` to save
  the prediction and residual figures to disk (and `--no-show` to skip the GUI pop-up).
