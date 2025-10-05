## Instruction for code folder

- Shared classes and functions should live in the top-level `src/` directory.
- Main loop code needs to be in the main loop code folder.
- All codes need to have docstrings and hints.
- The regression runners (`glm_model.py`, `xgboost_model.py`) depend on `scikit-learn`,
  `xgboost`, `matplotlib`, `numpy`, and `pandas`. Ensure these packages are installed in
  the interpreter you use to execute the scripts.
