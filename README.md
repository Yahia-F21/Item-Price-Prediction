Project 2 — Item Price Prediction (V6)

Overview
- File: `project2_v6.py`
- Description: End-to-end regression pipeline for predicting item prices (target `Y`). V6 adds Bayesian-smoothed target encoding, new features, Optuna LGBM tuning, two-round pseudo-labeling, multi-start blend optimization, and automated plotting/submission outputs.

Key Features
- Robust preprocessing and imputation for missing/zero values
- Extensive feature engineering (interaction terms, item/outlet statistics, frequency, price diversity)
- Bayesian-smoothed target encoding (leak-free, KFold-based)
- Model zoo: Ridge, ExtraTrees, HistGBM, XGBoost (3 seeds), LightGBM (multiple configs + Optuna-tuned), CatBoost (grid of depths & seeds)
- 10-fold OOF evaluation for all models
- Stacking (Ridge meta-learner), optimized blend (multi-start SLSQP), and simple averaging
- Two-round pseudo-labeling to refine test predictions
- Visualizations saved to `plots_v6/` and submissions to `submissions_v6/`

Repository structure
- `project2_v6.py`  — main script (single-file pipeline)
- `plots_v6/`       — generated visualizations (histogram, heatmap, residuals, etc.)
- `submissions_v6/` — CSVs ready for leaderboard submission

Quickstart
1. Install Python (3.9+ recommended) and create a virtual environment.

2. Install dependencies (example):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# or macOS/Linux
# source .venv/bin/activate
pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost scipy
# Optional (fast tuning):
pip install optuna
```

3. Place dataset files in a data folder and update `DATA_DIR` at the top of `project2_v6.py`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

4. Run the script:

```bash
python project2_v6.py
```

What the script produces
- `plots_v6/fig1_target_distribution.png` and other figures (saved in `PLOT_DIR`)
- `submissions_v6/submission_v6_blend.csv` (optimized blend)
- `submissions_v6/submission_v6_pseudo_r1.csv` / `_pseudo_r2.csv` (pseudo-labeled rounds)
- `submissions_v6/submission_v6_stack.csv` (stacked ensemble)

Important configuration knobs
- `SEED` — reproducibility seed
- `N_FOLDS` — OOF folds (default 10)
- `DATA_DIR` — path to your dataset folder (default set at top of file)
- `SMOOTH_M` — Bayesian smoothing parameter for target encoding
- Optuna: optional hyperparameter search (set `optuna` installed to enable)

Notes on reproducibility
- Script uses seeded KFold and deterministic seeds for model variants where supported. Exact run times and small numeric differences can arise from parallelism or non-deterministic library behavior.

Limitations & TODOs
- In-memory processing only; large datasets may need chunking or higher memory
- No CLI / config file — adjust constants inside `project2_v6.py`
- Consider adding a `requirements.txt` and lightweight runner script
- Optionally add saving/loading of trained models for quick inference

Suggested next steps
- Create `requirements.txt` with pinned versions and add a small `run.sh`/`run.ps1` wrapper
- Add argument parsing (`--data-dir`, `--tune`, `--n-jobs`) for flexibility
- (Optional) Save model artifacts for offline inference

Author / Credits
- Educational / competition-style pipeline combining modern ensembling and pseudo-labeling techniques. Designed for reproducible model development and leaderboard submission.

If you want, I can:
- generate a `requirements.txt` with likely package versions,
- add a small runner script, or
- convert the main script to accept CLI arguments.

File saved: [project2_v6_complete/README.md](project2_v6_complete/README.md)
