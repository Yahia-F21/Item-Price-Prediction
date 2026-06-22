"""
CSE 281 - Spring 2026 - Project 2 (V6 - LEADERBOARD PUSH)
Item Price Prediction

WHAT'S NEW IN V6 over V5:
  1. Bayesian-smoothed target encoding  — handles rare categories better,
       avoids over-fitting on outlets/item-types that appear very few times.
  2. Three new smart features:
       - Item-Outlet pair frequency  (how many times this exact product
         appears at this exact store — captures product-store familiarity)
       - Outlet price diversity ratio  (std/mean of MRP per outlet — proxy
         for how "specialized" vs "general" a store is)
       - Cross target encoding: Item Category × Outlet Type combined
         (captures e.g. "Dairy at Grocery Store" as one signal)
  3. Optuna hyperparameter search for LightGBM (40 fast trials) —
       automated search for the best learning_rate, num_leaves, etc.
       Very explainable: "we used Bayesian optimization to tune LGBM."
  4. Two-round pseudo-labeling — first round creates soft labels,
       second round retrains on them for a tighter test-set estimate.
  5. Multi-start blend optimizer — runs 20 random initializations of the
       weight search to avoid getting stuck in a local minimum.
  6. All key visualizations saved to plots_v6/ folder automatically.
"""

import os, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.base import clone
from scipy.optimize import minimize

# Optuna for hyperparameter search (pip install optuna if missing)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  [INFO] optuna not installed — skipping LGBM tuning. Run: pip install optuna")

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
SEED      = 42
N_FOLDS   = 10
DATA_DIR  = r'd:\AI project'           # <-- change if your data is elsewhere
PLOT_DIR  = os.path.join(DATA_DIR, 'plots_v6')
SUB_DIR   = os.path.join(DATA_DIR, 'submissions_v6')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SUB_DIR,  exist_ok=True)
np.random.seed(SEED)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
COLORS = sns.color_palette('muted')

print("=" * 70)
print("CSE 281 - Project 2: Item Price Prediction (V6)")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/8] Loading data...")
train_df   = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df    = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
print(f"  Train: {train_df.shape}  |  Test: {test_df.shape}")

# ============================================================
# 2. PREPROCESSING
# ============================================================
print("\n[2/8] Preprocessing...")

train_df['source'] = 'train'
test_df['source']  = 'test'
df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

# Fix X3 label noise (5 raw labels → 2 clean labels)
df['X3'] = df['X3'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

# Impute X2 (Item Weight): use same-product median, then same-type median
df['X2'] = df.groupby('X1')['X2'].transform(lambda x: x.fillna(x.median()))
df['X2'] = df.groupby('X5')['X2'].transform(lambda x: x.fillna(x.median()))
df['X2'] = df['X2'].fillna(df['X2'].median())

# Fix zero visibility — replace with item-type mean
mask_zero = df['X4'] == 0
df.loc[mask_zero, 'X4'] = df.groupby('X5')['X4'].transform('mean')[mask_zero]

# Impute X9 (Outlet Size) using known outlet mapping (deterministic)
outlet_size = (df.dropna(subset=['X9'])
               .drop_duplicates('X7')
               .set_index('X7')['X9'])
df['X9'] = df.apply(
    lambda r: outlet_size.get(r['X7'], 'Small') if pd.isnull(r['X9']) else r['X9'],
    axis=1
)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n[3/8] Feature engineering (V5 base + V6 additions)...")

train_mask = df['source'] == 'train'

# ── Ordinal encodings ─────────────────────────────────────────────────────
df['Outlet_Age'] = 2026 - df['X8']
df['Item_Cat']   = df['X1'].str[:2]
df['X9_ord']     = df['X9'].map({'Small': 0, 'Medium': 1, 'High': 2})
df['X10_ord']    = df['X10'].map({'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2})
df['X11_ord']    = df['X11'].map({
    'Grocery Store': 0, 'Supermarket Type1': 1,
    'Supermarket Type2': 2, 'Supermarket Type3': 3
})
df['Is_Grocery'] = (df['X11'] == 'Grocery Store').astype(int)

# ── V5 interaction features (proven valuable — keep all) ──────────────────
df['MRP_x_OutletType'] = df['X6'] * df['X11_ord']
df['MRP_x_Grocery']    = df['X6'] * df['Is_Grocery']
df['MRP_x_OutletAge']  = df['X6'] * df['Outlet_Age']
df['MRP_x_Tier']       = df['X6'] * df['X10_ord']
df['MRP_x_Size']       = df['X6'] * df['X9_ord']
df['Vis_Ratio']        = df['X4'] / (df.groupby('X5')['X4'].transform('mean') + 1e-3)
df['MRP_squared']      = df['X6'] ** 2
df['Log_MRP']          = np.log1p(df['X6'])
df['Sqrt_MRP']         = np.sqrt(df['X6'])
df['MRP_dev_outlet']   = df['X6'] - df.groupby('X7')['X6'].transform('mean')
df['Price_Per_Weight'] = df['X6'] / (df['X2'] + 1e-3)
df['MRP_bin']          = pd.qcut(df['X6'], q=10, labels=False, duplicates='drop')
df['MRP_pct_in_outlet']= df.groupby('X7')['X6'].rank(pct=True)
df['Outlet_item_count']= df.groupby('X7')['X1'].transform('count')
df['Vis_dev_outlet']   = df['X4'] - df.groupby('X7')['X4'].transform('mean')
outlet_mrp_mean        = df.groupby('X7')['X6'].transform('mean')
df['MRP_ratio_outlet'] = df['X6'] / (outlet_mrp_mean + 1e-3)
df['Weight_x_MRP']     = df['X2'] * df['X6']

item_mrp_stats = df.groupby('X1')['X6'].agg(['mean', 'std', 'min', 'max'])
item_mrp_stats.columns = ['Item_MRP_mean', 'Item_MRP_std', 'Item_MRP_min', 'Item_MRP_max']
df = df.merge(item_mrp_stats, on='X1', how='left')
df['MRP_dev_from_item_mean'] = df['X6'] - df['Item_MRP_mean']

# ── NEW V6 Feature 1: Item-Outlet pair frequency ──────────────────────────
# How many rows share the exact same (product_id, outlet_id) pair?
# If a product appears at a store many times it's a "regular" item there.
df['Item_Outlet_freq'] = df.groupby(['X1', 'X7'])['X6'].transform('count')

# ── NEW V6 Feature 2: Outlet price diversity ratio (std / mean of MRP) ───
# Low ratio → specialty store with similar-priced items
# High ratio → general store selling everything from cheap to expensive
outlet_mrp_std = df.groupby('X7')['X6'].transform('std').fillna(0)
df['Outlet_price_diversity'] = outlet_mrp_std / (outlet_mrp_mean + 1e-3)

# ── NEW V6 Feature 3: Item MRP deviation from item-type outlet mean ───────
# Within the same item type at the same outlet, how far is this item's MRP?
df['MRP_dev_type_outlet'] = (df['X6'] -
    df.groupby(['X5', 'X7'])['X6'].transform('mean'))

# ── Bayesian-smoothed target encoding (V6 upgrade over V5) ────────────────
# Formula: (count * category_mean + m * global_mean) / (count + m)
# m is the smoothing factor — higher m = more shrinkage toward global mean.
# This is better than plain mean for rare categories (e.g. outlets with few rows).
kf_te     = KFold(n_splits=5, shuffle=True, random_state=SEED)
SMOOTH_M  = 20   # smoothing strength (tune: higher = more conservative)

def bayesian_te(df, col, target, train_mask, kf, m=SMOOTH_M):
    """Leak-free Bayesian-smoothed target encoding."""
    col_name  = f'{col}_bte'
    df[col_name] = np.nan
    train_idx = df[train_mask].index
    test_idx  = df[~train_mask].index
    global_mean = df.loc[train_idx, target].mean()

    for tr_idx, val_idx in kf.split(train_idx):
        tr_rows  = train_idx[tr_idx]
        val_rows = train_idx[val_idx]
        stats = df.loc[tr_rows].groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        df.loc[val_rows, col_name] = df.loc[val_rows, col].map(smooth)

    # Test set: use full training data for smoothed global means
    stats_all = df.loc[train_idx].groupby(col)[target].agg(['mean', 'count'])
    smooth_all = (stats_all['count'] * stats_all['mean'] + m * global_mean) / (stats_all['count'] + m)
    df.loc[test_idx, col_name] = df.loc[test_idx, col].map(smooth_all)
    df[col_name] = df[col_name].fillna(global_mean)
    return df

for col in ['X7', 'X5', 'X11', 'Item_Cat', 'X10', 'MRP_bin']:
    df = bayesian_te(df, col, 'Y', train_mask, kf_te)

# NEW: cross target encoding — Item Category × Outlet Type combined
df['Cat_x_Outlet'] = df['Item_Cat'].astype(str) + '_' + df['X11'].astype(str)
df = bayesian_te(df, 'Cat_x_Outlet', 'Y', train_mask, kf_te)

# ── Aggregate Y stats from training labels ────────────────────────────────
outlet_stats = df[train_mask].groupby('X7')['Y'].agg(['mean', 'std', 'median'])
outlet_stats.columns = ['Out_Y_mean', 'Out_Y_std', 'Out_Y_med']
df = df.merge(outlet_stats, on='X7', how='left')

item_type_stats = df[train_mask].groupby('X5')['Y'].agg(['mean', 'std'])
item_type_stats.columns = ['Item_Y_mean', 'Item_Y_std']
df = df.merge(item_type_stats, on='X5', how='left')

item_id_stats = df[train_mask].groupby('X1')['Y'].agg(['mean', 'std', 'count'])
item_id_stats.columns = ['ItemID_Y_mean', 'ItemID_Y_std', 'ItemID_count']
df = df.merge(item_id_stats, on='X1', how='left')
df['ItemID_Y_std'] = df['ItemID_Y_std'].fillna(0)
df['ItemID_count'] = df['ItemID_count'].fillna(0)

# ── One-hot encode remaining categoricals ─────────────────────────────────
df = pd.get_dummies(df, columns=['X3', 'X5', 'X7', 'X11', 'Item_Cat', 'Cat_x_Outlet'],
                    drop_first=True)

drop_cols = ['X1', 'X8', 'X9', 'X10', 'source']
train_f = df[train_mask].drop(columns=drop_cols)
test_f  = df[~train_mask].drop(columns=drop_cols + ['Y'], errors='ignore')

X_train = train_f.drop(columns=['Y'])
y_train = train_f['Y']
X_test  = test_f.copy()

for c in set(X_train.columns) - set(X_test.columns):
    X_test[c] = 0
X_test = X_test[X_train.columns]
print(f"  Final feature count: {X_train.shape[1]}")

# ============================================================
# 4. OPTUNA HYPERPARAMETER SEARCH FOR LIGHTGBM  (V6 addition)
# ============================================================
print("\n[4/8] Optuna hyperparameter search for LightGBM (40 trials)...")
best_lgbm_params = None

if HAS_OPTUNA:
    kf_opt = KFold(n_splits=5, shuffle=True, random_state=SEED)

    def lgbm_objective(trial):
        params = dict(
            n_estimators     = 3000,
            learning_rate    = trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            num_leaves       = trial.suggest_int('num_leaves', 31, 255),
            max_depth        = trial.suggest_int('max_depth', 4, 9),
            min_child_samples= trial.suggest_int('min_child_samples', 10, 50),
            subsample        = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.4, 0.8),
            reg_alpha        = trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            reg_lambda       = trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
            objective        = 'mae',
            verbosity        = -1,
            n_jobs           = -1,
            random_state     = SEED,
        )
        maes = []
        for tr_idx, val_idx in kf_opt.split(X_train):
            m = LGBMRegressor(**params)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = m.predict(X_train.iloc[val_idx])
            maes.append(mean_absolute_error(y_train.iloc[val_idx], preds))
        return np.mean(maes)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lgbm_objective, n_trials=40, show_progress_bar=False)
    best_lgbm_params = study.best_params
    print(f"  Best LGBM params found: {best_lgbm_params}")
    print(f"  Best LGBM 5-fold MAE:   {study.best_value:.4f}")
else:
    print("  Skipping Optuna — using default LGBM params from V5.")

# ============================================================
# 5. MODEL TRAINING  (10-fold OOF)
# ============================================================
print("\n[5/8] Training all models (10-fold OOF)...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def train_oof(model, X, y, X_tst, kf, name='',
              use_early_stop=False, is_catboost=False):
    oof      = np.zeros(len(X))
    test_prd = np.zeros(len(X_tst))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        if is_catboost:
            m = CatBoostRegressor(**model.get_params())
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        else:
            m = clone(model)
            if use_early_stop:
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            else:
                m.fit(X_tr, y_tr)

        oof[val_idx] = m.predict(X_val)
        test_prd    += m.predict(X_tst) / kf.n_splits

    mae  = mean_absolute_error(y, oof)
    rmse = np.sqrt(mean_squared_error(y, oof))
    r2   = r2_score(y, oof)
    print(f"  {name:<34} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return oof, test_prd, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

all_oof, all_test, all_results = {}, {}, {}
feature_importance = {}   # for visualization

# 1. Ridge (linear baseline)
m = Ridge(alpha=5.0, random_state=SEED)
oof, tp, res = train_oof(m, X_train, y_train, X_test, kf, 'Ridge_a5')
all_oof['Ridge_a5'] = oof; all_test['Ridge_a5'] = tp; all_results['Ridge_a5'] = res

# 2. Extra Trees (diversity — different split criterion from RF)
m = ExtraTreesRegressor(
    n_estimators=600, max_depth=14, min_samples_leaf=2,
    max_features=0.5, criterion='absolute_error',
    random_state=SEED, n_jobs=-1
)
oof, tp, res = train_oof(m, X_train, y_train, X_test, kf, 'ExtraTrees')
all_oof['ExtraTrees'] = oof; all_test['ExtraTrees'] = tp; all_results['ExtraTrees'] = res

# 3. HistGradientBoosting (fast sklearn GBDT)
m = HistGradientBoostingRegressor(
    max_iter=1000, max_depth=6, learning_rate=0.03,
    min_samples_leaf=20, l2_regularization=0.5,
    loss='absolute_error', random_state=SEED
)
oof, tp, res = train_oof(m, X_train, y_train, X_test, kf, 'HistGBM')
all_oof['HistGBM'] = oof; all_test['HistGBM'] = tp; all_results['HistGBM'] = res

# 4. XGBoost — 3 seeds
for s in [42, 123, 777]:
    m = XGBRegressor(
        n_estimators=3000, max_depth=4, learning_rate=0.01,
        objective='reg:absoluteerror',
        subsample=0.8, colsample_bytree=0.6,
        reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, gamma=0.1,
        random_state=s, verbosity=0, early_stopping_rounds=150
    )
    nm = f'XGB_s{s}'
    oof, tp, res = train_oof(m, X_train, y_train, X_test, kf, nm, use_early_stop=True)
    all_oof[nm] = oof; all_test[nm] = tp; all_results[nm] = res

# 5. LightGBM — default 3-seed configs (always run)
for s in [42, 123, 777]:
    for leaves in [63, 127]:
        m = LGBMRegressor(
            n_estimators=3000, learning_rate=0.01,
            objective='mae', num_leaves=leaves,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0,
            min_child_samples=20, random_state=s,
            verbosity=-1, n_jobs=-1
        )
        nm = f'LGB_s{s}_l{leaves}'
        oof, tp, res = train_oof(m, X_train, y_train, X_test, kf, nm)
        all_oof[nm] = oof; all_test[nm] = tp; all_results[nm] = res

# 5b. Optuna-tuned LightGBM (V6 addition — if Optuna ran)
if HAS_OPTUNA and best_lgbm_params is not None:
    m = LGBMRegressor(
        n_estimators=3000,
        objective='mae',
        verbosity=-1, n_jobs=-1,
        random_state=SEED,
        **best_lgbm_params
    )
    oof, tp, res = train_oof(m, X_train, y_train, X_test, kf, 'LGB_Optuna')
    all_oof['LGB_Optuna'] = oof; all_test['LGB_Optuna'] = tp; all_results['LGB_Optuna'] = res

# 6. CatBoost — 4 depths × 3 seeds (strongest single-model family)
cb_best_oof, cb_best_test, cb_best_name = None, None, None
cb_best_mae = np.inf

for depth in [4, 5, 6, 7]:
    for s in [42, 123, 777]:
        m = CatBoostRegressor(
            iterations=3000, depth=depth, learning_rate=0.015,
            loss_function='MAE', l2_leaf_reg=3,
            random_seed=s, verbose=0,
            early_stopping_rounds=150, subsample=0.8
        )
        nm = f'CB_d{depth}_s{s}'
        oof, tp, res = train_oof(m, X_train, y_train, X_test, kf, nm, is_catboost=True)
        all_oof[nm] = oof; all_test[nm] = tp; all_results[nm] = res

        if res['MAE'] < cb_best_mae:
            cb_best_mae  = res['MAE']
            cb_best_oof  = oof
            cb_best_test = tp
            cb_best_name = nm

# Store best CatBoost separately for pseudo-labeling
print(f"\n  Best CatBoost: {cb_best_name}  MAE={cb_best_mae:.4f}")

# ============================================================
# 6. STACKING LAYER  (Ridge meta-learner on all OOF predictions)
# ============================================================
print("\n[6/8] Stacking layer...")

names    = list(all_oof.keys())
oof_mat  = np.column_stack([all_oof[n] for n in names])
test_mat = np.column_stack([all_test[n] for n in names])

stack_oof  = np.zeros(len(y_train))
stack_test = np.zeros(len(X_test))
meta = Ridge(alpha=1.0)

for tr_idx, val_idx in kf.split(oof_mat):
    meta.fit(oof_mat[tr_idx], y_train.iloc[tr_idx])
    stack_oof[val_idx] = meta.predict(oof_mat[val_idx])

meta.fit(oof_mat, y_train)
stack_test = meta.predict(test_mat)

stack_mae = mean_absolute_error(y_train, stack_oof)
print(f"  Stack (Ridge meta) OOF MAE: {stack_mae:.4f}")

all_oof['Stack_Ridge']  = stack_oof
all_test['Stack_Ridge'] = stack_test
all_results['Stack_Ridge'] = {
    'MAE':  stack_mae,
    'RMSE': np.sqrt(mean_squared_error(y_train, stack_oof)),
    'R2':   r2_score(y_train, stack_oof)
}

# ============================================================
# 7. OPTIMIZED BLEND  (multi-start to avoid local minima)
# ============================================================
print("\n[7/8] Building optimized ensemble (20 random starts)...")

names_all = list(all_oof.keys())
oof_all   = np.column_stack([all_oof[n] for n in names_all])
test_all  = np.column_stack([all_test[n] for n in names_all])
n_m       = len(names_all)

def blend_mae_fn(w):
    w = np.abs(w); w /= w.sum()
    return mean_absolute_error(y_train, oof_all @ w)

best_val = np.inf
best_w   = np.ones(n_m) / n_m

rng = np.random.RandomState(SEED)
for _ in range(20):                          # 20 random starting points
    w0 = rng.dirichlet(np.ones(n_m))
    res_opt = minimize(
        blend_mae_fn, w0, method='SLSQP',
        bounds=[(0, 1)] * n_m,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
    )
    if res_opt.fun < best_val:
        best_val = res_opt.fun
        best_w   = res_opt.x / res_opt.x.sum()

blend_oof  = oof_all @ best_w
blend_test = test_all @ best_w
blend_mae  = mean_absolute_error(y_train, blend_oof)
blend_rmse = np.sqrt(mean_squared_error(y_train, blend_oof))
blend_r2   = r2_score(y_train, blend_oof)

print("\n  Top blend weights (> 1%):")
for nm, w in sorted(zip(names_all, best_w), key=lambda x: -x[1]):
    if w > 0.01:
        print(f"    {nm:<34} {w:.4f}")
print(f"\n  Blended OOF MAE:  {blend_mae:.4f}")
print(f"  Blended OOF RMSE: {blend_rmse:.4f}   R²: {blend_r2:.4f}")

simple_oof  = oof_all.mean(axis=1)
simple_test = test_all.mean(axis=1)
simple_mae  = mean_absolute_error(y_train, simple_oof)
print(f"  Simple Avg OOF MAE: {simple_mae:.4f}")

# ============================================================
# 7b. TWO-ROUND PSEUDO-LABELING  (V6 addition)
# ============================================================
print("\n[7b/8] Two-round pseudo-labeling...")

def run_pseudo_round(base_test_pred, weight=0.1, depth=5, seed=SEED, label=''):
    """Train best CatBoost config on train + pseudo-labeled test, then blend back."""
    y_pseudo_full = pd.concat(
        [y_train, pd.Series(base_test_pred, name='Y')], ignore_index=True
    )
    X_pseudo = pd.concat([X_train, X_test], ignore_index=True)
    cb = CatBoostRegressor(
        iterations=3000, depth=depth, learning_rate=0.015,
        loss_function='MAE', l2_leaf_reg=3,
        random_seed=seed, verbose=0, subsample=0.8
    )
    cb.fit(X_pseudo, y_pseudo_full, verbose=False)
    test_pred_pseudo = cb.predict(X_test)
    blended = (1 - weight) * base_test_pred + weight * test_pred_pseudo
    print(f"    Round {label} pseudo blend ready (pseudo weight={weight})")
    return blended

# Round 1: use main blend as soft labels
pseudo_r1 = run_pseudo_round(blend_test, weight=0.10, depth=5, seed=42, label='1')
# Round 2: use Round-1 predictions as labels (tighter estimate)
pseudo_r2 = run_pseudo_round(pseudo_r1,  weight=0.10, depth=5, seed=123, label='2')

# ============================================================
# 8. VISUALIZATIONS  (all saved to plots_v6/)
# ============================================================
print("\n[8/8] Generating and saving visualizations...")

# --- Figure 1: Target Distribution + Boxplot ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Target Variable (Y) — Distribution & Box Plot', fontsize=14, fontweight='bold')

axes[0].hist(y_train, bins=40, color=COLORS[0], edgecolor='white', linewidth=0.5)
axes[0].axvline(y_train.mean(), color='crimson', ls='--', lw=1.5, label=f'Mean={y_train.mean():.2f}')
axes[0].axvline(y_train.median(), color='navy', ls='--', lw=1.5, label=f'Median={y_train.median():.2f}')
axes[0].set_xlabel('Y (Target)'); axes[0].set_ylabel('Count')
axes[0].set_title('Histogram'); axes[0].legend()

axes[1].boxplot(y_train, vert=True, patch_artist=True,
                boxprops=dict(facecolor=COLORS[0], alpha=0.7),
                medianprops=dict(color='crimson', lw=2))
axes[1].set_ylabel('Y (Target)'); axes[1].set_title('Box Plot')
plt.tight_layout()
p1 = os.path.join(PLOT_DIR, 'fig1_target_distribution.png')
plt.savefig(p1, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p1}")

# --- Figure 2: Correlation Heatmap ───────────────────────────────────────
num_cols = ['X2', 'X4', 'X6', 'Outlet_Age', 'MRP_x_OutletType',
            'MRP_squared', 'Log_MRP', 'Out_Y_mean', 'ItemID_Y_mean', 'Y']
corr_data = train_f[num_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax,
            annot_kws={'size': 9})
ax.set_title('Pearson Correlation Heatmap (Key Features vs Target Y)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
p2 = os.path.join(PLOT_DIR, 'fig2_correlation_heatmap.png')
plt.savefig(p2, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p2}")

# --- Figure 3: MRP vs Target (colored by Y) ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(train_f['X6'], y_train, c=y_train, cmap='viridis',
                alpha=0.35, s=8, rasterized=True)
plt.colorbar(sc, ax=ax, label='Y (Target)')
ax.set_xlabel('Item MRP (X6)', fontsize=12)
ax.set_ylabel('Target Y', fontsize=12)
ax.set_title('Item MRP vs Target Y  —  colored by Y value', fontsize=13, fontweight='bold')
plt.tight_layout()
p3 = os.path.join(PLOT_DIR, 'fig3_mrp_vs_target.png')
plt.savefig(p3, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p3}")

# --- Figure 4: Target by Outlet Type ─────────────────────────────────────
outlet_col = [c for c in train_f.columns if c.startswith('X11_')]
outlet_type_raw = train_df['X11']

fig, ax = plt.subplots(figsize=(10, 6))
outlet_types = sorted(outlet_type_raw.unique())
data_by_outlet = [y_train[outlet_type_raw == ot].values for ot in outlet_types]
bp = ax.boxplot(data_by_outlet, patch_artist=True, notch=False,
                medianprops=dict(color='crimson', lw=2))
for patch, color in zip(bp['boxes'], COLORS[:len(outlet_types)]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xticklabels(outlet_types, rotation=15, ha='right')
ax.set_ylabel('Target Y', fontsize=12)
ax.set_title('Target (Y) Distribution by Outlet Type (X11)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
p4 = os.path.join(PLOT_DIR, 'fig4_target_by_outlet_type.png')
plt.savefig(p4, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p4}")

# --- Figure 5: Model MAE Comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ranked = sorted(all_results.items(), key=lambda x: x[1]['MAE'])
r_names = [r[0] for r in ranked]
r_maes  = [r[1]['MAE'] for r in ranked]

colors_bar = ['#d62728' if n == r_names[0] else '#1f77b4' for n in r_names]
bars = ax.barh(r_names, r_maes, color=colors_bar, edgecolor='white', height=0.7)
ax.axvline(blend_mae, color='crimson', ls='--', lw=1.8,
           label=f'Blended Ensemble MAE = {blend_mae:.4f}')
ax.set_xlabel('OOF MAE (lower = better)', fontsize=12)
ax.set_title('OOF MAE — All Models vs Blended Ensemble (V6)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
for bar, val in zip(bars, r_maes):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
p5 = os.path.join(PLOT_DIR, 'fig5_model_mae_comparison.png')
plt.savefig(p5, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p5}")

# --- Figure 6: Actual vs Predicted + Residuals ───────────────────────────
residuals = y_train.values - blend_oof
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Blended Ensemble — Actual vs Predicted  |  RMSE={blend_rmse:.4f}  R²={blend_r2:.4f}',
             fontsize=13, fontweight='bold')

axes[0].scatter(y_train, blend_oof, alpha=0.25, s=8, color=COLORS[0], rasterized=True)
mn, mx = y_train.min(), y_train.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
axes[0].set_xlabel('Actual Y'); axes[0].set_ylabel('Predicted Y')
axes[0].set_title('Actual vs Predicted'); axes[0].legend()

axes[1].scatter(blend_oof, residuals, alpha=0.25, s=8, color=COLORS[1], rasterized=True)
axes[1].axhline(0, color='crimson', ls='--', lw=1.5)
axes[1].set_xlabel('Predicted Y'); axes[1].set_ylabel('Residual (Actual – Predicted)')
axes[1].set_title('Residual Plot')

plt.tight_layout()
p6 = os.path.join(PLOT_DIR, 'fig6_actual_vs_predicted_residuals.png')
plt.savefig(p6, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p6}")

# ============================================================
# SUBMISSIONS
# ============================================================
sub1 = sample_sub.copy(); sub1['Y'] = blend_test
sub1.to_csv(os.path.join(SUB_DIR, 'submission_v6_blend.csv'), index=False)

sub2 = sample_sub.copy(); sub2['Y'] = pseudo_r1
sub2.to_csv(os.path.join(SUB_DIR, 'submission_v6_pseudo_r1.csv'), index=False)

sub3 = sample_sub.copy(); sub3['Y'] = pseudo_r2
sub3.to_csv(os.path.join(SUB_DIR, 'submission_v6_pseudo_r2.csv'), index=False)

sub4 = sample_sub.copy(); sub4['Y'] = stack_test
sub4.to_csv(os.path.join(SUB_DIR, 'submission_v6_stack.csv'), index=False)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ALL MODELS — ranked by OOF MAE (lower = better)")
print("=" * 70)
for i, (nm, r) in enumerate(sorted(all_results.items(), key=lambda x: x[1]['MAE'])):
    flag = " ★ BEST" if i == 0 else ""
    print(f"  {i+1:>2}. {nm:<34} MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}{flag}")

print(f"\n  >> Optimized Blend (multi-start)  MAE={blend_mae:.4f}  RMSE={blend_rmse:.4f}  R²={blend_r2:.4f}")
print(f"  >> Simple Average                 MAE={simple_mae:.4f}")
print(f"  >> Stack (Ridge meta)             MAE={stack_mae:.4f}")

print(f"""
SUBMISSION ORDER (try in order, keep whichever scores lowest on leaderboard):
  1. submissions_v6/submission_v6_blend.csv     — main bet (optimized blend)
  2. submissions_v6/submission_v6_pseudo_r2.csv — pseudo-labeled round 2
  3. submissions_v6/submission_v6_pseudo_r1.csv — pseudo-labeled round 1
  4. submissions_v6/submission_v6_stack.csv     — stacked meta-learner

PLOTS saved to: {PLOT_DIR}
  fig1_target_distribution.png
  fig2_correlation_heatmap.png
  fig3_mrp_vs_target.png
  fig4_target_by_outlet_type.png
  fig5_model_mae_comparison.png
  fig6_actual_vs_predicted_residuals.png

Seed=42 — all models fully reproducible.
""")
print("=" * 70)
print("V6 complete.")
print("=" * 70)
