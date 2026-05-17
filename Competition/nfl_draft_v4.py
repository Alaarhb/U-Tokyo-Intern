"""
NFL Draft Prediction - V4 (Target: 0.87 on OmniCampus)
======================================================
Key improvements over V3:
1. NFL scouting metrics (Speed Score, Explosion Index, etc.)
2. Position percentile ranks for each drill
3. Position group encoding (6 broad groups)
4. Missing pattern fingerprint
5. Pairwise drill ratios
6. Multi-seed ensemble (5 seeds x 3 models = 15 models)
7. Optuna hyperparameter tuning (150 trials/model)
8. Weighted rank averaging
"""

import warnings
warnings.filterwarnings("ignore")
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import os, hashlib

# Config
N_FOLDS = 10
N_SEEDS = 5
OPTUNA_TRIALS = 50
INPUT_DIR = "input"
OUTPUT_FILE = "submission_v4.csv"

NUMERIC_COLS = [
    "Age", "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"
]
DRILL_COLS = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
              "Broad_Jump", "Agility_3cone", "Shuttle"]
CAT_COLS = ["Player_Type", "Position_Type", "Position"]
TARGET = "Drafted"

# Position group mapping
POS_GROUP = {
    'CB': 'DB', 'FS': 'DB', 'SS': 'DB', 'S': 'DB', 'DB': 'DB',
    'DE': 'DL', 'DT': 'DL',
    'OT': 'OL', 'OG': 'OL', 'C': 'OL',
    'OLB': 'LB', 'ILB': 'LB',
    'WR': 'SKILL', 'RB': 'SKILL', 'QB': 'SKILL', 'TE': 'SKILL', 'FB': 'SKILL',
    'K': 'SPEC', 'P': 'SPEC', 'LS': 'SPEC'
}

def engineer_features(df, train_stats=None, is_train=True):
    df = df.copy()

    # === Missing flags ===
    for col in NUMERIC_COLS:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    df["n_missing_drills"] = df[DRILL_COLS].isnull().sum(axis=1)
    df["n_completed_drills"] = 6 - df["n_missing_drills"]
    df["all_drills_complete"] = (df["n_missing_drills"] == 0).astype(int)

    # Missing pattern fingerprint
    missing_pattern = ""
    for col in DRILL_COLS:
        missing_pattern = df[DRILL_COLS].isnull().astype(int).astype(str).apply(''.join, axis=1)
    df["missing_pattern"] = pd.Categorical(missing_pattern).codes

    # === Body features ===
    df["BMI"] = (df["Weight"] * 0.453592) / ((df["Height"] * 0.0254) ** 2)
    df["Weight_per_inch"] = df["Weight"] / df["Height"]
    df["Height_Weight_ratio"] = df["Height"] / df["Weight"]

    # === NFL Scouting Metrics ===
    df["Speed_Score"] = (df["Weight"] * 200) / (df["Sprint_40yd"] ** 4 + 1e-8)
    df["Height_Adj_Speed"] = df["Sprint_40yd"] / (df["Height"] + 1e-8)
    df["Explosion_Index"] = df["Vertical_Jump"] * df["Broad_Jump"] / 1000
    df["Bench_Weight_Ratio"] = df["Bench_Press_Reps"] / (df["Weight"] + 1e-8)
    df["Agility_Score"] = 1.0 / (df["Agility_3cone"] * df["Shuttle"] + 1e-8)

    # === Composite features ===
    df["Speed_Agility"] = df["Sprint_40yd"] * df["Agility_3cone"]
    df["Explosiveness"] = df["Vertical_Jump"] + df["Broad_Jump"]
    df["Strength_Speed"] = df["Bench_Press_Reps"] / (df["Sprint_40yd"] + 1e-8)
    df["Agility_Shuttle_Avg"] = (df["Agility_3cone"] + df["Shuttle"]) / 2
    df["Power_Score"] = df["Bench_Press_Reps"] * df["Vertical_Jump"]
    df["Lower_Body_Power"] = df["Vertical_Jump"] * df["Broad_Jump"] * df["Weight"]
    df["Upper_Lower_Ratio"] = df["Bench_Press_Reps"] / (df["Vertical_Jump"] + 1e-8)

    # === Pairwise drill ratios (top pairs only) ===
    pairs = [
        ("Sprint_40yd", "Weight"), ("Vertical_Jump", "Weight"),
        ("Broad_Jump", "Sprint_40yd"), ("Bench_Press_Reps", "Sprint_40yd"),
        ("Agility_3cone", "Sprint_40yd"), ("Shuttle", "Sprint_40yd"),
        ("Vertical_Jump", "Broad_Jump"), ("Bench_Press_Reps", "Weight"),
    ]
    for c1, c2 in pairs:
        df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-8)

    # === Position group ===
    df["Pos_Group"] = df["Position"].map(POS_GROUP).fillna("OTHER")

    # === Position percentile ranks ===
    if is_train:
        pos_stats = {}
        pos_percentiles = {}
        for col in NUMERIC_COLS:
            grp = df.groupby("Position")[col].agg(["mean", "std"])
            grp["std"] = grp["std"].replace(0, 1)
            pos_stats[col] = grp.to_dict("index")
        train_stats = {"pos_zscores": pos_stats}

    for col in NUMERIC_COLS:
        # Z-scores
        zscore_col = f"{col}_pos_zscore"
        df[zscore_col] = np.nan
        # Percentile ranks within position
        pctile_col = f"{col}_pos_pctile"
        df[pctile_col] = np.nan

        for pos in df["Position"].unique():
            mask = df["Position"] == pos
            # Z-score
            stats = train_stats["pos_zscores"][col].get(pos)
            if stats and stats["std"] and stats["std"] > 0:
                df.loc[mask, zscore_col] = (df.loc[mask, col] - stats["mean"]) / stats["std"]
            # Percentile rank within position
            vals = df.loc[mask, col]
            if vals.notna().sum() > 1:
                df.loc[mask, pctile_col] = vals.rank(pct=True)

    # === Position group percentile ranks ===
    for col in DRILL_COLS:
        pctile_col = f"{col}_grp_pctile"
        df[pctile_col] = np.nan
        for grp in df["Pos_Group"].unique():
            mask = df["Pos_Group"] == grp
            vals = df.loc[mask, col]
            if vals.notna().sum() > 1:
                df.loc[mask, pctile_col] = vals.rank(pct=True)

    # === School grouping ===
    if is_train:
        school_counts = df["School"].value_counts()
        valid_schools = set(school_counts[school_counts >= 5].index)
        train_stats["valid_schools"] = valid_schools
    else:
        valid_schools = train_stats["valid_schools"]

    df["School_Grouped"] = df["School"].apply(lambda x: x if x in valid_schools else "Other")

    # === Categories ===
    for col in CAT_COLS + ["School_Grouped", "Pos_Group"]:
        df[col] = df[col].astype("category")

    df["Year_norm"] = (df["Year"] - 2009) / (2019 - 2009)
    df = df.drop(columns=["School", "Id"])

    return df, train_stats

# === OOF Target Encoding ===
def oof_target_encode(X_tr, y_tr, X_val, X_test, cols):
    global_mean = y_tr.mean()
    smoothing = 10
    X_tr_enc, X_val_enc, X_test_enc = X_tr.copy(), X_val.copy(), X_test.copy()

    for col in cols:
        tmp = X_tr_enc.copy()
        tmp['_target'] = y_tr
        counts = tmp.groupby(col).size()
        sums = tmp.groupby(col)['_target'].sum()
        enc_map = (sums + smoothing * global_mean) / (counts + smoothing)

        new_col = f"{col}_TE"
        X_tr_enc[new_col] = X_tr[col].map(enc_map).astype(float).fillna(global_mean)
        X_val_enc[new_col] = X_val[col].map(enc_map).astype(float).fillna(global_mean)
        X_test_enc[new_col] = X_test[col].map(enc_map).astype(float).fillna(global_mean)

    return X_tr_enc, X_val_enc, X_test_enc

# === Load & Engineer ===
print("=" * 70); sys.stdout.flush()
print("NFL Draft Prediction - V4 (Multi-Seed Ensemble + Optuna)"); sys.stdout.flush()
print("=" * 70); sys.stdout.flush()

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
test_ids = test_df["Id"].values

train_fe, train_stats = engineer_features(train_df, is_train=True)
test_fe, _ = engineer_features(test_df, train_stats=train_stats, is_train=False)

y = train_fe[TARGET].values
feature_cols = [c for c in train_fe.columns if c != TARGET]
for c in feature_cols:
    if c not in test_fe.columns:
        test_fe[c] = 0
common_cols = [c for c in feature_cols if c in test_fe.columns]
X_train = train_fe[common_cols].copy()
X_test = test_fe[common_cols].copy()

cat_features = [c for c in X_train.columns if X_train[c].dtype.name == 'category']
print(f"Features: {len(common_cols)} ({len(cat_features)} categorical)"); sys.stdout.flush()

# === Optuna Tuning ===
def optuna_xgb(trial, X, y_arr, cat_feats):
    params = {
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'random_state': 42, 'eval_metric': 'auc', 'early_stopping_rounds': 50,
        'enable_categorical': True, 'tree_method': 'hist'
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y_arr):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
        X_tr, X_val, _ = oof_target_encode(X_tr, y_tr, X_val, X, cat_feats)
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        scores.append(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
    return np.mean(scores)

def optuna_lgb(trial, X, y_arr, cat_feats):
    params = {
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': 42, 'verbose': -1
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y_arr):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
        X_tr, X_val, _ = oof_target_encode(X_tr, y_tr, X_val, X, cat_feats)
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        scores.append(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
    return np.mean(scores)

def optuna_cb(trial, X, y_arr, cat_feats):
    params = {
        'iterations': 1000,
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2),
        'random_strength': trial.suggest_float('random_strength', 0.1, 5),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42, 'verbose': 0, 'eval_metric': 'AUC',
        'early_stopping_rounds': 50, 'cat_features': cat_feats
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y_arr):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
        X_tr, X_val, _ = oof_target_encode(X_tr, y_tr, X_val, X, cat_feats)
        X_tr_cb, X_val_cb = X_tr.copy(), X_val.copy()
        for c in cat_feats:
            X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("missing")
            X_val_cb[c] = X_val_cb[c].astype(str).fillna("missing")
        m = cb.CatBoostClassifier(**params)
        m.fit(X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), verbose=0)
        scores.append(roc_auc_score(y_val, m.predict_proba(X_val_cb)[:, 1]))
    return np.mean(scores)

# Run Optuna
print(f"\n[Phase 1] Optuna Tuning ({OPTUNA_TRIALS} trials per model)..."); sys.stdout.flush()

print("  Tuning XGBoost..."); sys.stdout.flush()
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(lambda t: optuna_xgb(t, X_train, y, cat_features), n_trials=OPTUNA_TRIALS)
xgb_params = study_xgb.best_params
xgb_params.update({'n_estimators': 1000, 'eval_metric': 'auc', 'early_stopping_rounds': 50,
                    'enable_categorical': True, 'tree_method': 'hist'})
print(f"    Best XGB CV: {study_xgb.best_value:.6f}"); sys.stdout.flush()

print("  Tuning LightGBM..."); sys.stdout.flush()
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(lambda t: optuna_lgb(t, X_train, y, cat_features), n_trials=OPTUNA_TRIALS)
lgb_params = study_lgb.best_params
lgb_params.update({'n_estimators': 1000, 'verbose': -1})
print(f"    Best LGB CV: {study_lgb.best_value:.6f}"); sys.stdout.flush()

print("  Tuning CatBoost..."); sys.stdout.flush()
study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(lambda t: optuna_cb(t, X_train, y, cat_features), n_trials=OPTUNA_TRIALS)
cb_params = study_cb.best_params
cb_params.update({'iterations': 1000, 'verbose': 0, 'eval_metric': 'AUC',
                   'early_stopping_rounds': 50, 'cat_features': cat_features})
print(f"    Best CB CV: {study_cb.best_value:.6f}"); sys.stdout.flush()

# === Multi-Seed Ensemble Training ===
print(f"\n[Phase 2] Multi-Seed Ensemble ({N_SEEDS} seeds x 3 models x {N_FOLDS} folds)..."); sys.stdout.flush()

all_test_preds = []
all_oof_aucs = []

for seed_idx in range(N_SEEDS):
    seed = 42 + seed_idx * 7

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = list(skf.split(X_train, y))

    xgb_oof = np.zeros(len(y)); xgb_test = np.zeros(len(X_test))
    lgb_oof = np.zeros(len(y)); lgb_test = np.zeros(len(X_test))
    cb_oof = np.zeros(len(y));  cb_test = np.zeros(len(X_test))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y[val_idx]
        X_tr_enc, X_val_enc, X_test_enc = oof_target_encode(X_tr, y_tr, X_val, X_test, cat_features)

        # XGBoost
        xgb_p = {**xgb_params, 'random_state': seed}
        m_xgb = xgb.XGBClassifier(**xgb_p)
        m_xgb.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=0)
        xgb_oof[val_idx] = m_xgb.predict_proba(X_val_enc)[:, 1]
        xgb_test += m_xgb.predict_proba(X_test_enc)[:, 1] / N_FOLDS

        # LightGBM
        lgb_p = {**lgb_params, 'random_state': seed}
        m_lgb = lgb.LGBMClassifier(**lgb_p)
        m_lgb.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_oof[val_idx] = m_lgb.predict_proba(X_val_enc)[:, 1]
        lgb_test += m_lgb.predict_proba(X_test_enc)[:, 1] / N_FOLDS

        # CatBoost
        X_tr_cb, X_val_cb, X_test_cb = X_tr_enc.copy(), X_val_enc.copy(), X_test_enc.copy()
        for c in cat_features:
            X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("missing")
            X_val_cb[c] = X_val_cb[c].astype(str).fillna("missing")
            X_test_cb[c] = X_test_cb[c].astype(str).fillna("missing")
        cb_p = {**cb_params, 'random_seed': seed}
        m_cb = cb.CatBoostClassifier(**cb_p)
        m_cb.fit(X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), verbose=0)
        cb_oof[val_idx] = m_cb.predict_proba(X_val_cb)[:, 1]
        cb_test += m_cb.predict_proba(X_test_cb)[:, 1] / N_FOLDS

    xgb_auc = roc_auc_score(y, xgb_oof)
    lgb_auc = roc_auc_score(y, lgb_oof)
    cb_auc = roc_auc_score(y, cb_oof)
    print(f"  Seed {seed}: XGB={xgb_auc:.4f}  LGB={lgb_auc:.4f}  CB={cb_auc:.4f}"); sys.stdout.flush()

    # Weighted rank averaging for this seed
    xgb_test_r = rankdata(xgb_test) / len(xgb_test)
    lgb_test_r = rankdata(lgb_test) / len(lgb_test)
    cb_test_r = rankdata(cb_test) / len(cb_test)

    # Weight by OOF AUC
    total_w = xgb_auc + lgb_auc + cb_auc
    w_xgb = xgb_auc / total_w
    w_lgb = lgb_auc / total_w
    w_cb = cb_auc / total_w

    seed_test = w_xgb * xgb_test_r + w_lgb * lgb_test_r + w_cb * cb_test_r
    all_test_preds.append(seed_test)

    # Also compute OOF ensemble AUC
    xgb_oof_r = rankdata(xgb_oof) / len(xgb_oof)
    lgb_oof_r = rankdata(lgb_oof) / len(lgb_oof)
    cb_oof_r = rankdata(cb_oof) / len(cb_oof)
    oof_ens = w_xgb * xgb_oof_r + w_lgb * lgb_oof_r + w_cb * cb_oof_r
    oof_auc = roc_auc_score(y, oof_ens)
    all_oof_aucs.append(oof_auc)
    print(f"    Ensemble OOF AUC: {oof_auc:.6f}"); sys.stdout.flush()

# Average across all seeds
final_test = np.mean(all_test_preds, axis=0)
print(f"\n  Mean OOF AUC across seeds: {np.mean(all_oof_aucs):.6f}")
print(f"  Std OOF AUC across seeds:  {np.std(all_oof_aucs):.6f}")

# === Generate Submission ===
print(f"\n[Phase 3] Generating {OUTPUT_FILE}...")
submission = pd.DataFrame({"Id": test_ids, "Drafted": final_test})
submission.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved {OUTPUT_FILE}")
print("Done!")
