"""
NFL Draft Prediction - V5 (Target: 0.87 on OmniCampus)
======================================================
Key improvements over V4:
1. KNN Imputation for missing physical metrics.
2. Unsupervised Learning Features: KMeans Clustering Distances.
3. Neural Network Integration: Added MLPClassifier to the ensemble.
4. Native CatBoost Categorical Encoding.
"""

import warnings
warnings.filterwarnings("ignore")
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import os

# Config
N_FOLDS = 10
N_SEEDS = 5
OPTUNA_TRIALS = 30 # Reduced to 30 because we have 4 models now
INPUT_DIR = "input"
OUTPUT_FILE = "submission_v5.csv"

NUMERIC_COLS = [
    "Age", "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"
]
DRILL_COLS = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
              "Broad_Jump", "Agility_3cone", "Shuttle"]
CAT_COLS = ["Player_Type", "Position_Type", "Position"]
TARGET = "Drafted"

POS_GROUP = {
    'CB': 'DB', 'FS': 'DB', 'SS': 'DB', 'S': 'DB', 'DB': 'DB',
    'DE': 'DL', 'DT': 'DL',
    'OT': 'OL', 'OG': 'OL', 'C': 'OL',
    'OLB': 'LB', 'ILB': 'LB',
    'WR': 'SKILL', 'RB': 'SKILL', 'QB': 'SKILL', 'TE': 'SKILL', 'FB': 'SKILL',
    'K': 'SPEC', 'P': 'SPEC', 'LS': 'SPEC'
}

print("=" * 70); sys.stdout.flush()
print("NFL Draft Prediction - V5 (Deep Ensemble + Clustering)"); sys.stdout.flush()
print("=" * 70); sys.stdout.flush()

# === 1. Load Data ===
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
test_ids = test_df["Id"].values
y_train_full = train_df[TARGET].values

train_df.drop(columns=['Id', TARGET], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

df_all = pd.concat([train_df, test_df], ignore_index=True)

# === 2. Base Feature Engineering ===
# Missing flags (very important)
for col in NUMERIC_COLS:
    df_all[f"{col}_missing"] = df_all[col].isnull().astype(int)

df_all["n_missing_drills"] = df_all[DRILL_COLS].isnull().sum(axis=1)
df_all["n_completed_drills"] = 6 - df_all["n_missing_drills"]

# KNN Imputation for Numerics
print("Applying KNN Imputation..."); sys.stdout.flush()
imputer = KNNImputer(n_neighbors=5, weights="distance")
df_all[NUMERIC_COLS] = imputer.fit_transform(df_all[NUMERIC_COLS])

# Body features
df_all["BMI"] = (df_all["Weight"] * 0.453592) / ((df_all["Height"] * 0.0254) ** 2)
df_all["Weight_per_inch"] = df_all["Weight"] / df_all["Height"]
df_all["Height_Weight_ratio"] = df_all["Height"] / df_all["Weight"]

# NFL Scouting Metrics
df_all["Speed_Score"] = (df_all["Weight"] * 200) / (df_all["Sprint_40yd"] ** 4 + 1e-8)
df_all["Height_Adj_Speed"] = df_all["Sprint_40yd"] / (df_all["Height"] + 1e-8)
df_all["Explosion_Index"] = df_all["Vertical_Jump"] * df_all["Broad_Jump"] / 1000
df_all["Bench_Weight_Ratio"] = df_all["Bench_Press_Reps"] / (df_all["Weight"] + 1e-8)
df_all["Agility_Score"] = 1.0 / (df_all["Agility_3cone"] * df_all["Shuttle"] + 1e-8)

# Pos Group
df_all["Pos_Group"] = df_all["Position"].map(POS_GROUP).fillna("OTHER")

# Percentile Ranks
for col in NUMERIC_COLS:
    pctile_col = f"{col}_pos_pctile"
    df_all[pctile_col] = df_all.groupby("Position")[col].rank(pct=True)

# School grouping
school_counts = df_all["School"].value_counts()
valid_schools = set(school_counts[school_counts >= 5].index)
df_all["School_Grouped"] = df_all["School"].apply(lambda x: x if x in valid_schools else "Other")

# === 3. Unsupervised Clustering Features ===
print("Extracting KMeans Clustering Features..."); sys.stdout.flush()
cluster_cols = NUMERIC_COLS + ["BMI"]
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_all[cluster_cols])

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df_all["Cluster_ID"] = kmeans.fit_predict(df_cluster_scaled)

# Distance to cluster centers
distances = kmeans.transform(df_cluster_scaled)
for i in range(8):
    df_all[f"Dist_to_Cluster_{i}"] = distances[:, i]

# === 4. Encoding for different model types ===
cat_features = CAT_COLS + ["School_Grouped", "Pos_Group", "Cluster_ID"]

# Convert categoricals to strings for CatBoost
for c in cat_features:
    df_all[c] = df_all[c].astype(str)

df_all["Year_norm"] = (df_all["Year"] - 2009) / (2019 - 2009)
df_all.drop(columns=["School", "Year"], inplace=True)

X_train_full = df_all.iloc[:len(train_df)].copy()
X_test_full = df_all.iloc[len(train_df):].copy()
y = y_train_full

# OOF Target Encoding (for XGB/LGB/MLP, CatBoost does its own)
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
        
        # Drop original string categorical for models other than CatBoost
        X_tr_enc.drop(columns=[col], inplace=True)
        X_val_enc.drop(columns=[col], inplace=True)
        X_test_enc.drop(columns=[col], inplace=True)

    return X_tr_enc, X_val_enc, X_test_enc

# Ensure XGB/LGB features are fully numeric
print(f"Total features: {len(X_train_full.columns)}"); sys.stdout.flush()

# === 5. Optuna Tuning ===
print(f"\n[Phase 1] Optuna Tuning ({OPTUNA_TRIALS} trials per model)..."); sys.stdout.flush()

# We tune XGB, LGB, CB, and MLP
def optuna_mlp(trial, X, y_arr, cat_feats):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64, 32), (128, 64), (64, 64, 32)]),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'max_iter': 300,
        'early_stopping': True,
        'random_state': 42
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y_arr):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
        X_tr_enc, X_val_enc, _ = oof_target_encode(X_tr, y_tr, X_val, X, cat_feats)
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_enc)
        X_val_scaled = scaler.transform(X_val_enc)
        
        m = MLPClassifier(**params)
        m.fit(X_tr_scaled, y_tr)
        scores.append(roc_auc_score(y_val, m.predict_proba(X_val_scaled)[:, 1]))
    return np.mean(scores)

print("  Tuning MLP (Neural Network)..."); sys.stdout.flush()
study_mlp = optuna.create_study(direction='maximize')
study_mlp.optimize(lambda t: optuna_mlp(t, X_train_full, y, cat_features), n_trials=OPTUNA_TRIALS)
mlp_params = study_mlp.best_params
mlp_params.update({'max_iter': 400, 'early_stopping': True})
print(f"    Best MLP CV: {study_mlp.best_value:.6f}"); sys.stdout.flush()

def optuna_xgb(trial, X, y_arr, cat_feats):
    params = {
        'n_estimators': 800,
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'random_state': 42, 'eval_metric': 'auc', 'early_stopping_rounds': 50,
        'tree_method': 'hist'
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y_arr):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
        X_tr_enc, X_val_enc, _ = oof_target_encode(X_tr, y_tr, X_val, X, cat_feats)
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=0)
        scores.append(roc_auc_score(y_val, m.predict_proba(X_val_enc)[:, 1]))
    return np.mean(scores)

print("  Tuning XGBoost..."); sys.stdout.flush()
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(lambda t: optuna_xgb(t, X_train_full, y, cat_features), n_trials=OPTUNA_TRIALS)
xgb_params = study_xgb.best_params
xgb_params.update({'n_estimators': 1000, 'eval_metric': 'auc', 'early_stopping_rounds': 50, 'tree_method': 'hist'})
print(f"    Best XGB CV: {study_xgb.best_value:.6f}"); sys.stdout.flush()

def optuna_lgb(trial, X, y_arr, cat_feats):
    params = {
        'n_estimators': 800,
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'random_state': 42, 'verbose': -1
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y_arr):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
        X_tr_enc, X_val_enc, _ = oof_target_encode(X_tr, y_tr, X_val, X, cat_feats)
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        scores.append(roc_auc_score(y_val, m.predict_proba(X_val_enc)[:, 1]))
    return np.mean(scores)

print("  Tuning LightGBM..."); sys.stdout.flush()
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(lambda t: optuna_lgb(t, X_train_full, y, cat_features), n_trials=OPTUNA_TRIALS)
lgb_params = study_lgb.best_params
lgb_params.update({'n_estimators': 1000, 'verbose': -1})
print(f"    Best LGB CV: {study_lgb.best_value:.6f}"); sys.stdout.flush()

def optuna_cb(trial, X, y_arr, cat_feats):
    params = {
        'iterations': 800,
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
        'random_seed': 42, 'verbose': 0, 'eval_metric': 'AUC',
        'early_stopping_rounds': 50, 'cat_features': cat_feats
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y_arr):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
        m = cb.CatBoostClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        scores.append(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))
    return np.mean(scores)

print("  Tuning CatBoost..."); sys.stdout.flush()
study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(lambda t: optuna_cb(t, X_train_full, y, cat_features), n_trials=OPTUNA_TRIALS)
cb_params = study_cb.best_params
cb_params.update({'iterations': 1000, 'verbose': 0, 'eval_metric': 'AUC',
                   'early_stopping_rounds': 50, 'cat_features': cat_features})
print(f"    Best CB CV: {study_cb.best_value:.6f}"); sys.stdout.flush()

# === 6. Multi-Seed Ensemble Training ===
print(f"\n[Phase 2] Multi-Seed 4-Model Ensemble ({N_SEEDS} seeds x 4 models x {N_FOLDS} folds)..."); sys.stdout.flush()

all_test_preds = []
all_oof_aucs = []

for seed_idx in range(N_SEEDS):
    seed = 42 + seed_idx * 7

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = list(skf.split(X_train_full, y))

    xgb_oof = np.zeros(len(y)); xgb_test = np.zeros(len(X_test_full))
    lgb_oof = np.zeros(len(y)); lgb_test = np.zeros(len(X_test_full))
    cb_oof = np.zeros(len(y));  cb_test = np.zeros(len(X_test_full))
    mlp_oof = np.zeros(len(y)); mlp_test = np.zeros(len(X_test_full))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_train_full.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X_train_full.iloc[val_idx], y[val_idx]
        
        # OOF Encode for XGB, LGB, MLP
        X_tr_enc, X_val_enc, X_test_enc = oof_target_encode(X_tr, y_tr, X_val, X_test_full, cat_features)

        # XGBoost
        m_xgb = xgb.XGBClassifier(**{**xgb_params, 'random_state': seed})
        m_xgb.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=0)
        xgb_oof[val_idx] = m_xgb.predict_proba(X_val_enc)[:, 1]
        xgb_test += m_xgb.predict_proba(X_test_enc)[:, 1] / N_FOLDS

        # LightGBM
        m_lgb = lgb.LGBMClassifier(**{**lgb_params, 'random_state': seed})
        m_lgb.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_oof[val_idx] = m_lgb.predict_proba(X_val_enc)[:, 1]
        lgb_test += m_lgb.predict_proba(X_test_enc)[:, 1] / N_FOLDS

        # MLP (Neural Network)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_enc)
        X_val_s = scaler.transform(X_val_enc)
        X_test_s = scaler.transform(X_test_enc)
        m_mlp = MLPClassifier(**{**mlp_params, 'random_state': seed})
        m_mlp.fit(X_tr_s, y_tr)
        mlp_oof[val_idx] = m_mlp.predict_proba(X_val_s)[:, 1]
        mlp_test += m_mlp.predict_proba(X_test_s)[:, 1] / N_FOLDS

        # CatBoost (Native string categoricals)
        m_cb = cb.CatBoostClassifier(**{**cb_params, 'random_seed': seed})
        m_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        cb_oof[val_idx] = m_cb.predict_proba(X_val)[:, 1]
        cb_test += m_cb.predict_proba(X_test_full)[:, 1] / N_FOLDS

    xgb_auc = roc_auc_score(y, xgb_oof)
    lgb_auc = roc_auc_score(y, lgb_oof)
    cb_auc = roc_auc_score(y, cb_oof)
    mlp_auc = roc_auc_score(y, mlp_oof)
    print(f"  Seed {seed}: XGB={xgb_auc:.4f} LGB={lgb_auc:.4f} CB={cb_auc:.4f} MLP={mlp_auc:.4f}"); sys.stdout.flush()

    # Rank Average
    xgb_test_r = rankdata(xgb_test) / len(xgb_test)
    lgb_test_r = rankdata(lgb_test) / len(lgb_test)
    cb_test_r = rankdata(cb_test) / len(cb_test)
    mlp_test_r = rankdata(mlp_test) / len(mlp_test)

    # Weights by OOF AUC
    total_w = xgb_auc + lgb_auc + cb_auc + mlp_auc
    w_xgb, w_lgb, w_cb, w_mlp = xgb_auc/total_w, lgb_auc/total_w, cb_auc/total_w, mlp_auc/total_w

    seed_test = w_xgb*xgb_test_r + w_lgb*lgb_test_r + w_cb*cb_test_r + w_mlp*mlp_test_r
    all_test_preds.append(seed_test)

    # Ensure OOF tracks similarly
    xgb_oof_r = rankdata(xgb_oof) / len(xgb_oof)
    lgb_oof_r = rankdata(lgb_oof) / len(lgb_oof)
    cb_oof_r = rankdata(cb_oof) / len(cb_oof)
    mlp_oof_r = rankdata(mlp_oof) / len(mlp_oof)
    oof_ens = w_xgb*xgb_oof_r + w_lgb*lgb_oof_r + w_cb*cb_oof_r + w_mlp*mlp_oof_r
    oof_auc = roc_auc_score(y, oof_ens)
    all_oof_aucs.append(oof_auc)
    print(f"    Ensemble OOF AUC: {oof_auc:.6f}"); sys.stdout.flush()

final_test = np.mean(all_test_preds, axis=0)
print(f"\n  Mean 4-Model OOF AUC across seeds: {np.mean(all_oof_aucs):.6f}"); sys.stdout.flush()

# === Generate Submission ===
print(f"\n[Phase 3] Generating {OUTPUT_FILE}..."); sys.stdout.flush()
submission = pd.DataFrame({"Id": test_ids, "Drafted": final_test})
submission.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved {OUTPUT_FILE}")
print("Done!")
