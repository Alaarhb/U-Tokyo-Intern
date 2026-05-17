"""
NFL Draft Prediction — Version 2 (Robust)
=========================================
Addresses overfitting by using 10-fold CV, native missing value handling,
OOF target encoding for Schools, and Rank Averaging for the ensemble.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import os
import time

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Configuration ──────────────────────────────────────────────────────────────
N_FOLDS = 10
RANDOM_STATE = 42
OPTUNA_TRIALS = 150  # 150 trials per model
INPUT_DIR = "input"
OUTPUT_FILE = "submission_v2.csv"

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("NFL Draft Prediction — Robust Ensemble (V2)")
print("=" * 70)

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("\n[Step 2] Feature Engineering (Robust)...")

NUMERIC_COLS = [
    "Age", "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"
]
CAT_COLS = ["Player_Type", "Position_Type", "Position"]
TARGET = "Drafted"

def engineer_features(df, train_stats=None, is_train=True):
    df = df.copy()

    # ── 2a. Missing value indicators (but DO NOT impute) ──
    for col in NUMERIC_COLS:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    drill_cols = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
                  "Broad_Jump", "Agility_3cone", "Shuttle"]
    df["n_missing_drills"] = df[drill_cols].isnull().sum(axis=1)
    df["n_completed_drills"] = 6 - df["n_missing_drills"]

    # ── 2b. Body composition features ──
    df["BMI"] = (df["Weight"] * 0.453592) / ((df["Height"] * 0.0254) ** 2)
    df["Weight_per_inch"] = df["Weight"] / df["Height"]
    df["Height_Weight_ratio"] = df["Height"] / df["Weight"]

    # ── 2c. Athletic performance composites ──
    df["Speed_Agility"] = df["Sprint_40yd"] * df["Agility_3cone"]
    df["Explosiveness"] = df["Vertical_Jump"] + df["Broad_Jump"]
    df["Strength_Speed"] = df["Bench_Press_Reps"] / (df["Sprint_40yd"] + 1e-8)
    df["Agility_Shuttle_Avg"] = (df["Agility_3cone"] + df["Shuttle"]) / 2
    df["Power_Score"] = df["Bench_Press_Reps"] * df["Vertical_Jump"]

    # ── 2d. Position-relative z-scores (ignoring NaNs) ──
    if is_train:
        pos_stats = {}
        for col in NUMERIC_COLS:
            grp = df.groupby("Position")[col].agg(["mean", "std"])
            grp["std"] = grp["std"].replace(0, 1)  # avoid div by zero
            pos_stats[col] = grp.to_dict("index")
        train_stats = {"pos_zscores": pos_stats}

    for col in NUMERIC_COLS:
        zscore_col = f"{col}_pos_zscore"
        df[zscore_col] = np.nan
        for pos in df["Position"].unique():
            mask = df["Position"] == pos
            stats = train_stats["pos_zscores"][col].get(pos)
            if stats and stats["std"] and stats["std"] > 0:
                df.loc[mask, zscore_col] = (
                    (df.loc[mask, col] - stats["mean"]) / stats["std"]
                )

    # ── 2e. Group rare Schools ──
    if is_train:
        school_counts = df["School"].value_counts()
        valid_schools = set(school_counts[school_counts >= 5].index)
        train_stats["valid_schools"] = valid_schools
    else:
        valid_schools = train_stats["valid_schools"]
    
    df["School_Grouped"] = df["School"].apply(lambda x: x if x in valid_schools else "Other")

    # ── 2f. Cast categorical columns to Category dtype ──
    for col in CAT_COLS + ["School_Grouped"]:
        df[col] = df[col].astype("category")

    df["Year_norm"] = (df["Year"] - 2009) / (2019 - 2009)
    df = df.drop(columns=["School", "Id"])
    
    return df, train_stats

train_fe, train_stats = engineer_features(train_df, is_train=True)
test_fe, _ = engineer_features(test_df, train_stats=train_stats, is_train=False)

# ── 2g. OOF Target Encoding for School_Grouped ──
# To prevent leakage, we encode School within the CV loop later.
# For now, just keep it as a categorical feature.

y = train_fe[TARGET].values
test_ids = test_df["Id"].values

drop_from_X = [TARGET]
feature_cols = [c for c in train_fe.columns if c not in drop_from_X]

# Align columns
for c in feature_cols:
    if c not in test_fe.columns:
        test_fe[c] = 0
common_cols = [c for c in feature_cols if c in test_fe.columns]
X_train_df = train_fe[common_cols]
X_test_df = test_fe[common_cols]

print(f"  Features: {len(common_cols)}")
print(f"  X_train shape: {X_train_df.shape}")
print(f"  X_test shape:  {X_test_df.shape}")

# ── 3. Optuna Tuning + Training (10-Fold CV) ──────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds = list(skf.split(X_train_df, y))

# Identify categorical features for native handling
cat_features = [c for c in X_train_df.columns if X_train_df[c].dtype.name == 'category']
print(f"  Categorical features: {cat_features}")

def get_oof_encoded_school(X_tr, y_tr, X_val, X_test):
    """Out-of-fold target encoding for School_Grouped with smoothing."""
    global_mean = y_tr.mean()
    smoothing = 10
    
    X_tr_enc = X_tr.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    
    counts = X_tr.groupby('School_Grouped').size()
    sums = X_tr_enc.copy()
    sums['target'] = y_tr
    sums = sums.groupby('School_Grouped')['target'].sum()
    
    encoded_map = (sums + smoothing * global_mean) / (counts + smoothing)
    
    X_tr_enc['School_Target_Enc'] = X_tr['School_Grouped'].map(encoded_map).fillna(global_mean)
    X_val_enc['School_Target_Enc'] = X_val['School_Grouped'].map(encoded_map).fillna(global_mean)
    X_test_enc['School_Target_Enc'] = X_test['School_Grouped'].map(encoded_map).fillna(global_mean)
    
    return X_tr_enc, X_val_enc, X_test_enc

# ── 3a. Tune XGBoost ──────────────────────────────────────────────────────────
print(f"\n[Step 3a] Tuning XGBoost ({OPTUNA_TRIALS} trials)...")
t0 = time.time()

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 8), # Lower max depth
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50), # Higher min_child_weight
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True), # Strong L1
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True), # Strong L2
        "gamma": trial.suggest_float("gamma", 0.1, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "enable_categorical": True,
        "tree_method": "hist"
    }
    
    scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr = X_train_df.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_val = X_train_df.iloc[val_idx]
        y_val = y[val_idx]
        
        # OOF Target Encoding
        X_tr, X_val, _ = get_oof_encoded_school(X_tr, y_tr, X_val, X_test_df)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        
        val_pred = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, val_pred))
        
    return np.mean(scores)

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
xgb_best = xgb_study.best_params
xgb_best.update({"random_state": RANDOM_STATE, "eval_metric": "auc",
                  "early_stopping_rounds": 50, "enable_categorical": True, "tree_method": "hist"})
print(f"  XGBoost best AUC: {xgb_study.best_value:.6f} ({time.time()-t0:.1f}s)")

# ── 3b. Tune LightGBM ─────────────────────────────────────────────────────────
print(f"\n[Step 3b] Tuning LightGBM ({OPTUNA_TRIALS} trials)...")
t0 = time.time()

def lgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

    scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr = X_train_df.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_val = X_train_df.iloc[val_idx]
        y_val = y[val_idx]
        
        X_tr, X_val, _ = get_oof_encoded_school(X_tr, y_tr, X_val, X_test_df)
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
        val_pred = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, val_pred))
    return np.mean(scores)

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
lgb_best = lgb_study.best_params
lgb_best.update({"random_state": RANDOM_STATE, "verbose": -1})
print(f"  LightGBM best AUC: {lgb_study.best_value:.6f} ({time.time()-t0:.1f}s)")

# ── 3c. Tune CatBoost ─────────────────────────────────────────────────────────
print(f"\n[Step 3c] Tuning CatBoost ({OPTUNA_TRIALS} trials)...")
t0 = time.time()

def cb_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "cat_features": cat_features
    }

    scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr = X_train_df.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_val = X_train_df.iloc[val_idx]
        y_val = y[val_idx]
        
        X_tr, X_val, _ = get_oof_encoded_school(X_tr, y_tr, X_val, X_test_df)
        
        # CatBoost needs strings/ints for categories
        for c in cat_features:
            X_tr[c] = X_tr[c].astype(str).fillna("missing")
            X_val[c] = X_val[c].astype(str).fillna("missing")
            
        model = cb.CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        val_pred = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, val_pred))
    return np.mean(scores)

cb_study = optuna.create_study(direction="maximize")
cb_study.optimize(cb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
cb_best = cb_study.best_params
cb_best.update({"random_seed": RANDOM_STATE, "verbose": 0,
                "eval_metric": "AUC", "early_stopping_rounds": 50, "cat_features": cat_features})
print(f"  CatBoost best AUC: {cb_study.best_value:.6f} ({time.time()-t0:.1f}s)")

# ── 4. Final Training with Best Params ────────────────────────────────────────
print("\n[Step 4] Final training and Rank Ensembling...")

xgb_oof = np.zeros(len(y)); xgb_test = np.zeros(len(X_test_df)); xgb_scores = []
lgb_oof = np.zeros(len(y)); lgb_test = np.zeros(len(X_test_df)); lgb_scores = []
cb_oof = np.zeros(len(y));  cb_test = np.zeros(len(X_test_df));  cb_scores = []

callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    X_tr = X_train_df.iloc[tr_idx]
    y_tr = y[tr_idx]
    X_val = X_train_df.iloc[val_idx]
    y_val = y[val_idx]
    
    X_tr, X_val, X_test_fold = get_oof_encoded_school(X_tr, y_tr, X_val, X_test_df)
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_best)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    xgb_oof[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    xgb_test += model_xgb.predict_proba(X_test_fold)[:, 1] / N_FOLDS
    xgb_scores.append(roc_auc_score(y_val, xgb_oof[val_idx]))
    
    # LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_best)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
    lgb_oof[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    lgb_test += model_lgb.predict_proba(X_test_fold)[:, 1] / N_FOLDS
    lgb_scores.append(roc_auc_score(y_val, lgb_oof[val_idx]))
    
    # CatBoost
    X_tr_cb = X_tr.copy()
    X_val_cb = X_val.copy()
    X_test_cb = X_test_fold.copy()
    for c in cat_features:
        X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("missing")
        X_val_cb[c] = X_val_cb[c].astype(str).fillna("missing")
        X_test_cb[c] = X_test_cb[c].astype(str).fillna("missing")
        
    model_cb = cb.CatBoostClassifier(**cb_best)
    model_cb.fit(X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), verbose=0)
    cb_oof[val_idx] = model_cb.predict_proba(X_val_cb)[:, 1]
    cb_test += model_cb.predict_proba(X_test_cb)[:, 1] / N_FOLDS
    cb_scores.append(roc_auc_score(y_val, cb_oof[val_idx]))

xgb_auc = np.mean(xgb_scores)
lgb_auc = np.mean(lgb_scores)
cb_auc = np.mean(cb_scores)

print(f"\n  Individual model 10-Fold AUCs:")
print(f"    XGBoost:  {xgb_auc:.6f}")
print(f"    LightGBM: {lgb_auc:.6f}")
print(f"    CatBoost: {cb_auc:.6f}")

# ── 5. Rank Averaging Ensemble ───────────────────────────────────────────────
print("\n[Step 5] Rank Averaging...")

# Rank the OOF predictions (normalize to 0-1)
xgb_oof_rank = rankdata(xgb_oof) / len(xgb_oof)
lgb_oof_rank = rankdata(lgb_oof) / len(lgb_oof)
cb_oof_rank = rankdata(cb_oof) / len(cb_oof)

# Blend ranks
oof_rank_ens = (xgb_oof_rank + lgb_oof_rank + cb_oof_rank) / 3
rank_ens_auc = roc_auc_score(y, oof_rank_ens)

print(f"  Rank Ensemble OOF AUC: {rank_ens_auc:.6f}")

# Compare with simple probability average
oof_avg = (xgb_oof + lgb_oof + cb_oof) / 3
avg_auc = roc_auc_score(y, oof_avg)
print(f"  Simple Average OOF AUC: {avg_auc:.6f}")

# Rank the Test predictions
xgb_test_rank = rankdata(xgb_test) / len(xgb_test)
lgb_test_rank = rankdata(lgb_test) / len(lgb_test)
cb_test_rank = rankdata(cb_test) / len(cb_test)

test_rank_ens = (xgb_test_rank + lgb_test_rank + cb_test_rank) / 3
test_avg = (xgb_test + lgb_test + cb_test) / 3

individual_aucs = {"xgb": (xgb_auc, xgb_test), "lgb": (lgb_auc, lgb_test), "cb": (cb_auc, cb_test)}
best_ind_name = max(individual_aucs, key=lambda k: individual_aucs[k][0])
best_ind_auc, best_ind_test = individual_aucs[best_ind_name]
print(f"  Best individual model: {best_ind_name} with AUC {best_ind_auc:.6f}")

candidates = {
    "rank_ensemble": (rank_ens_auc, test_rank_ens),
    "simple_average": (avg_auc, test_avg),
    f"best_individual_{best_ind_name}": (best_ind_auc, best_ind_test),
}
best_method = max(candidates, key=lambda k: candidates[k][0])
final_auc, final_test_preds = candidates[best_method]

print(f"  >> Using {best_method} for final submission (AUC={final_auc:.6f})")

# ── 6. Generate Submission ────────────────────────────────────────────────────
print(f"\n[Step 6] Generating {OUTPUT_FILE}...")
submission = pd.DataFrame({"Id": test_ids, "Drafted": final_test_preds})
submission.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved {OUTPUT_FILE} with {len(submission)} rows")

# Also save the rank ensemble specifically as a fallback
sub_rank = pd.DataFrame({"Id": test_ids, "Drafted": test_rank_ens})
sub_rank.to_csv("submission_v2_rank.csv", index=False)

print(f"\n{'=' * 70}")
print(f"FINAL V2 CV AUC: {final_auc:.6f}")
print(f"{'=' * 70}")
print("Done! Submit submission_v2.csv to Omnicampus.")
