"""
NFL Draft Prediction — Best Model Quick Run
===========================================
Uses the best hyperparameters found by Optuna during the long training run
to quickly generate the submission.

Best individual model: CatBoost
Ensemble: CatBoost + XGBoost + LightGBM
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import os

N_FOLDS = 5
RANDOM_STATE = 42
INPUT_DIR = "input"
OUTPUT_FILE = "submission.csv"

# ── 1. Load Data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
test_ids = test_df["Id"].values

# ── 2. Feature Engineering ────────────────────────────────────────────────────
NUMERIC_COLS = [
    "Age", "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"
]
TARGET = "Drafted"

def engineer_features(df, train_stats=None, school_stats=None, is_train=True):
    df = df.copy()

    for col in NUMERIC_COLS:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    drill_cols = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
                  "Broad_Jump", "Agility_3cone", "Shuttle"]
    df["n_missing_drills"] = df[drill_cols].isnull().sum(axis=1)
    df["n_completed_drills"] = 6 - df["n_missing_drills"]

    if is_train:
        train_stats = {}
        for col in NUMERIC_COLS:
            train_stats[col] = {
                "global_median": df[col].median(),
                "position_medians": df.groupby("Position")[col].median().to_dict()
            }
        school_counts = df.groupby("School")[TARGET].agg(["sum", "count"])
        global_mean = df[TARGET].mean()
        school_stats = {
            "global_mean": global_mean, "smoothing": 10,
            "school_data": school_counts.to_dict("index")
        }

    for col in NUMERIC_COLS:
        for pos in df["Position"].unique():
            mask = (df["Position"] == pos) & df[col].isnull()
            med = train_stats[col]["position_medians"].get(pos, train_stats[col]["global_median"])
            if pd.isna(med):
                med = train_stats[col]["global_median"]
            df.loc[mask, col] = med
        df[col] = df[col].fillna(train_stats[col]["global_median"])

    df["BMI"] = (df["Weight"] * 0.453592) / ((df["Height"] * 0.0254) ** 2)
    df["Weight_per_inch"] = df["Weight"] / df["Height"]
    df["Height_Weight_ratio"] = df["Height"] / df["Weight"]

    df["Speed_Agility"] = df["Sprint_40yd"] * df["Agility_3cone"]
    df["Explosiveness"] = df["Vertical_Jump"] + df["Broad_Jump"]
    df["Strength_Speed"] = df["Bench_Press_Reps"] / (df["Sprint_40yd"] + 1e-8)
    df["Agility_Shuttle_Avg"] = (df["Agility_3cone"] + df["Shuttle"]) / 2
    df["Power_Score"] = df["Bench_Press_Reps"] * df["Vertical_Jump"]

    if is_train:
        pos_stats = {}
        for col in NUMERIC_COLS:
            grp = df.groupby("Position")[col].agg(["mean", "std"])
            grp["std"] = grp["std"].replace(0, 1)
            pos_stats[col] = grp.to_dict("index")
        train_stats["pos_zscores"] = pos_stats

    for col in NUMERIC_COLS:
        df[f"{col}_pos_zscore"] = 0.0
        for pos in df["Position"].unique():
            mask = df["Position"] == pos
            stats = train_stats["pos_zscores"][col].get(pos)
            if stats and stats["std"] and stats["std"] > 0:
                df.loc[mask, f"{col}_pos_zscore"] = (
                    (df.loc[mask, col] - stats["mean"]) / stats["std"]
                )

    gm = school_stats["global_mean"]
    sm = school_stats["smoothing"]
    sd = school_stats["school_data"]
    df["School_encoded"] = df["School"].apply(
        lambda s: (sd[s]["sum"] + sm * gm) / (sd[s]["count"] + sm) if s in sd else gm
    )

    df = pd.get_dummies(df, columns=["Player_Type", "Position_Type"], drop_first=False)
    if is_train:
        le = LabelEncoder()
        df["Position_le"] = le.fit_transform(df["Position"])
        train_stats["le_pos"] = le
    else:
        le = train_stats["le_pos"]
        df["Position_le"] = df["Position"].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    df["Year_norm"] = (df["Year"] - 2009) / (2019 - 2009)
    df = df.drop(columns=["School", "Position"], errors="ignore")

    return df, train_stats, school_stats

train_fe, ts, ss = engineer_features(train_df, is_train=True)
test_fe, _, _ = engineer_features(test_df, train_stats=ts, school_stats=ss, is_train=False)

y = train_fe[TARGET].values
drop_cols = ["Id", TARGET]
feat_cols = [c for c in train_fe.columns if c not in drop_cols]
for c in feat_cols:
    if c not in test_fe.columns:
        test_fe[c] = 0
common = [c for c in feat_cols if c in test_fe.columns]
X_train = train_fe[common].values
X_test = test_fe[common].values

print(f"Features: {len(common)}")

# ── 3. Train with best params ─────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds = list(skf.split(X_train, y))

def cv_score(model_fn):
    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = model_fn()
        # LightGBM needs callbacks
        if "LGBM" in str(type(model)):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        # CatBoost needs tuple eval_set
        elif "CatBoost" in str(type(model)):
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
            
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        scores.append(roc_auc_score(y_val, val_pred))
    return np.mean(scores), oof_preds, test_preds

# Hardcoded best params from tuning run
cb_best = {
    'iterations': 1200, 'depth': 6, 'learning_rate': 0.05, 
    'l2_leaf_reg': 3.0, 'bagging_temperature': 0.5, 
    'random_strength': 1.0, 'border_count': 128,
    'random_seed': RANDOM_STATE, 'verbose': 0, 'eval_metric': 'AUC', 'early_stopping_rounds': 50
}

xgb_best = {
    'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.04,
    'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'gamma': 0.1,
    'random_state': RANDOM_STATE, 'eval_metric': 'auc', 'early_stopping_rounds': 50, 'use_label_encoder': False
}

lgb_best = {
    'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.03,
    'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 0.01,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'num_leaves': 31, 'min_child_samples': 20,
    'random_state': RANDOM_STATE, 'verbose': -1
}

print("Training models...")
cb_auc, cb_oof, cb_test = cv_score(lambda: cb.CatBoostClassifier(**cb_best))
xgb_auc, xgb_oof, xgb_test = cv_score(lambda: xgb.XGBClassifier(**xgb_best))
lgb_auc, lgb_oof, lgb_test = cv_score(lambda: lgb.LGBMClassifier(**lgb_best))

print(f"XGBoost AUC:  {xgb_auc:.6f}")
print(f"LightGBM AUC: {lgb_auc:.6f}")
print(f"CatBoost AUC: {cb_auc:.6f}")

aucs = np.array([xgb_auc, lgb_auc, cb_auc])
weights = aucs / aucs.sum()
oof_ens = weights[0]*xgb_oof + weights[1]*lgb_oof + weights[2]*cb_oof
test_ens = weights[0]*xgb_test + weights[1]*lgb_test + weights[2]*cb_test
ens_auc = roc_auc_score(y, oof_ens)

print(f"Ensemble AUC: {ens_auc:.6f}")

# CatBoost was the best individual model in tuning
print(">> Saving CatBoost predictions (best individual model)")
submission_cb = pd.DataFrame({"Id": test_ids, "Drafted": cb_test})
submission_cb.to_csv("submission_catboost.csv", index=False)

print(f">> Saving Final predictions to {OUTPUT_FILE}")
if cb_auc > ens_auc:
    print("Using CatBoost (better than ensemble)")
    final_test_preds = cb_test
else:
    print("Using Ensemble (better than CatBoost)")
    final_test_preds = test_ens
    
submission = pd.DataFrame({"Id": test_ids, "Drafted": final_test_preds})
submission.to_csv(OUTPUT_FILE, index=False)
print("Done!")
