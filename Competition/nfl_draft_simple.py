"""
NFL Draft Prediction — Simple Version (Single XGBoost)
======================================================
A simpler, faster version using just XGBoost with good features.
Runs in ~2-3 minutes instead of 30-60+ min.

Usage:
    python nfl_draft_simple.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os

# ── Configuration ──────────────────────────────────────────────────────────────
N_FOLDS = 5
RANDOM_STATE = 42
INPUT_DIR = "input"
OUTPUT_FILE = "submission_simple.csv"

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("NFL Draft Prediction — Simple XGBoost Version")
print("=" * 60)

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

print(f"\nTrain: {train_df.shape}, Test: {test_df.shape}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
NUMERIC_COLS = [
    "Age", "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"
]
TARGET = "Drafted"

def engineer_features(df, train_stats=None, school_stats=None, is_train=True):
    df = df.copy()

    # Missing flags
    for col in NUMERIC_COLS:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    drill_cols = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
                  "Broad_Jump", "Agility_3cone", "Shuttle"]
    df["n_missing_drills"] = df[drill_cols].isnull().sum(axis=1)

    # Position-aware imputation
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

    # Body features
    df["BMI"] = (df["Weight"] * 0.453592) / ((df["Height"] * 0.0254) ** 2)
    df["Weight_per_inch"] = df["Weight"] / df["Height"]

    # Performance composites
    df["Explosiveness"] = df["Vertical_Jump"] + df["Broad_Jump"]
    eps = 1e-8
    df["Strength_Speed"] = df["Bench_Press_Reps"] / (df["Sprint_40yd"] + eps)

    # Position z-scores
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

    # School target encoding
    gm = school_stats["global_mean"]
    sm = school_stats["smoothing"]
    sd = school_stats["school_data"]
    df["School_encoded"] = df["School"].apply(
        lambda s: (sd[s]["sum"] + sm * gm) / (sd[s]["count"] + sm) if s in sd else gm
    )

    # Categorical encoding
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
test_ids = test_fe["Id"].values

drop_cols = ["Id", TARGET]
feat_cols = [c for c in train_fe.columns if c not in drop_cols]
for c in feat_cols:
    if c not in test_fe.columns:
        test_fe[c] = 0
common = [c for c in feat_cols if c in test_fe.columns]
X_train = train_fe[common].values
X_test = test_fe[common].values

print(f"Features: {len(common)}")

# ── 3. Train XGBoost with good defaults ───────────────────────────────────────
print("\n[Training XGBoost 5-fold CV...]")

params = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "random_state": RANDOM_STATE,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "use_label_encoder": False,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    val_pred = model.predict_proba(X_val)[:, 1]
    oof[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    auc = roc_auc_score(y_val, val_pred)
    print(f"  Fold {fold+1}: AUC = {auc:.6f}")

overall_auc = roc_auc_score(y, oof)
print(f"\n  Overall OOF AUC: {overall_auc:.6f}")

# ── 4. Save Submission ────────────────────────────────────────────────────────
submission = pd.DataFrame({"Id": test_ids, "Drafted": test_preds})
submission.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {OUTPUT_FILE} ({len(submission)} rows)")
print(f"Predictions: min={test_preds.min():.4f}, max={test_preds.max():.4f}, mean={test_preds.mean():.4f}")
print("Done!")
