"""
NFL Draft Prediction — Full Ensemble Solution
===============================================
Models: XGBoost + LightGBM + CatBoost
Tuning: Optuna (200 trials per model)
Features: Advanced engineering (position z-scores, BMI, school target encoding, etc.)
Evaluation: ROC AUC with Stratified 5-Fold CV

Usage:
    python nfl_draft_model.py
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
import optuna
import os
import time

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Configuration ──────────────────────────────────────────────────────────────
N_FOLDS = 5
RANDOM_STATE = 42
OPTUNA_TRIALS = 200  # 200 trials per model for best results
INPUT_DIR = "input"
OUTPUT_FILE = "submission.csv"

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("NFL Draft Prediction — Full Ensemble Solution")
print("=" * 70)

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"Target distribution:\n{train_df['Drafted'].value_counts().to_dict()}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("\n[Step 2] Feature Engineering...")

NUMERIC_COLS = [
    "Age", "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"
]
CAT_COLS = ["Player_Type", "Position_Type", "Position", "School"]
TARGET = "Drafted"

def engineer_features(df, train_stats=None, school_stats=None, is_train=True):
    """Apply all feature engineering steps."""
    df = df.copy()

    # ── 2a. Missing value indicator flags ──
    for col in NUMERIC_COLS:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    # Count total missing drills per player (informative!)
    drill_cols = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
                  "Broad_Jump", "Agility_3cone", "Shuttle"]
    df["n_missing_drills"] = df[drill_cols].isnull().sum(axis=1)
    df["n_completed_drills"] = 6 - df["n_missing_drills"]

    # ── 2b. Position-aware imputation ──
    if is_train:
        # Compute median per Position for each numeric column
        train_stats = {}
        for col in NUMERIC_COLS:
            train_stats[col] = {
                "global_median": df[col].median(),
                "position_medians": df.groupby("Position")[col].median().to_dict()
            }
        # School target encoding stats
        school_counts = df.groupby("School")[TARGET].agg(["sum", "count"])
        global_mean = df[TARGET].mean()
        smoothing = 10  # regularization parameter
        school_stats = {
            "global_mean": global_mean,
            "smoothing": smoothing,
            "school_data": school_counts.to_dict("index")
        }

    for col in NUMERIC_COLS:
        for pos in df["Position"].unique():
            mask = (df["Position"] == pos) & df[col].isnull()
            pos_median = train_stats[col]["position_medians"].get(
                pos, train_stats[col]["global_median"]
            )
            if pd.isna(pos_median):
                pos_median = train_stats[col]["global_median"]
            df.loc[mask, col] = pos_median
        # Fill any remaining NaN with global median
        df[col] = df[col].fillna(train_stats[col]["global_median"])

    # ── 2c. Body composition features ──
    df["BMI"] = (df["Weight"] * 0.453592) / ((df["Height"] * 0.0254) ** 2)
    df["Weight_per_inch"] = df["Weight"] / df["Height"]
    df["Height_Weight_ratio"] = df["Height"] / df["Weight"]

    # ── 2d. Athletic performance composites ──
    df["Speed_Agility"] = df["Sprint_40yd"] * df["Agility_3cone"]
    df["Explosiveness"] = df["Vertical_Jump"] + df["Broad_Jump"]
    eps = 1e-8
    df["Strength_Speed"] = df["Bench_Press_Reps"] / (df["Sprint_40yd"] + eps)
    df["Agility_Shuttle_Avg"] = (df["Agility_3cone"] + df["Shuttle"]) / 2
    df["Power_Score"] = df["Bench_Press_Reps"] * df["Vertical_Jump"]

    # ── 2e. Position-relative z-scores ──
    if is_train:
        # Compute mean/std per position from training data
        pos_stats = {}
        for col in NUMERIC_COLS:
            grp = df.groupby("Position")[col].agg(["mean", "std"])
            grp["std"] = grp["std"].replace(0, 1)  # avoid div by zero
            pos_stats[col] = grp.to_dict("index")
        train_stats["pos_zscores"] = pos_stats

    for col in NUMERIC_COLS:
        zscore_col = f"{col}_pos_zscore"
        df[zscore_col] = 0.0
        for pos in df["Position"].unique():
            mask = df["Position"] == pos
            stats = train_stats["pos_zscores"][col].get(pos)
            if stats and stats["std"] and stats["std"] > 0:
                df.loc[mask, zscore_col] = (
                    (df.loc[mask, col] - stats["mean"]) / stats["std"]
                )

    # ── 2f. School target encoding ──
    global_mean = school_stats["global_mean"]
    smoothing = school_stats["smoothing"]
    school_data = school_stats["school_data"]

    def encode_school(school):
        if school in school_data:
            s = school_data[school]
            n = s["count"]
            return (s["sum"] + smoothing * global_mean) / (n + smoothing)
        return global_mean

    df["School_encoded"] = df["School"].apply(encode_school)

    # ── 2g. Encode categoricals ──
    # One-hot encode low-cardinality categoricals
    df = pd.get_dummies(df, columns=["Player_Type", "Position_Type"], drop_first=False)

    # Label encode Position (for tree models)
    if is_train:
        le_pos = LabelEncoder()
        df["Position_le"] = le_pos.fit_transform(df["Position"])
        train_stats["le_pos"] = le_pos
    else:
        le_pos = train_stats["le_pos"]
        df["Position_le"] = df["Position"].apply(
            lambda x: le_pos.transform([x])[0] if x in le_pos.classes_ else -1
        )

    # ── 2h. Year-based features ──
    df["Year_norm"] = (df["Year"] - 2009) / (2019 - 2009)

    # Drop original string columns and Id
    drop_cols = ["School", "Position"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df, train_stats, school_stats


# Apply feature engineering
train_fe, train_stats, school_stats = engineer_features(train_df, is_train=True)
test_fe, _, _ = engineer_features(test_df, train_stats=train_stats,
                                   school_stats=school_stats, is_train=False)

# Prepare X, y
y = train_fe[TARGET].values
train_ids = train_fe["Id"].values
test_ids = test_fe["Id"].values

drop_from_X = ["Id", TARGET]
feature_cols = [c for c in train_fe.columns if c not in drop_from_X]
# Align columns
for c in feature_cols:
    if c not in test_fe.columns:
        test_fe[c] = 0
test_fe = test_fe[[c for c in feature_cols if c in test_fe.columns]]
# Re-align
common_cols = [c for c in feature_cols if c in test_fe.columns]
X_train = train_fe[common_cols].values
X_test = test_fe[common_cols].values

print(f"  Features: {len(common_cols)}")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape:  {X_test.shape}")

# ── 3. Optuna Tuning + Training ───────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds = list(skf.split(X_train, y))


def cv_score(model_fn):
    """Run 5-fold CV and return mean AUC + OOF predictions."""
    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        scores.append(roc_auc_score(y_val, val_pred))

    return np.mean(scores), oof_preds, test_preds


# ── 3a. Tune XGBoost ──────────────────────────────────────────────────────────
print(f"\n[Step 3a] Tuning XGBoost ({OPTUNA_TRIALS} trials)...")
t0 = time.time()

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "use_label_encoder": False,
    }
    def model_fn():
        return xgb.XGBClassifier(**params)
    score, _, _ = cv_score(model_fn)
    return score

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
xgb_best = xgb_study.best_params
xgb_best.update({"random_state": RANDOM_STATE, "eval_metric": "auc",
                  "early_stopping_rounds": 50, "use_label_encoder": False})
print(f"  XGBoost best AUC: {xgb_study.best_value:.6f} ({time.time()-t0:.1f}s)")

# ── 3b. Tune LightGBM ─────────────────────────────────────────────────────────
print(f"\n[Step 3b] Tuning LightGBM ({OPTUNA_TRIALS} trials)...")
t0 = time.time()

def lgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

    oof_preds = np.zeros(len(y))
    scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
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
        "iterations": trial.suggest_int("iterations", 100, 1500),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
    }

    oof_preds = np.zeros(len(y))
    scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = cb.CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        scores.append(roc_auc_score(y_val, val_pred))
    return np.mean(scores)

cb_study = optuna.create_study(direction="maximize")
cb_study.optimize(cb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
cb_best = cb_study.best_params
cb_best.update({"random_seed": RANDOM_STATE, "verbose": 0,
                "eval_metric": "AUC", "early_stopping_rounds": 50})
print(f"  CatBoost best AUC: {cb_study.best_value:.6f} ({time.time()-t0:.1f}s)")

# ── 4. Final Training with Best Params ────────────────────────────────────────
print("\n[Step 4] Final training with best hyperparameters...")

# XGBoost final
def xgb_model_fn():
    return xgb.XGBClassifier(**xgb_best)
xgb_auc, xgb_oof, xgb_test = cv_score(xgb_model_fn)

# LightGBM final
lgb_oof = np.zeros(len(y))
lgb_test = np.zeros(len(X_test))
lgb_scores = []
callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    model = lgb.LGBMClassifier(**lgb_best)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
    val_pred = model.predict_proba(X_val)[:, 1]
    lgb_oof[val_idx] = val_pred
    lgb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS
    lgb_scores.append(roc_auc_score(y_val, val_pred))
lgb_auc = np.mean(lgb_scores)

# CatBoost final
cb_oof = np.zeros(len(y))
cb_test = np.zeros(len(X_test))
cb_scores = []
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    model = cb.CatBoostClassifier(**cb_best)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    val_pred = model.predict_proba(X_val)[:, 1]
    cb_oof[val_idx] = val_pred
    cb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS
    cb_scores.append(roc_auc_score(y_val, val_pred))
cb_auc = np.mean(cb_scores)

print(f"\n  Individual model AUCs:")
print(f"    XGBoost:  {xgb_auc:.6f}")
print(f"    LightGBM: {lgb_auc:.6f}")
print(f"    CatBoost: {cb_auc:.6f}")

# ── 5. Ensemble ───────────────────────────────────────────────────────────────
print("\n[Step 5] Ensemble blending...")

# Weight by relative AUC performance
aucs = np.array([xgb_auc, lgb_auc, cb_auc])
weights = aucs / aucs.sum()
print(f"  Weights: XGB={weights[0]:.4f}, LGB={weights[1]:.4f}, CB={weights[2]:.4f}")

oof_ensemble = weights[0] * xgb_oof + weights[1] * lgb_oof + weights[2] * cb_oof
test_ensemble = weights[0] * xgb_test + weights[1] * lgb_test + weights[2] * cb_test

ensemble_auc = roc_auc_score(y, oof_ensemble)
print(f"  Ensemble OOF AUC: {ensemble_auc:.6f}")

# Also try simple average
oof_avg = (xgb_oof + lgb_oof + cb_oof) / 3
test_avg = (xgb_test + lgb_test + cb_test) / 3
avg_auc = roc_auc_score(y, oof_avg)
print(f"  Simple Average OOF AUC: {avg_auc:.6f}")

# Also check best individual model
individual_aucs = {"xgb": (xgb_auc, xgb_test), "lgb": (lgb_auc, lgb_test), "cb": (cb_auc, cb_test)}
best_ind_name = max(individual_aucs, key=lambda k: individual_aucs[k][0])
best_ind_auc, best_ind_test = individual_aucs[best_ind_name]
print(f"  Best individual model: {best_ind_name} with AUC {best_ind_auc:.6f}")

# Use whichever is best among: weighted ensemble, simple avg, best individual
candidates = {
    "weighted_ensemble": (ensemble_auc, test_ensemble),
    "simple_average": (avg_auc, test_avg),
    f"best_individual_{best_ind_name}": (best_ind_auc, best_ind_test),
}
best_method = max(candidates, key=lambda k: candidates[k][0])
final_auc, final_test_preds = candidates[best_method]
print(f"  >> Using {best_method} (AUC={final_auc:.6f})")

# ── 6. Generate Submission ────────────────────────────────────────────────────
print(f"\n[Step 6] Generating {OUTPUT_FILE}...")
submission = pd.DataFrame({"Id": test_ids, "Drafted": final_test_preds})
submission.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved {OUTPUT_FILE} with {len(submission)} rows")
print(f"  Prediction stats: min={final_test_preds.min():.4f}, "
      f"max={final_test_preds.max():.4f}, mean={final_test_preds.mean():.4f}")

# Also save ensemble version separately
submission_ens = pd.DataFrame({"Id": test_ids, "Drafted": test_ensemble})
submission_ens.to_csv("submission_ensemble.csv", index=False)
print(f"  Also saved submission_ensemble.csv")

print(f"\n{'=' * 70}")
print(f"FINAL CV AUC: {final_auc:.6f}")
print(f"{'=' * 70}")
print("Done! Submit the file to Omnicampus.")
