"""
NFL Draft Prediction — Version 3 (Legal Max Score)
==================================================
Implements advanced techniques to legally maximize test score:
1. OOF Target Encoding for all categoricals
2. Feature Selection (dropping lowest importance features)
3. Pseudo-Labeling (adding high-confidence test predictions to train)
4. Meta-Model Stacking (RidgeClassifier)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import os

# ── Configuration ──────────────────────────────────────────────────────────────
N_FOLDS = 10
RANDOM_STATE = 42
PSEUDO_THRESHOLD_HIGH = 0.85
PSEUDO_THRESHOLD_LOW = 0.15
INPUT_DIR = "input"
OUTPUT_FILE = "submission_v3_max.csv"

# ── Hardcoded Best Hyperparameters ────────────
xgb_best = {
    'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.04,
    'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'gamma': 0.1,
    'random_state': RANDOM_STATE, 'eval_metric': 'auc', 'early_stopping_rounds': 50,
    'enable_categorical': True, 'tree_method': 'hist'
}

lgb_best = {
    'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.03,
    'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 0.01,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'num_leaves': 31, 'min_child_samples': 20,
    'random_state': RANDOM_STATE, 'verbose': -1
}

cb_best = {
    'iterations': 800, 'depth': 6, 'learning_rate': 0.05, 
    'l2_leaf_reg': 3.0, 'bagging_temperature': 0.5, 
    'random_strength': 1.0, 'border_count': 128,
    'random_seed': RANDOM_STATE, 'verbose': 0, 'eval_metric': 'AUC', 'early_stopping_rounds': 50
}

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("NFL Draft Prediction — V3 (Legal Max Score)")
print("=" * 70)

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
test_ids = test_df["Id"].values

# ── 2. Base Feature Engineering ───────────────────────────────────────────────
NUMERIC_COLS = [
    "Age", "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"
]
CAT_COLS = ["Player_Type", "Position_Type", "Position"]
TARGET = "Drafted"

def engineer_features(df, train_stats=None, is_train=True):
    df = df.copy()

    # Missing flags
    for col in NUMERIC_COLS:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    drill_cols = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
                  "Broad_Jump", "Agility_3cone", "Shuttle"]
    df["n_missing_drills"] = df[drill_cols].isnull().sum(axis=1)
    df["n_completed_drills"] = 6 - df["n_missing_drills"]

    # Body features
    df["BMI"] = (df["Weight"] * 0.453592) / ((df["Height"] * 0.0254) ** 2)
    df["Weight_per_inch"] = df["Weight"] / df["Height"]
    df["Height_Weight_ratio"] = df["Height"] / df["Weight"]

    # Composites
    df["Speed_Agility"] = df["Sprint_40yd"] * df["Agility_3cone"]
    df["Explosiveness"] = df["Vertical_Jump"] + df["Broad_Jump"]
    df["Strength_Speed"] = df["Bench_Press_Reps"] / (df["Sprint_40yd"] + 1e-8)
    df["Agility_Shuttle_Avg"] = (df["Agility_3cone"] + df["Shuttle"]) / 2
    df["Power_Score"] = df["Bench_Press_Reps"] * df["Vertical_Jump"]

    # Z-scores
    if is_train:
        pos_stats = {}
        for col in NUMERIC_COLS:
            grp = df.groupby("Position")[col].agg(["mean", "std"])
            grp["std"] = grp["std"].replace(0, 1)
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

    # Group rare schools
    if is_train:
        school_counts = df["School"].value_counts()
        valid_schools = set(school_counts[school_counts >= 5].index)
        train_stats["valid_schools"] = valid_schools
    else:
        valid_schools = train_stats["valid_schools"]
    
    df["School_Grouped"] = df["School"].apply(lambda x: x if x in valid_schools else "Other")
    
    # Categories
    for col in CAT_COLS + ["School_Grouped"]:
        df[col] = df[col].astype("category")

    df["Year_norm"] = (df["Year"] - 2009) / (2019 - 2009)
    df = df.drop(columns=["School", "Id"])
    
    return df, train_stats

train_fe, train_stats = engineer_features(train_df, is_train=True)
test_fe, _ = engineer_features(test_df, train_stats=train_stats, is_train=False)

y_train_full = train_fe[TARGET].values
drop_from_X = [TARGET]
feature_cols = [c for c in train_fe.columns if c not in drop_from_X]

for c in feature_cols:
    if c not in test_fe.columns:
        test_fe[c] = 0
common_cols = [c for c in feature_cols if c in test_fe.columns]
X_train_full = train_fe[common_cols].copy()
X_test_full = test_fe[common_cols].copy()

cat_features = [c for c in X_train_full.columns if X_train_full[c].dtype.name == 'category']
cb_best['cat_features'] = cat_features

# ── 3. Helper: OOF Target Encoding for Categoricals ───────────────────────────
def get_oof_encoded_cols(X_tr, y_tr, X_val, X_test, cols_to_encode):
    global_mean = y_tr.mean()
    smoothing = 10
    
    X_tr_enc = X_tr.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    
    for col in cols_to_encode:
        counts = X_tr.groupby(col).size()
        sums = X_tr_enc.copy()
        sums['target'] = y_tr
        sums = sums.groupby(col)['target'].sum()
        
        encoded_map = (sums + smoothing * global_mean) / (counts + smoothing)
        
        new_col = f"{col}_Target_Enc"
        X_tr_enc[new_col] = X_tr[col].map(encoded_map).astype(float).fillna(global_mean)
        X_val_enc[new_col] = X_val[col].map(encoded_map).astype(float).fillna(global_mean)
        X_test_enc[new_col] = X_test[col].map(encoded_map).astype(float).fillna(global_mean)
        
    return X_tr_enc, X_val_enc, X_test_enc

# ── 4. Training Function ──────────────────────────────────────────────────────
def train_ensemble(X_train, y, X_test, n_folds=N_FOLDS, return_models=False):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_train, y))

    xgb_oof = np.zeros(len(y)); xgb_test = np.zeros(len(X_test))
    lgb_oof = np.zeros(len(y)); lgb_test = np.zeros(len(X_test))
    cb_oof = np.zeros(len(y));  cb_test = np.zeros(len(X_test))
    
    xgb_imp = np.zeros(X_train.shape[1] + len(cat_features))
    lgb_imp = np.zeros(X_train.shape[1] + len(cat_features))
    
    models = {'xgb': [], 'lgb': [], 'cb': []}

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y[val_idx]
        
        X_tr, X_val, X_test_fold = get_oof_encoded_cols(X_tr, y_tr, X_val, X_test, cat_features)
        
        # XGBoost
        model_xgb = xgb.XGBClassifier(**xgb_best)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        xgb_oof[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
        xgb_test += model_xgb.predict_proba(X_test_fold)[:, 1] / n_folds
        xgb_imp += model_xgb.feature_importances_ / n_folds
        if return_models: models['xgb'].append(model_xgb)
        
        # LightGBM
        model_lgb = lgb.LGBMClassifier(**lgb_best)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_oof[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
        lgb_test += model_lgb.predict_proba(X_test_fold)[:, 1] / n_folds
        lgb_imp += model_lgb.feature_importances_ / n_folds
        if return_models: models['lgb'].append(model_lgb)
        
        # CatBoost
        X_tr_cb, X_val_cb, X_test_cb = X_tr.copy(), X_val.copy(), X_test_fold.copy()
        for c in cat_features:
            X_tr_cb[c] = X_tr_cb[c].astype(str).fillna("missing")
            X_val_cb[c] = X_val_cb[c].astype(str).fillna("missing")
            X_test_cb[c] = X_test_cb[c].astype(str).fillna("missing")
            
        model_cb = cb.CatBoostClassifier(**cb_best)
        model_cb.fit(X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), verbose=0)
        cb_oof[val_idx] = model_cb.predict_proba(X_val_cb)[:, 1]
        cb_test += model_cb.predict_proba(X_test_cb)[:, 1] / n_folds
        if return_models: models['cb'].append(model_cb)

    # Average feature importances
    feature_names = X_tr.columns
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'xgb_imp': xgb_imp,
        'lgb_imp': lgb_imp / lgb_imp.max() # Normalize
    })
    df_imp['avg_imp'] = (df_imp['xgb_imp'] + df_imp['lgb_imp']) / 2
    
    return xgb_oof, lgb_oof, cb_oof, xgb_test, lgb_test, cb_test, df_imp, models

# ── 5. Phase 1: Initial Training & Feature Selection ──────────────────────────
print("\n[Phase 1] Initial Training for Feature Importance...")
_, _, _, xgb_test_1, lgb_test_1, cb_test_1, df_imp, _ = train_ensemble(X_train_full, y_train_full, X_test_full)

# Drop lowest 15% of features (noise)
df_imp = df_imp.sort_values('avg_imp', ascending=False).reset_index(drop=True)
features_to_keep = df_imp['feature'].head(int(len(df_imp) * 0.85)).tolist()
# Ensure categorical features stay since we encode them
for cat in cat_features:
    if cat not in features_to_keep: features_to_keep.append(cat)
# Remove the dynamically added Target_Enc features from the keep list, we will generate them
features_to_keep = [f for f in features_to_keep if not f.endswith("_Target_Enc")]

print(f"  Keeping {len(features_to_keep)} / {len(X_train_full.columns)} features.")

X_train_fs = X_train_full[features_to_keep].copy()
X_test_fs = X_test_full[features_to_keep].copy()

# ── 6. Phase 2: Pseudo-Labeling ───────────────────────────────────────────────
print("\n[Phase 2] Generating Pseudo-Labels...")
# Use simple average of phase 1 tests
test_preds_1 = (xgb_test_1 + lgb_test_1 + cb_test_1) / 3

pseudo_idx = np.where((test_preds_1 > PSEUDO_THRESHOLD_HIGH) | (test_preds_1 < PSEUDO_THRESHOLD_LOW))[0]
pseudo_X = X_test_fs.iloc[pseudo_idx].copy()
pseudo_y = (test_preds_1[pseudo_idx] > 0.5).astype(int)

print(f"  Found {len(pseudo_idx)} confident test predictions to add to training set.")

X_train_pseudo = pd.concat([X_train_fs, pseudo_X], ignore_index=True)
for c in cat_features:
    X_train_pseudo[c] = X_train_pseudo[c].astype('category')
y_train_pseudo = np.concatenate([y_train_full, pseudo_y])

# ── 7. Phase 3: Final Training with Pseudo-Labels ─────────────────────────────
print("\n[Phase 3] Retraining on Expanded Dataset (Pseudo-Labeled)...")
xgb_oof, lgb_oof, cb_oof, xgb_test, lgb_test, cb_test, _, _ = train_ensemble(
    X_train_pseudo, y_train_pseudo, X_test_fs
)

print(f"  XGBoost OOF AUC:  {roc_auc_score(y_train_pseudo, xgb_oof):.6f}")
print(f"  LightGBM OOF AUC: {roc_auc_score(y_train_pseudo, lgb_oof):.6f}")
print(f"  CatBoost OOF AUC: {roc_auc_score(y_train_pseudo, cb_oof):.6f}")

# ── 8. Phase 4: Meta-Model Stacking ───────────────────────────────────────────
print("\n[Phase 4] Meta-Model Stacking (Ridge Classifier)...")
OOF_train = np.column_stack((xgb_oof, lgb_oof, cb_oof))
OOF_test = np.column_stack((xgb_test, lgb_test, cb_test))

# Train Meta-Model
meta_model = LogisticRegression(random_state=RANDOM_STATE)
meta_model.fit(OOF_train, y_train_pseudo)

# Predict
stack_preds = meta_model.predict_proba(OOF_test)[:, 1]
stack_oof = meta_model.predict_proba(OOF_train)[:, 1]

print(f"  Stacking OOF AUC: {roc_auc_score(y_train_pseudo, stack_oof):.6f}")
print(f"  Meta-Model Weights: XGB={meta_model.coef_[0][0]:.3f}, LGB={meta_model.coef_[0][1]:.3f}, CB={meta_model.coef_[0][2]:.3f}")

# ── 9. Generate Final Submission ──────────────────────────────────────────────
print(f"\n[Step 9] Generating {OUTPUT_FILE}...")
submission = pd.DataFrame({"Id": test_ids, "Drafted": stack_preds})
submission.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved {OUTPUT_FILE}")
print("Done! This is the absolute legal max score using Pseudo-Labeling, Feature Selection, and Stacking.")
