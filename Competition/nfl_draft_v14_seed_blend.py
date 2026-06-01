"""
NFL Draft Prediction - V14 (Seed Blending)
==========================================
Target: 0.845+ without external data.
Technique: 
Take the EXACT architecture of V10 (which scored 0.843 on LB) 
and train it 3 times with different random seeds. 
Average the final predictions to reduce variance.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from scipy.stats import rankdata

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Config
N_SPLITS = 5
N_REPEATS = 1
INPUT_DIR = "input"
OUTPUT_FILE = "submission_v14_seed_blend.csv"
SEEDS = [42, 2026, 777]

NUMERIC_COLS = ["Age","Height","Weight","Sprint_40yd","Vertical_Jump",
                "Bench_Press_Reps","Broad_Jump","Agility_3cone","Shuttle"]
DRILL_COLS = ["Sprint_40yd","Vertical_Jump","Bench_Press_Reps",
              "Broad_Jump","Agility_3cone","Shuttle"]
CAT_COLS = ["Player_Type","Position_Type","Position"]
TARGET = "Drafted"

POS_GROUP = {
    'CB':'DB','FS':'DB','SS':'DB','S':'DB','DB':'DB',
    'DE':'DL','DT':'DL','OT':'OL','OG':'OL','C':'OL',
    'OLB':'LB','ILB':'LB',
    'WR':'SKILL','RB':'SKILL','QB':'SKILL','TE':'SKILL','FB':'SKILL',
    'K':'SPEC','P':'SPEC','LS':'SPEC'
}

print("="*70)
print("NFL Draft V14 - V10 Architecture with Seed Blending")
print("="*70)

# === Load Data ===
train_df = pd.read_csv(os.path.join(INPUT_DIR,"train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR,"test.csv"))
test_ids = test_df["Id"].values
y_full = train_df[TARGET].values
n_train = len(train_df)

df_all = pd.concat([train_df.drop(columns=[TARGET]), test_df], ignore_index=True)

# === 1. Base Feature Engineering ===
print("[1] Base Feature Engineering...")

df_all["n_miss"] = df_all[DRILL_COLS].isnull().sum(axis=1)
df_all["BMI"] = (df_all["Weight"]*0.453592)/((df_all["Height"]*0.0254)**2)
df_all["Wt_per_in"] = df_all["Weight"]/df_all["Height"]
df_all["Ht_Wt_ratio"] = df_all["Height"]/df_all["Weight"]

df_all["Speed_Score"] = (df_all["Weight"]*200)/(df_all["Sprint_40yd"]**4+1e-8)
df_all["Explosion_Idx"] = df_all["Vertical_Jump"]*df_all["Broad_Jump"]/1000
df_all["Bench_Wt_Ratio"] = df_all["Bench_Press_Reps"]/(df_all["Weight"]+1e-8)
df_all["Agility_Score"] = 1.0/(df_all["Agility_3cone"]*df_all["Shuttle"]+1e-8)

df_all["Speed_Agility"] = df_all["Sprint_40yd"]*df_all["Agility_3cone"]
df_all["Explosiveness"] = df_all["Vertical_Jump"]+df_all["Broad_Jump"]
df_all["Str_Speed"] = df_all["Bench_Press_Reps"]/(df_all["Sprint_40yd"]+1e-8)
df_all["Year_norm"] = (df_all["Year"]-2009)/(2019-2009)
df_all["Pos_Group"] = df_all["Position"].map(POS_GROUP).fillna("OTHER")

for c in NUMERIC_COLS:
    grp_mean = df_all.groupby("Position")[c].transform("mean")
    grp_std = df_all.groupby("Position")[c].transform("std").replace(0,1)
    df_all[f"{c}_pos_z"] = (df_all[c] - grp_mean)/grp_std
    df_all[f"{c}_pos_pct"] = df_all.groupby("Position")[c].rank(pct=True)

school_counts = df_all["School"].value_counts()
valid_schools = set(school_counts[school_counts>=5].index)
df_all["School_Grp"] = df_all["School"].apply(lambda x: x if x in valid_schools else "Other")

cat_feats = CAT_COLS + ["School_Grp", "Pos_Group"]
for c in cat_feats: df_all[c] = df_all[c].astype(str)
df_all.drop(columns=["School","Id","Year"], inplace=True)

# === 2. Unsupervised Feature Engineering ===
print("[2] Unsupervised Feature Engineering (PCA & K-Means)...")
numeric_feats_for_unsup = NUMERIC_COLS + ["BMI", "Speed_Score", "Explosion_Idx"]
X_unsup = df_all[numeric_feats_for_unsup].copy()
for c in X_unsup.columns: X_unsup[c] = X_unsup[c].fillna(X_unsup[c].median())

scaler = StandardScaler()
X_unsup_scaled = scaler.fit_transform(X_unsup)

pca = PCA(n_components=5, random_state=42)
pca_feats = pca.fit_transform(X_unsup_scaled)
for i in range(5): df_all[f"PCA_{i}"] = pca_feats[:, i]

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
cluster_dists = kmeans.fit_transform(X_unsup_scaled)
for i in range(8): df_all[f"KMeans_Dist_{i}"] = cluster_dists[:, i]

X_train_base = df_all.iloc[:n_train].reset_index(drop=True)
X_test_base = df_all.iloc[n_train:].reset_index(drop=True)

def oof_te(X_tr, y_tr, X_val, X_te, cols, smoothing=20):
    gm = y_tr.mean()
    Xtr,Xv,Xte = X_tr.copy(), X_val.copy(), X_te.copy()
    for c in cols:
        tmp = Xtr.copy(); tmp['_y']=y_tr
        cnts = tmp.groupby(c).size()
        sums = tmp.groupby(c)['_y'].sum()
        enc = (sums + smoothing*gm)/(cnts + smoothing)
        nc = f"{c}_TE"
        Xtr[nc] = X_tr[c].map(enc).astype(float).fillna(gm)
        Xv[nc] = X_val[c].map(enc).astype(float).fillna(gm)
        Xte[nc] = X_te[c].map(enc).astype(float).fillna(gm)
        Xtr.drop(columns=[c],inplace=True); Xv.drop(columns=[c],inplace=True); Xte.drop(columns=[c],inplace=True)
    return Xtr,Xv,Xte

# === 3. Seed Blending Loop ===
print(f"\n[3] Training Meta-Ensemble across {len(SEEDS)} Seeds...")

seed_predictions = []

for SEED in SEEDS:
    print(f"\n---> Training with SEED: {SEED}")
    
    xgb_p = {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5, 'reg_alpha': 0.5, 'reg_lambda': 3.0, 'eval_metric': 'auc', 'tree_method': 'hist', 'random_state': SEED}
    lgb_p = {'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_samples': 20, 'reg_alpha': 0.5, 'reg_lambda': 3.0, 'verbose': -1, 'random_state': SEED}
    cb_p = {'iterations': 500, 'learning_rate': 0.05, 'depth': 4, 'l2_leaf_reg': 5.0, 'random_strength': 1.0, 'eval_metric': 'AUC', 'verbose': 0, 'random_seed': SEED}
    rf_p = {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 4, 'max_features': 0.5, 'n_jobs': 4, 'random_state': SEED}
    et_p = {'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 4, 'max_features': 0.5, 'n_jobs': 4, 'random_state': SEED}
    
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    
    preds_test = {'xgb':[], 'lgb':[], 'cb':[], 'rf':[], 'et':[], 'ridge':[], 'lr':[], 'svm':[]}
    oof_aucs = {k:[] for k in preds_test.keys()}
    
    for fold, (tr, va) in enumerate(rskf.split(X_train_base, y_full)):
        Xtr, ytr = X_train_base.iloc[tr], y_full[tr]
        Xva, yva = X_train_base.iloc[va], y_full[va]
        
        Xtr_e, Xva_e, Xte_e = oof_te(Xtr, ytr, Xva, X_test_base, cat_feats)
        Xtr_i, Xva_i, Xte_i = Xtr_e.fillna(-999), Xva_e.fillna(-999), Xte_e.fillna(-999)
        
        scaler = RobustScaler()
        Xtr_s = scaler.fit_transform(Xtr_i); Xva_s = scaler.transform(Xva_i); Xte_s = scaler.transform(Xte_i)
        
        # Models
        m1 = xgb.XGBClassifier(**xgb_p); m1.fit(Xtr_e, ytr, eval_set=[(Xva_e, yva)], verbose=0)
        preds_test['xgb'].append(m1.predict_proba(Xte_e)[:,1]); oof_aucs['xgb'].append(roc_auc_score(yva, m1.predict_proba(Xva_e)[:,1]))
        
        m2 = lgb.LGBMClassifier(**lgb_p); m2.fit(Xtr_e, ytr, eval_set=[(Xva_e, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds_test['lgb'].append(m2.predict_proba(Xte_e)[:,1]); oof_aucs['lgb'].append(roc_auc_score(yva, m2.predict_proba(Xva_e)[:,1]))
        
        Xtr_cb, Xva_cb, Xte_cb = Xtr.copy(), Xva.copy(), X_test_base.copy()
        for c in cat_feats: Xtr_cb[c] = Xtr_cb[c].astype(str).fillna("missing"); Xva_cb[c] = Xva_cb[c].astype(str).fillna("missing"); Xte_cb[c] = Xte_cb[c].astype(str).fillna("missing")
        m3 = cb.CatBoostClassifier(**{**cb_p, 'cat_features':cat_feats}); m3.fit(Xtr_cb, ytr, eval_set=(Xva_cb, yva), early_stopping_rounds=50, verbose=0)
        preds_test['cb'].append(m3.predict_proba(Xte_cb)[:,1]); oof_aucs['cb'].append(roc_auc_score(yva, m3.predict_proba(Xva_cb)[:,1]))
        
        m4 = RandomForestClassifier(**rf_p); m4.fit(Xtr_i, ytr)
        preds_test['rf'].append(m4.predict_proba(Xte_i)[:,1]); oof_aucs['rf'].append(roc_auc_score(yva, m4.predict_proba(Xva_i)[:,1]))
        
        m5 = ExtraTreesClassifier(**et_p); m5.fit(Xtr_i, ytr)
        preds_test['et'].append(m5.predict_proba(Xte_i)[:,1]); oof_aucs['et'].append(roc_auc_score(yva, m5.predict_proba(Xva_i)[:,1]))
        
        m6 = RidgeClassifier(alpha=10.0, random_state=SEED); m6.fit(Xtr_s, ytr)
        preds_test['ridge'].append(m6.decision_function(Xte_s)); oof_aucs['ridge'].append(roc_auc_score(yva, m6.decision_function(Xva_s)))
        
        m7 = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED); m7.fit(Xtr_s, ytr)
        preds_test['lr'].append(m7.predict_proba(Xte_s)[:,1]); oof_aucs['lr'].append(roc_auc_score(yva, m7.predict_proba(Xva_s)[:,1]))
        
        m8 = SVC(C=1.0, kernel='rbf', probability=True, random_state=SEED); m8.fit(Xtr_s, ytr)
        preds_test['svm'].append(m8.predict_proba(Xte_s)[:,1]); oof_aucs['svm'].append(roc_auc_score(yva, m8.predict_proba(Xva_s)[:,1]))
        
    weights = {k: np.mean(oof_aucs[k]) for k in oof_aucs}
    top_models = {k: v for k, v in weights.items() if v > 0.82}
    total_w = sum(top_models.values())
    top_models = {k: v/total_w for k, v in top_models.items()}
    
    final_pred_seed = np.zeros(len(X_test_base))
    for name in top_models.keys():
        model_pred = np.mean(preds_test[name], axis=0)
        final_pred_seed += (rankdata(model_pred) / len(model_pred)) * top_models[name]
        
    seed_predictions.append(final_pred_seed)
    print(f"Completed SEED {SEED} | Top Models Used: {list(top_models.keys())}")

# === 4. Final Averaging across Seeds ===
print("\n[4] Averaging Seeds & Blending with V4...")
avg_seed_pred = np.mean(seed_predictions, axis=0)
avg_seed_pred = rankdata(avg_seed_pred) / len(avg_seed_pred)

# Blend with historic V4 to solidify base ML power
try:
    s4 = pd.read_csv("submission_v4.csv")
    final_blend = (avg_seed_pred * 0.5) + (rankdata(s4['Drafted']) / len(s4) * 0.5)
except FileNotFoundError:
    final_blend = avg_seed_pred

# Save
sub = pd.DataFrame({"Id": test_ids, "Drafted": final_blend})
sub.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {OUTPUT_FILE}")
print("Done! Ready for Submission.")
