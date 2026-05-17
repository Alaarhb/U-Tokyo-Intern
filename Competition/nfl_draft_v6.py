"""
NFL Draft Prediction - V6 (Target: 0.88+)
==========================================
- Rich position-aware feature engineering
- 2-Level Stacking: L1(XGB+LGB+CB+ExtraTrees) -> L2(LogisticRegression)
- Multi-seed averaging (7 seeds)
- Optuna tuning (80 trials/model)
- Proper OOF target encoding
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

# Config
N_FOLDS = 10
N_SEEDS = 7
OPTUNA_TRIALS = 80
INPUT_DIR = "input"
OUTPUT_FILE = "submission_v6.csv"

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

print("="*70); print("NFL Draft V6 - 2-Level Stacking"); print("="*70)
sys.stdout.flush()

# === Load ===
train_df = pd.read_csv(os.path.join(INPUT_DIR,"train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR,"test.csv"))
test_ids = test_df["Id"].values
y_full = train_df[TARGET].values
n_train = len(train_df)

df_all = pd.concat([train_df.drop(columns=[TARGET]), test_df], ignore_index=True)

# === Feature Engineering ===
print("[1] Feature Engineering..."); sys.stdout.flush()

# Missing indicators
for c in NUMERIC_COLS:
    df_all[f"{c}_miss"] = df_all[c].isnull().astype(int)
df_all["n_miss"] = df_all[DRILL_COLS].isnull().sum(axis=1)
df_all["n_done"] = 6 - df_all["n_miss"]
df_all["all_done"] = (df_all["n_miss"]==0).astype(int)

# Missing pattern as integer code
mp = df_all[DRILL_COLS].isnull().astype(int).astype(str).apply(''.join, axis=1)
df_all["miss_pattern"] = pd.Categorical(mp).codes

# Body
df_all["BMI"] = (df_all["Weight"]*0.453592)/((df_all["Height"]*0.0254)**2)
df_all["Wt_per_in"] = df_all["Weight"]/df_all["Height"]
df_all["Ht_Wt_ratio"] = df_all["Height"]/df_all["Weight"]

# NFL scouting
df_all["Speed_Score"] = (df_all["Weight"]*200)/(df_all["Sprint_40yd"]**4+1e-8)
df_all["Ht_Adj_Speed"] = df_all["Sprint_40yd"]/(df_all["Height"]+1e-8)
df_all["Explosion_Idx"] = df_all["Vertical_Jump"]*df_all["Broad_Jump"]/1000
df_all["Bench_Wt_Ratio"] = df_all["Bench_Press_Reps"]/(df_all["Weight"]+1e-8)
df_all["Agility_Score"] = 1.0/(df_all["Agility_3cone"]*df_all["Shuttle"]+1e-8)

# Composites
df_all["Speed_Agility"] = df_all["Sprint_40yd"]*df_all["Agility_3cone"]
df_all["Explosiveness"] = df_all["Vertical_Jump"]+df_all["Broad_Jump"]
df_all["Str_Speed"] = df_all["Bench_Press_Reps"]/(df_all["Sprint_40yd"]+1e-8)
df_all["Agil_Shuttle_Avg"] = (df_all["Agility_3cone"]+df_all["Shuttle"])/2
df_all["Power_Score"] = df_all["Bench_Press_Reps"]*df_all["Vertical_Jump"]
df_all["Lower_Body_Pwr"] = df_all["Vertical_Jump"]*df_all["Broad_Jump"]*df_all["Weight"]
df_all["Upper_Lower_R"] = df_all["Bench_Press_Reps"]/(df_all["Vertical_Jump"]+1e-8)

# Key pairwise ratios
pairs = [("Sprint_40yd","Weight"),("Vertical_Jump","Weight"),
         ("Broad_Jump","Sprint_40yd"),("Bench_Press_Reps","Sprint_40yd"),
         ("Agility_3cone","Sprint_40yd"),("Shuttle","Sprint_40yd"),
         ("Vertical_Jump","Broad_Jump"),("Bench_Press_Reps","Weight")]
for c1,c2 in pairs:
    df_all[f"{c1}_div_{c2}"] = df_all[c1]/(df_all[c2]+1e-8)

# Position group
df_all["Pos_Group"] = df_all["Position"].map(POS_GROUP).fillna("OTHER")

# Position-relative percentile ranks (within full data to get best signal)
for c in NUMERIC_COLS:
    df_all[f"{c}_pos_pct"] = df_all.groupby("Position")[c].rank(pct=True)
    df_all[f"{c}_grp_pct"] = df_all.groupby("Pos_Group")[c].rank(pct=True)

# Position-relative z-scores (within full data)
for c in NUMERIC_COLS:
    grp_mean = df_all.groupby("Position")[c].transform("mean")
    grp_std = df_all.groupby("Position")[c].transform("std").replace(0,1)
    df_all[f"{c}_pos_z"] = (df_all[c] - grp_mean)/grp_std

# School grouping
school_counts = df_all["School"].value_counts()
valid_schools = set(school_counts[school_counts>=5].index)
df_all["School_Grp"] = df_all["School"].apply(lambda x: x if x in valid_schools else "Other")

# Year features
df_all["Year_norm"] = (df_all["Year"]-2009)/(2019-2009)

# Categoricals
cat_feats = CAT_COLS + ["School_Grp","Pos_Group"]
for c in cat_feats:
    df_all[c] = df_all[c].astype(str)

df_all.drop(columns=["School","Id","Year"], inplace=True)

X_train = df_all.iloc[:n_train].reset_index(drop=True)
X_test = df_all.iloc[n_train:].reset_index(drop=True)

print(f"  Features: {X_train.shape[1]} ({len(cat_feats)} categorical)")
sys.stdout.flush()

# === OOF Target Encoding ===
def oof_te(X_tr, y_tr, X_val, X_te, cols, smoothing=10):
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
        Xtr.drop(columns=[c],inplace=True)
        Xv.drop(columns=[c],inplace=True)
        Xte.drop(columns=[c],inplace=True)
    return Xtr,Xv,Xte

# === Optuna Tuning (5-fold for speed) ===
print(f"\n[2] Optuna Tuning ({OPTUNA_TRIALS} trials/model)..."); sys.stdout.flush()

def tune_xgb(trial):
    p = {'n_estimators':1000,'max_depth':trial.suggest_int('md',3,7),
         'learning_rate':trial.suggest_float('lr',0.01,0.1,log=True),
         'subsample':trial.suggest_float('ss',0.6,0.9),
         'colsample_bytree':trial.suggest_float('cs',0.4,0.9),
         'min_child_weight':trial.suggest_int('mcw',1,20),
         'reg_alpha':trial.suggest_float('ra',1e-3,10,log=True),
         'reg_lambda':trial.suggest_float('rl',1e-3,10,log=True),
         'gamma':trial.suggest_float('g',0,2),
         'random_state':42,'eval_metric':'auc','early_stopping_rounds':50,
         'tree_method':'hist'}
    skf = StratifiedKFold(5,shuffle=True,random_state=42)
    scores=[]
    for tr,va in skf.split(X_train,y_full):
        Xtr,Xv,_ = oof_te(X_train.iloc[tr],y_full[tr],X_train.iloc[va],X_test,cat_feats)
        m=xgb.XGBClassifier(**p); m.fit(Xtr,y_full[tr],eval_set=[(Xv,y_full[va])],verbose=0)
        scores.append(roc_auc_score(y_full[va],m.predict_proba(Xv)[:,1]))
    return np.mean(scores)

def tune_lgb(trial):
    p = {'n_estimators':1000,'max_depth':trial.suggest_int('md',3,7),
         'learning_rate':trial.suggest_float('lr',0.01,0.1,log=True),
         'subsample':trial.suggest_float('ss',0.6,0.9),
         'colsample_bytree':trial.suggest_float('cs',0.4,0.9),
         'min_child_weight':trial.suggest_float('mcw',0.001,10,log=True),
         'reg_alpha':trial.suggest_float('ra',1e-3,10,log=True),
         'reg_lambda':trial.suggest_float('rl',1e-3,10,log=True),
         'num_leaves':trial.suggest_int('nl',15,63),
         'min_child_samples':trial.suggest_int('mcs',5,50),
         'random_state':42,'verbose':-1}
    skf = StratifiedKFold(5,shuffle=True,random_state=42)
    scores=[]
    for tr,va in skf.split(X_train,y_full):
        Xtr,Xv,_ = oof_te(X_train.iloc[tr],y_full[tr],X_train.iloc[va],X_test,cat_feats)
        m=lgb.LGBMClassifier(**p); m.fit(Xtr,y_full[tr],eval_set=[(Xv,y_full[va])],callbacks=[lgb.early_stopping(50,verbose=False)])
        scores.append(roc_auc_score(y_full[va],m.predict_proba(Xv)[:,1]))
    return np.mean(scores)

def tune_cb(trial):
    p = {'iterations':1000,'depth':trial.suggest_int('d',4,8),
         'learning_rate':trial.suggest_float('lr',0.01,0.1,log=True),
         'l2_leaf_reg':trial.suggest_float('l2',0.1,10,log=True),
         'bagging_temperature':trial.suggest_float('bt',0,2),
         'random_strength':trial.suggest_float('rs',0.1,5),
         'border_count':trial.suggest_int('bc',32,255),
         'random_seed':42,'verbose':0,'eval_metric':'AUC',
         'early_stopping_rounds':50,'cat_features':cat_feats}
    skf = StratifiedKFold(5,shuffle=True,random_state=42)
    scores=[]
    for tr,va in skf.split(X_train,y_full):
        Xtr = X_train.iloc[tr].copy(); Xv = X_train.iloc[va].copy()
        for c in cat_feats:
            Xtr[c] = Xtr[c].astype(str).fillna('missing')
            Xv[c] = Xv[c].astype(str).fillna('missing')
        m=cb.CatBoostClassifier(**p); m.fit(Xtr,y_full[tr],eval_set=(Xv,y_full[va]),verbose=0)
        scores.append(roc_auc_score(y_full[va],m.predict_proba(Xv)[:,1]))
    return np.mean(scores)

def tune_et(trial):
    p = {'n_estimators':trial.suggest_int('ne',300,1500),
         'max_depth':trial.suggest_int('md',5,20),
         'min_samples_split':trial.suggest_int('mss',5,30),
         'min_samples_leaf':trial.suggest_int('msl',2,15),
         'max_features':trial.suggest_float('mf',0.3,0.9),
         'random_state':42,'n_jobs':-1}
    skf = StratifiedKFold(5,shuffle=True,random_state=42)
    scores=[]
    for tr,va in skf.split(X_train,y_full):
        Xtr,Xv,_ = oof_te(X_train.iloc[tr],y_full[tr],X_train.iloc[va],X_test,cat_feats)
        Xtr=Xtr.fillna(-999); Xv=Xv.fillna(-999)
        m=ExtraTreesClassifier(**p); m.fit(Xtr,y_full[tr])
        scores.append(roc_auc_score(y_full[va],m.predict_proba(Xv)[:,1]))
    return np.mean(scores)

print("  XGBoost..."); sys.stdout.flush()
s1=optuna.create_study(direction='maximize'); s1.optimize(tune_xgb,n_trials=OPTUNA_TRIALS)
xgb_p=s1.best_params; print(f"    Best CV: {s1.best_value:.6f}"); sys.stdout.flush()

print("  LightGBM..."); sys.stdout.flush()
s2=optuna.create_study(direction='maximize'); s2.optimize(tune_lgb,n_trials=OPTUNA_TRIALS)
lgb_p=s2.best_params; print(f"    Best CV: {s2.best_value:.6f}"); sys.stdout.flush()

print("  CatBoost..."); sys.stdout.flush()
s3=optuna.create_study(direction='maximize'); s3.optimize(tune_cb,n_trials=OPTUNA_TRIALS)
cb_p=s3.best_params; print(f"    Best CV: {s3.best_value:.6f}"); sys.stdout.flush()

print("  ExtraTrees..."); sys.stdout.flush()
s4=optuna.create_study(direction='maximize'); s4.optimize(tune_et,n_trials=OPTUNA_TRIALS)
et_p=s4.best_params; print(f"    Best CV: {s4.best_value:.6f}"); sys.stdout.flush()

# Rebuild full param dicts
xgb_params = {k.replace('md','max_depth').replace('lr','learning_rate').replace('ss','subsample')
               .replace('cs','colsample_bytree').replace('mcw','min_child_weight')
               .replace('ra','reg_alpha').replace('rl','reg_lambda').replace('g','gamma'):v 
               for k,v in xgb_p.items()}
xgb_params.update({'n_estimators':1200,'eval_metric':'auc','early_stopping_rounds':60,'tree_method':'hist'})

lgb_params = {k.replace('md','max_depth').replace('lr','learning_rate').replace('ss','subsample')
               .replace('cs','colsample_bytree').replace('mcw','min_child_weight')
               .replace('ra','reg_alpha').replace('rl','reg_lambda')
               .replace('nl','num_leaves').replace('mcs','min_child_samples'):v 
               for k,v in lgb_p.items()}
lgb_params.update({'n_estimators':1200,'verbose':-1})

cb_params = {k.replace('d','depth').replace('lr','learning_rate').replace('l2','l2_leaf_reg')
              .replace('bt','bagging_temperature').replace('rs','random_strength')
              .replace('bc','border_count'):v for k,v in cb_p.items()}
cb_params.update({'iterations':1200,'verbose':0,'eval_metric':'AUC','early_stopping_rounds':60})

et_params = {k.replace('ne','n_estimators').replace('md','max_depth')
              .replace('mss','min_samples_split').replace('msl','min_samples_leaf')
              .replace('mf','max_features'):v for k,v in et_p.items()}
et_params.update({'n_jobs':-1})

# === Multi-Seed 2-Level Stacking ===
print(f"\n[3] Multi-Seed Stacking ({N_SEEDS} seeds x 4 models x {N_FOLDS} folds)...")
sys.stdout.flush()

all_test_preds = []
all_oof_aucs = []

for si in range(N_SEEDS):
    seed = 42 + si*7
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    folds = list(skf.split(X_train, y_full))
    
    # Level-1 OOF predictions
    xgb_oof=np.zeros(n_train); xgb_te=np.zeros(len(X_test))
    lgb_oof=np.zeros(n_train); lgb_te=np.zeros(len(X_test))
    cb_oof=np.zeros(n_train);  cb_te=np.zeros(len(X_test))
    et_oof=np.zeros(n_train);  et_te=np.zeros(len(X_test))
    
    for fi,(tr,va) in enumerate(folds):
        Xtr,ytr = X_train.iloc[tr], y_full[tr]
        Xva,yva = X_train.iloc[va], y_full[va]
        
        # OOF encode for XGB/LGB/ET
        Xtr_e,Xva_e,Xte_e = oof_te(Xtr,ytr,Xva,X_test,cat_feats)
        
        # XGBoost
        m=xgb.XGBClassifier(**{**xgb_params,'random_state':seed})
        m.fit(Xtr_e,ytr,eval_set=[(Xva_e,yva)],verbose=0)
        xgb_oof[va]=m.predict_proba(Xva_e)[:,1]
        xgb_te+=m.predict_proba(Xte_e)[:,1]/N_FOLDS
        
        # LightGBM
        m=lgb.LGBMClassifier(**{**lgb_params,'random_state':seed})
        m.fit(Xtr_e,ytr,eval_set=[(Xva_e,yva)],callbacks=[lgb.early_stopping(60,verbose=False)])
        lgb_oof[va]=m.predict_proba(Xva_e)[:,1]
        lgb_te+=m.predict_proba(Xte_e)[:,1]/N_FOLDS
        
        # ExtraTrees
        Xtr_f=Xtr_e.fillna(-999); Xva_f=Xva_e.fillna(-999); Xte_f=Xte_e.fillna(-999)
        m=ExtraTreesClassifier(**{**et_params,'random_state':seed})
        m.fit(Xtr_f,ytr)
        et_oof[va]=m.predict_proba(Xva_f)[:,1]
        et_te+=m.predict_proba(Xte_f)[:,1]/N_FOLDS
        
        # CatBoost (native categoricals)
        Xtr_cb=Xtr.copy(); Xva_cb=Xva.copy(); Xte_cb=X_test.copy()
        for c in cat_feats:
            Xtr_cb[c]=Xtr_cb[c].astype(str).fillna('missing')
            Xva_cb[c]=Xva_cb[c].astype(str).fillna('missing')
            Xte_cb[c]=Xte_cb[c].astype(str).fillna('missing')
        m=cb.CatBoostClassifier(**{**cb_params,'random_seed':seed,'cat_features':cat_feats})
        m.fit(Xtr_cb,ytr,eval_set=(Xva_cb,yva),verbose=0)
        cb_oof[va]=m.predict_proba(Xva_cb)[:,1]
        cb_te+=m.predict_proba(Xte_cb)[:,1]/N_FOLDS
    
    # Level-1 AUCs
    a1=roc_auc_score(y_full,xgb_oof)
    a2=roc_auc_score(y_full,lgb_oof)
    a3=roc_auc_score(y_full,cb_oof)
    a4=roc_auc_score(y_full,et_oof)
    print(f"  Seed {seed}: XGB={a1:.4f} LGB={a2:.4f} CB={a3:.4f} ET={a4:.4f}")
    sys.stdout.flush()
    
    # Level-2: Stack with LogisticRegression
    oof_stack = np.column_stack([xgb_oof,lgb_oof,cb_oof,et_oof])
    te_stack = np.column_stack([xgb_te,lgb_te,cb_te,et_te])
    
    meta = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)
    meta.fit(oof_stack, y_full)
    stack_oof = meta.predict_proba(oof_stack)[:,1]
    stack_te = meta.predict_proba(te_stack)[:,1]
    stack_auc = roc_auc_score(y_full, stack_oof)
    print(f"    Stack AUC: {stack_auc:.6f}")
    sys.stdout.flush()
    
    # Also try rank averaging as alternative
    r1=rankdata(xgb_te)/len(xgb_te)
    r2=rankdata(lgb_te)/len(lgb_te)
    r3=rankdata(cb_te)/len(cb_te)
    r4=rankdata(et_te)/len(et_te)
    tw=a1+a2+a3+a4
    rank_te = (a1*r1+a2*r2+a3*r3+a4*r4)/tw
    
    # Blend stack + rank
    blend_te = 0.6*stack_te + 0.4*rank_te
    
    all_test_preds.append(blend_te)
    all_oof_aucs.append(stack_auc)

final = np.mean(all_test_preds, axis=0)
print(f"\nMean Stack OOF AUC: {np.mean(all_oof_aucs):.6f}")
print(f"Std:  {np.std(all_oof_aucs):.6f}")

# === Save ===
sub = pd.DataFrame({"Id": test_ids, "Drafted": final})
sub.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {OUTPUT_FILE}")
print("Done!")
