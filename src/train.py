import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import optuna

DATA=Path("data/processed"); MODELS=Path("models"); REPORTS=Path("reports")
MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

def load_data():
    Xtr=pd.read_parquet(DATA/"X_train.parquet")
    ytr=pd.read_parquet(DATA/"y_train.parquet")["target"].values
    Xte=pd.read_parquet(DATA/"X_test.parquet")
    yte=pd.read_parquet(DATA/"y_test.parquet")["target"].values
    return Xtr,ytr,Xte,yte

def objective(trial,X,y):
    params=dict(
        n_estimators=trial.suggest_int("n_estimators",300,900),
        learning_rate=trial.suggest_float("learning_rate",0.01,0.2,log=True),
        num_leaves=trial.suggest_int("num_leaves",31,127),
        subsample=trial.suggest_float("subsample",0.7,1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree",0.7,1.0),
        random_state=42, n_jobs=-1
    )
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    aucs=[]
    for tr,va in cv.split(X,y):
        clf=LGBMClassifier(**params); clf.fit(X.iloc[tr],y[tr])
        p=clf.predict_proba(X.iloc[va])[:,1]; aucs.append(roc_auc_score(y[va],p))
    return float(np.mean(aucs))

def main(cost_fp=1.0,cost_fn=5.0,n_trials=20):
    Xtr,ytr,Xte,yte=load_data()
    study=optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t,Xtr,ytr),n_trials=n_trials,show_progress_bar=False)
    best=study.best_params
    base=LGBMClassifier(**best).fit(Xtr,ytr)
    cal=CalibratedClassifierCV(base,method="sigmoid",cv=5).fit(Xtr,ytr)
    proba=cal.predict_proba(Xte)[:,1]
    auc=roc_auc_score(yte,proba); ap=average_precision_score(yte,proba)

    # umbral por costo
    ts=np.linspace(0.01,0.99,99); best_t=0.5; best_c=1e18
    for t in ts:
        pred=(proba>=t).astype(int)
        fp=((pred==1)&(yte==0)).sum(); fn=((pred==0)&(yte==1)).sum()
        c=fp*cost_fp+fn*cost_fn
        if c<best_c: best_c, best_t=c, float(t)

    joblib.dump({"model":cal,"columns":Xtr.columns.tolist()}, MODELS/"lgbm_calibrated.joblib")
    (MODELS/"threshold.txt").write_text(f"{best_t:.2f}\n")
    (REPORTS/"metrics.json").write_text(json.dumps(
        {"AUC":float(auc),"AP":float(ap),"threshold_opt":best_t,"cost_opt":best_c}, indent=2))
    print(f"✅ AUC={auc:.3f} | AP={ap:.3f} | threshold*={best_t:.2f} | cost*={best_c:.1f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cost-fp",type=float,default=1.0)
    ap.add_argument("--cost-fn",type=float,default=5.0)
    ap.add_argument("--n-trials",type=int,default=20)
    a=ap.parse_args()
    main(a.cost_fp,a.cost_fn,a.n_trials)
