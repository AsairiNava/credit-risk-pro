import argparse, pandas as pd, numpy as np, joblib
from pathlib import Path

MODELS=Path("models")
bundle=joblib.load(MODELS/"lgbm_calibrated.joblib")
clf, cols = bundle["model"], bundle["columns"]
thr_path=MODELS/"threshold.txt"
raw = thr_path.read_text().strip().replace("%","") if thr_path.exists() else "0.50"
thr=float(raw) if raw else 0.5

ap=argparse.ArgumentParser()
ap.add_argument("--in",dest="in_path",required=True)
ap.add_argument("--out",required=True)
ap.add_argument("--threshold",type=float,default=None)
a=ap.parse_args()
thr=a.threshold if a.threshold is not None else thr

df = pd.read_csv(a.in_path) if a.in_path.endswith(".csv") else pd.read_parquet(a.in_path)
df.columns=[c.strip().lower().replace(" ","_").replace(".","_") for c in df.columns]
tgt=[c for c in df.columns if "default" in c and "next" in c]
if tgt: df=df.drop(columns=tgt)
X=df.reindex(columns=cols,fill_value=0)
p=clf.predict_proba(X)[:,1]; pred=(p>=thr).astype(int)
out=df.copy(); out["risk_probability"]=p.round(4); out["prediction"]=pred
out["risk_bucket"]=pd.cut(out["risk_probability"],[-.01,.33,.66,1],labels=["Bajo","Medio","Alto"])
(out.to_csv if a.out.endswith(".csv") else out.to_parquet)(a.out,index=False)
print(f"✅ Guardado {a.out} (umbral={thr:.2f})")
