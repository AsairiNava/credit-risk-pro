import pandas as pd, glob
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW=Path("data/raw"); OUT=Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
paths = glob.glob(str(RAW/"*.csv")) + glob.glob(str(RAW/"*.xls*"))
if not paths: raise SystemExit("No hay archivos en data/raw. Descarga primero.")
p = paths[0]
df = pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
df.columns=[c.strip().lower().replace(" ","_").replace(".","_") for c in df.columns]
t="default_payment_next_month"
if t not in df.columns:
    cand=[c for c in df.columns if "default" in c and "next" in c]
    if not cand: raise SystemExit("No encontré la columna target.")
    t=cand[0]
y=df[t].astype(int); X=df.drop(columns=[t])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
Xtr.to_parquet(OUT/"X_train.parquet"); Xte.to_parquet(OUT/"X_test.parquet")
ytr.to_frame("target").to_parquet(OUT/"y_train.parquet"); yte.to_frame("target").to_parquet(OUT/"y_test.parquet")
print("✅ Datos listos en data/processed")
