import pandas as pd, numpy as np, joblib, streamlit as st
from pathlib import Path

st.set_page_config(page_title="Riesgo Crediticio", page_icon="📊", layout="wide")
st.title("📊 Riesgo Crediticio – UCI (Pro)")

bundle_path = Path("models/lgbm_calibrated.joblib")
thr_path = Path("models/threshold.txt")

if not bundle_path.exists():
    st.error("No se encontró el modelo. Sube 'models/lgbm_calibrated.joblib' al repo.")
    st.stop()

bundle = joblib.load(bundle_path)
clf, cols = bundle["model"], bundle["columns"]

raw_thr = thr_path.read_text().strip().replace("%","") if thr_path.exists() else "0.50"
thr_default = float(raw_thr) if raw_thr else 0.5

st.sidebar.header("⚙️ Configuración")
thr_user = st.sidebar.slider("Umbral de aprobación", 0.01, 0.99, thr_default, 0.01)

up = st.file_uploader("Sube CSV o Parquet con las mismas columnas que el train", type=["csv","parquet"])
if up:
    df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_parquet(up)
    df.columns = [c.strip().lower().replace(" ","_").replace(".","_") for c in df.columns]
    tgt = [c for c in df.columns if "default" in c and "next" in c]
    if tgt: df = df.drop(columns=tgt)
    X = df.reindex(columns=cols, fill_value=0)
    p = clf.predict_proba(X)[:,1]
    pred = (p >= thr_user).astype(int)
    res = df.copy()
    res["risk_probability"] = p.round(4)
    res["prediction"] = pred
    res["bucket"] = pd.cut(res["risk_probability"], [-.01,.33,.66,1], labels=["Bajo","Medio","Alto"])
    st.dataframe(res.head(200))
    st.download_button("⬇️ Descargar resultados", res.to_csv(index=False).encode("utf-8"),
                       file_name="predicciones_riesgo.csv", mime="text/csv")
else:
    st.info("Sube un archivo para obtener probabilidades por cliente.")
