import argparse
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="uciml/default-of-credit-card-clients-dataset")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--unzip", action="store_true")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files(a.dataset, path=str(out), unzip=a.unzip)
    print(f"✅ Descargado en {out}")

if __name__ == "__main__":
    main()
