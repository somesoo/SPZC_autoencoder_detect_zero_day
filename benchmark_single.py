import os
import argparse
import pandas as pd
import numpy as np
import torch
from test_util import load_model, load_scaler, mse_tensor
from sklearn.metrics import recall_score
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--models-dir", required=True)
parser.add_argument("--data-dir", default="separated_kitsune")
parser.add_argument("--benign-test-csv", default="benign_splits/benign_test.csv")
parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01,0.02,0.03,0.04,0.05])
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = load_scaler()
model_paths = [os.path.join(args.models_dir, f) for f in os.listdir(args.models_dir) if f.endswith(".pt")]

attack_files = [f for f in os.listdir(args.data_dir) if f.endswith(".csv") and not f.lower().startswith("benign")]
timestamp = datetime.now().strftime("%Y%m%d")
out_root = os.path.join("tests", f"{timestamp}_kitsune_single")
os.makedirs(out_root, exist_ok=True)

def clean_kitsune_df(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    df = df.iloc[:, :-1]  # usuń ostatnią kolumnę (etykieta)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df

# === benchmark: każdy atak osobno
for attack_file in attack_files:
    attack_path = os.path.join(args.data_dir, attack_file)
    label = attack_file.split("__")[0]
    print(f"\nTest ataku: {label}")

    try:
        df_attack = clean_kitsune_df(attack_path)
        df_benign = clean_kitsune_df(args.benign_test_csv)
    except Exception as e:
        print(f"Błąd wczytywania: {e}")
        continue

    X_attack = scaler.transform(df_attack)
    X_benign = scaler.transform(df_benign)
    X = np.vstack([X_benign, X_attack])
    y = np.hstack([np.zeros(len(X_benign)), np.ones(len(X_attack))]).astype(int)

    for model_path in sorted(model_paths):
        model_name = os.path.basename(model_path)
        model = load_model(model_path, X.shape[1], DEVICE)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            recon = model(X_tensor)
            mse_vals = mse_tensor(X_tensor, recon).detach().cpu().numpy()

        per_threshold_rows = []

        for th in args.thresholds:
            y_pred = (mse_vals > th).astype(int)
            tp = ((y_pred == 1) & (y == 1)).sum()
            fn = ((y_pred == 0) & (y == 1)).sum()
            fp = ((y_pred == 1) & (y == 0)).sum()
            tn = ((y_pred == 0) & (y == 0)).sum()
            fpr = fp / max(1, (fp + tn))
            rec = recall_score(y, y_pred, zero_division=0)
            precision = tp / (tp + fp + 1e-6)
            recall_ = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall_ / (precision + recall_ + 1e-6)

            per_threshold_rows.append({
                "attack": label,
                "model": model_name,
                "threshold": th,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall_, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 4)
            })


        subdir = os.path.join(out_root, f"{label}_{model_name}".replace(".pt", ""))
        os.makedirs(subdir, exist_ok=True)
        np.save(os.path.join(subdir, "X.npy"), X)
        np.save(os.path.join(subdir, "y.npy"), y)
        np.save(os.path.join(subdir, "mse_vals.npy"), mse_vals)

# zapis końcowy
print(f"\nZapisano benchmark do {out_root}/benchmark_single_results.csv")
# zapis zbiorczy wszystkich wyników (per threshold)
pd.DataFrame(per_threshold_rows).to_csv(os.path.join(out_root, "benchmark_single_detailed.csv"), index=False)
