import os
import argparse
import pandas as pd
import numpy as np
import torch
from test_util import load_model, load_scaler, mse_tensor, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--models-dir", required=True)
parser.add_argument("--data-dir", default="separated")
parser.add_argument("--benign-test-csv", default="benign_splits/benign_test.csv")
parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10])
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = load_scaler()

# Zbierz modele
model_paths = [os.path.join(args.models_dir, f) for f in os.listdir(args.models_dir) if f.endswith(".pt")]
attack_files = [f for f in os.listdir(args.data_dir) if f.endswith(".csv") and not f.startswith("BENIGN")]

results = []

for model_path in sorted(model_paths):
    print(f"\nModel: {os.path.basename(model_path)}")
    model = None
    for attack_file in sorted(attack_files):
        attack_path = os.path.join(args.data_dir, attack_file)
        label = attack_file.split("__")[0]

        try:
            df_attack = pd.read_csv(attack_path, encoding="latin1").select_dtypes(include=["number"]).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            df_benign = pd.read_csv(args.benign_test_csv, encoding="latin1").select_dtypes(include=["number"]).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        except Exception as e:
            print(f"Błąd wczytywania {attack_file}: {e}")
            continue

        if len(df_attack) == 0:
            continue

        df_benign = df_benign.sample(n=min(len(df_attack), len(df_benign)), random_state=42)
        X_attack = scaler.transform(df_attack)
        X_benign = scaler.transform(df_benign)
        X = np.vstack([X_benign, X_attack])
        y = np.hstack([np.zeros(len(X_benign)), np.ones(len(X_attack))]).astype(int)

        if model is None:
            input_dim = X.shape[1]
            model = load_model(model_path, input_dim, DEVICE)

        X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            recon = model(X_tensor)
            mse_vals = mse_tensor(X_tensor, recon).detach().cpu().numpy()

        best = None
        for th in args.thresholds:
            y_pred = (mse_vals > th).astype(int)
            m = compute_metrics(y, y_pred)
            if best is None or m["f1"] > best["f1"]:
                best = {"threshold": th, **m}

        results.append({
            "attack": label,
            "model": os.path.basename(model_path),
            **best
        })
        print(f"  {label:25} best_f1={best['f1']:.3f}  recall={best['recall']:.3f}  thr={best['threshold']:.3f}")

# Zapisz zbiorczy raport
out_path = f"benchmark/benchmark_single_results_{args.models_dir}.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print("\nZapisano podsumowanie do:", out_path)
