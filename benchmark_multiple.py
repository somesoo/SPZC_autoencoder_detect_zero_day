import os
import argparse
import pandas as pd
import numpy as np
import torch
import itertools
from test_util import load_model, load_scaler, mse_tensor
from sklearn.metrics import recall_score

parser = argparse.ArgumentParser()
parser.add_argument("--models-dir", required=True)
parser.add_argument("--data-dir", default="separated")
parser.add_argument("--benign-test-csv", default="benign_splits/benign_test.csv")
parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01,0.02,0.03,0.04,0.05])
parser.add_argument("--n", type=int, default=5, help="Ilość kombinacji")
parser.add_argument("--c", type=int, default=2, help="Ile ataków w jednej kombinacji")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = load_scaler()
model_paths = [os.path.join(args.models_dir, f) for f in os.listdir(args.models_dir) if f.endswith(".pt")]

attack_files = [f for f in os.listdir(args.data_dir) if f.endswith(".csv") and not f.startswith("BENIGN")]
attack_paths = [os.path.join(args.data_dir, f) for f in attack_files]

attack_combos = list(itertools.combinations(attack_paths, args.c))[:args.n]

results = []

for combo in attack_combos:
    labels = [os.path.basename(p).split("__")[0] for p in combo]
    attack_label = "+".join(labels)
    print(f"\nKombinacja: {attack_label}")

    try:
        df_attacks = [pd.read_csv(p, encoding="latin1").select_dtypes(include=["number"]).fillna(0.0).replace([np.inf, -np.inf], 0.0) for p in combo]
        df_attack = pd.concat(df_attacks, ignore_index=True)
        df_benign = pd.read_csv(args.benign_test_csv, encoding="latin1").select_dtypes(include=["number"]).fillna(0.0).replace([np.inf, -np.inf], 0.0)
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

        best = {"f1": 0.0, "recall": 0.0, "threshold": None, "tp": 0, "fn": 0, "fpr": 0.0}

        for th in args.thresholds:
            y_pred = (mse_vals > th).astype(int)
            tp = ((y_pred == 1) & (y == 1)).sum()
            fn = ((y_pred == 0) & (y == 1)).sum()
            fpr = ((y_pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = 2 * (tp / (tp + fn + 1e-6)) * (tp / (tp + ((y_pred == 1) & (y == 0)).sum() + 1e-6)) / (tp / (tp + fn + 1e-6) + tp / (tp + ((y_pred == 1) & (y == 0)).sum() + 1e-6) + 1e-6)
            if f1 > best["f1"]:
                best.update({"threshold": th, "recall": rec, "f1": f1, "tp": tp, "fn": fn, "fpr": fpr})

        results.append({
            "attacks": attack_label,
            "model": model_name,
            "threshold": best["threshold"],
            "tp": best["tp"],
            "fn": best["fn"],
            "recall": best["recall"],
            "f1": best["f1"],
            "fpr": best["fpr"]
        })
        print(f"  Model: {model_name:35} TP={best['tp']}, FN={best['fn']}, Recall={best['recall']:.3f}, F1={best['f1']:.3f}, FPR={best['fpr']:.3f}, Th={best['threshold']:.3f}")

pd.DataFrame(results).to_csv(f"benchmark/benchmark_pairwise_results{args.c}.csv", index=False)
print(f"\nZapisano benchmark do benchmark_pairwise_results{args.c}.csv")
