import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from models import Autoencoder

# === Argumenty CLI ===
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="./prepared_data", help="Folder z prepared_data")
parser.add_argument("--model", type=str, default="autoencoder_prepared_20250526_2321.pt", help="Plik .pt z modelem")
parser.add_argument("--thresholds", type=float, nargs="+", required=True, help="Lista progów do testowania")
parser.add_argument("--plot", action="store_true", help="Jeśli podane, generuje wykresy")
args = parser.parse_args()

# === Wczytanie danych
X_valid = np.load(os.path.join(args.data_dir, "X_valid.npy"))
X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

input_dim = X_valid.shape[1]
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# === Wczytanie modelu
model = Autoencoder(input_dim=input_dim)
model.load_state_dict(torch.load(args.model))
model.eval()

# === Obliczenie MSE
with torch.no_grad():
    output_valid = model(X_valid_tensor)
    mse_valid = torch.mean((output_valid - X_valid_tensor) ** 2, dim=1).numpy()
    
    output_test = model(X_test_tensor)
    mse_test = torch.mean((output_test - X_test_tensor) ** 2, dim=1).numpy()

# === Ewaluacja dla różnych progów
results = []

for t in args.thresholds:
    # Validacja: tylko FP
    fp_valid = int(np.sum(mse_valid > t))
    total_valid = len(mse_valid)
    fp_rate_valid = fp_valid / total_valid

    # Test: pełne metryki
    y_pred = (mse_test > t).astype(int)
    y_true = y_test.astype(int)

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-8)

    results.append({
        "Threshold": t,
        "Accuracy": acc,
        "Recall": rec,
        "Precision": prec,
        "F1": f1,
        "FPR": fpr,
        "FP_Valid": fp_valid,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn
    })

# === Tabela wyników
print("\n=== WYNIKI EWALUACJI ===\n")
print(f"{'Threshold':>9} {'F1':>6} {'Recall':>7} {'Precision':>9} {'FPR':>6} {'FP_Valid':>10} {'TP/All_Attacks':>18}")
total_attacks = int(np.sum(y_test))

for r in results:
    print(f"{r['Threshold']:9.3f} {r['F1']:6.3f} {r['Recall']:7.3f} {r['Precision']:9.3f} {r['FPR']:6.3f} {r['FP_Valid']:10d} {r['TP']:5d} / {total_attacks}")

# === Wykresy
if args.plot:
    thresholds = [r['Threshold'] for r in results]
    f1s = [r['F1'] for r in results]
    recalls = [r['Recall'] for r in results]
    fprs = [r['FPR'] for r in results]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(thresholds, f1s, marker='o')
    axs[0].set_title("F1 Score vs Threshold")
    axs[0].set_xlabel("Threshold")
    axs[0].set_ylabel("F1 Score")
    axs[0].grid(True)

    axs[1].plot(thresholds, recalls, marker='o', color='orange')
    axs[1].set_title("Recall vs Threshold")
    axs[1].set_xlabel("Threshold")
    axs[1].set_ylabel("Recall")
    axs[1].grid(True)

    axs[2].plot(thresholds, fprs, marker='o', color='red')
    axs[2].set_title("FPR vs Threshold")
    axs[2].set_xlabel("Threshold")
    axs[2].set_ylabel("False Positive Rate")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("plots/threshold_analysis.png")
    print("\nWykres zapisany jako: threshold_analysis.png")
