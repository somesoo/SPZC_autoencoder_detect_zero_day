import argparse, os
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from test_util import load_scaler, load_model, mse_tensor, compute_metrics, save_artifacts
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_kitsune_csv(path):
    df = pd.read_csv(path, header=None, low_memory=False)
    df = df.iloc[:, :-1]  # usuń ostatnią kolumnę (etykieta: 0 lub 1)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df

def build_kitsune_dataset(attack_files, benign_file, scaler):
    benign_df = clean_kitsune_csv(benign_file)
    X_benign = scaler.transform(benign_df)
    y_labels = ["BENIGN"] * len(X_benign)

    attacks_df_list = []
    attack_labels = []

    for f in attack_files:
        df = clean_kitsune_csv(f)
        label = os.path.basename(f).split("__")[0]  # np. 'scan', 'flooding', etc.
        df["Label"] = label
        attacks_df_list.append(df)
        attack_labels.extend([label] * len(df))

    if attacks_df_list:
        attack_df = pd.concat(attacks_df_list, ignore_index=True)
        attack_labels_final = attack_df["Label"].tolist()
        X_attack = scaler.transform(attack_df.drop(columns=["Label"]))
    else:
        X_attack = np.empty((0, scaler.mean_.shape[0]))
        attack_labels_final = []

    X_test = np.vstack([X_benign, X_attack])
    y_test = np.hstack([
        np.zeros(len(X_benign)),
        np.ones(len(X_attack))
    ]).astype(int)

    y_labels = y_labels + attack_labels_final
    return X_test, y_test, y_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ścieżka do modelu .pt")
    parser.add_argument("--benign-file", required=True, help="benign_test.csv")
    parser.add_argument("--attack-files", nargs="*", help="lista plików attack CSV")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.1])
    args = parser.parse_args()

    scaler = load_scaler()
    X_test, y_test, y_labels = build_kitsune_dataset(args.attack_files or [], args.benign_file, scaler)
    input_dim = X_test.shape[1]

    model = load_model(args.model, input_dim, DEVICE)
    X_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    recon = model(X_tensor)
    mse_vals = mse_tensor(X_tensor, recon).detach().cpu().numpy()

    summary = []
    per_class = defaultdict(lambda: {"tp": 0, "fn": 0})

    for th in args.thresholds:
        y_pred = (mse_vals > th).astype(int)
        metrics = compute_metrics(y_test, y_pred)
        metrics["threshold"] = th

        for yt, yp, lbl in zip(y_test, y_pred, y_labels):
            if lbl == "BENIGN":
                continue
            if yt == 1:
                if yp == 1:
                    per_class[(th, lbl)]["tp"] += 1
                else:
                    per_class[(th, lbl)]["fn"] += 1

        summary.append(metrics)

    per_class_results = []
    for (th, lbl), d in per_class.items():
        tp, fn = d["tp"], d["fn"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_results.append({
            "threshold": th, "label": lbl, "tp": tp, "fn": fn, "recall": round(recall, 4)
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    name_suffix = "_".join(sorted(set(os.path.basename(f).split("__")[0] for f in args.attack_files)) if args.attack_files else [])
    out_root = os.path.join("tests", f"{timestamp}_kitsune_{name_suffix}")
    os.makedirs(out_root, exist_ok=True)

    save_artifacts(out_root, X_test, y_test,
                   params=vars(args),
                   metrics={"per_threshold": summary, "per_class": per_class_results})

    pd.DataFrame(summary).to_csv(os.path.join(out_root, "metrics_table.csv"), index=False)
    pd.DataFrame(per_class_results).to_csv(os.path.join(out_root, "per_class_recall.csv"), index=False)

    print("Zakończono test. Wyniki zapisane w:", out_root)

if __name__ == "__main__":
    main()
