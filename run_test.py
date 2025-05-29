#!/usr/bin/env python3
import argparse, glob, os, numpy as np, pandas as pd, torch
from datetime import datetime
from test_util import load_scaler, load_model, mse_tensor, compute_metrics, save_artifacts
from collections import defaultdict, Counter, OrderedDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_csv_list(paths):
    dfs = [pd.read_csv(p, encoding="latin1", low_memory=False) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.select_dtypes(include=["number"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df

def build_dataset(test_type, attack_files, benign_test_csv, scaler):
    benign_df  = pd.DataFrame()
    attacks_df_list = []
    attack_labels   = []

    if test_type in ("single", "mix"):
        for f in attack_files:
            df = pd.read_csv(f, encoding="latin1", low_memory=False)
            label = os.path.basename(f).split("__")[0]  # np. 'DDoS'
            df["__label__"] = label
            attacks_df_list.append(df)
            attack_labels.append(label)

    if test_type != "benign":
        attacks_df = pd.concat(attacks_df_list, ignore_index=True)
    else:
        attacks_df = pd.DataFrame()

    if test_type in ("mix", "benign"):
        benign_df = pd.read_csv(benign_test_csv, encoding="latin1", low_memory=False)

    # ----- czyszczenie kolumn numerycznych -----
    def clean(df):
        df = df.select_dtypes(include=["number"])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df.fillna(0.0)

    benign_df  = clean(benign_df)
    attacks_df = clean(attacks_df)

    # ----- skalowanie -----
    X_benign  = scaler.transform(benign_df)  if len(benign_df)  else np.empty((0, scaler.mean_.shape[0]))
    X_attack  = scaler.transform(attacks_df) if len(attacks_df) else np.empty((0, scaler.mean_.shape[0]))

    X_test = np.vstack([X_benign, X_attack])
    # y_true binarne
    y_test = np.hstack([np.zeros(len(X_benign)), np.ones(len(X_attack))]).astype(int)

    # etykiety string (BENIGN / konkretny atak)
    y_labels = (
        ["BENIGN"] * len(X_benign) +
        attacks_df["__label__"].tolist() if len(attacks_df) else []
    )

    return X_test, y_test, y_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ścieżka do .pt")
    parser.add_argument("--test-type", choices=["single", "mix", "benign"], required=True)
    parser.add_argument("--attack-files", nargs="*", help="pliki ataków CSV dla single/mix")
    parser.add_argument("--benign-test-csv", default="./benign_splits/benign_test.csv")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01,0.02,0.03,0.04])
    args = parser.parse_args()

    scaler = load_scaler()
    X_test, y_test, y_labels = build_dataset(args.test_type, args.attack_files or [], args.benign_test_csv, scaler)
    input_dim = X_test.shape[1]

    model = load_model(args.model, input_dim, DEVICE)
    X_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    recon    = model(X_tensor)
    mse_vals = mse_tensor(X_tensor, recon).cpu().numpy()

    summary = []
    per_class = defaultdict(lambda: {"tp":0,"fn":0})

    for th in args.thresholds:
        y_pred = (mse_vals > th).astype(int)

        # --- zbiorcze metryki ---
        metrics = compute_metrics(y_test, y_pred)
        metrics["threshold"] = th

        # --- per-klasa recall ---
        for yt, yp, lbl in zip(y_test, y_pred, y_labels):
            if lbl == "BENIGN":
                continue
            if yt == 1:
                if yp == 1:
                    per_class[(th,lbl)]["tp"] += 1
                else:
                    per_class[(th,lbl)]["fn"] += 1

        summary.append(metrics)

    # ----- policz recall per klasa -----
    per_class_results=[]
    for (th,lbl), d in per_class.items():
        tp, fn = d["tp"], d["fn"]
        recall = tp / (tp+fn) if (tp+fn)>0 else 0.0
        per_class_results.append(
            {"threshold":th,"label":lbl,"tp":tp,"fn":fn,"recall":round(recall,4)}
        )

    # ----- zapisz artefakty -----
    save_artifacts(
        out_root, X_test, y_test,
        params=vars(args),
        metrics={"per_threshold": summary, "per_class": per_class_results}
    )

    # dodatkowe tabelki CSV
    pd.DataFrame(summary).to_csv(os.path.join(out_root, "metrics_table.csv"), index=False)
    pd.DataFrame(per_class_results).to_csv(os.path.join(out_root, "per_class_recall.csv"), index=False)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_root  = os.path.join("tests", f"{timestamp}_{args.test_type}")
    os.makedirs(out_root, exist_ok=True)

    summary = []
    for th in args.thresholds:
        y_pred = (mse_vals > th).astype(int)
        metrics = compute_metrics(y_test, y_pred)
        metrics["threshold"] = th
        summary.append(metrics)

    # zapisz artefakty ostatniej iteracji (pełny zbiór)
    save_artifacts(out_root, X_test, y_test,
                   params=vars(args),
                   metrics={"per_threshold": summary})

    # zapisz zbiorczą tabelę
    import pandas as pd
    pd.DataFrame(summary).to_csv(os.path.join(out_root, "metrics_table.csv"), index=False)
    print("Zakończono test. Wyniki zapisane w", out_root)

if __name__ == "__main__":
    main()
