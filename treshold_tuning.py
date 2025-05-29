#!/usr/bin/env python3
"""
Znajduje optymalny threshold:
 - minimalny FPR przy Recall ≥ target (domyślnie 0.90)
Używa pliku tests/<timestamp>_<type>/X_test.npy  y_test.npy oraz wytrenowanego modelu
"""
import argparse, json, os, numpy as np, torch
from test_util import load_model, load_scaler, mse_tensor, compute_metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test-dir", required=True)  # katalog tests/**/
    p.add_argument("--min-recall", type=float, default=0.90)
    p.add_argument("--steps", type=int, default=100)   # ile kandydatów progów
    args = p.parse_args()

    X = np.load(os.path.join(args.test_dir, "X_test.npy"))
    y = np.load(os.path.join(args.test_dir, "y_test.npy"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, X.shape[1], device)

    with torch.no_grad():
        mse = mse_tensor(
            torch.tensor(X, dtype=torch.float32, device=device),
            model(torch.tensor(X, dtype=torch.float32, device=device))
        ).cpu().numpy()

    # kandydaci progów między min a max mse
    thresholds = np.linspace(mse.min(), mse.max(), args.steps)
    best = {"thr": None, "metrics": None}

    for th in thresholds:
        y_pred = (mse > th).astype(int)
        m = compute_metrics(y, y_pred)
        if m["recall"] >= args.min_recall:
            if best["thr"] is None or m["fpr"] < best["metrics"]["fpr"]:
                best["thr"], best["metrics"] = th, m

    if best["thr"] is None:
        print("Nie znaleziono progu spełniającego min_recall.")
    else:
        print("Najlepszy próg:", round(best["thr"], 6))
        print(json.dumps(best["metrics"], indent=2))

        with open(os.path.join(args.test_dir, "best_threshold.json"), "w") as f:
            json.dump({"threshold": best["thr"], **best["metrics"]}, f, indent=2)

if __name__ == "__main__":
    main()
