import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# --- Argumenty CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--test-dir", required=True, help="Folder testu zawierający metrics_table.csv i per_class_recall.csv")
args = parser.parse_args()

# --- Wczytaj dane ---
metrics_path = os.path.join(args.test_dir, "metrics_table.csv")
perclass_path = os.path.join(args.test_dir, "per_class_recall.csv")

if not os.path.exists(metrics_path):
    raise FileNotFoundError(f"Brak pliku: {metrics_path}")

metrics_df = pd.read_csv(metrics_path)
try:
    if os.path.exists(perclass_path) and os.path.getsize(perclass_path) > 0:
        perclass_df = pd.read_csv(perclass_path)
    else:
        perclass_df = pd.DataFrame()
except pd.errors.EmptyDataError:
    perclass_df = pd.DataFrame()

# --- Wybierz najlepszy threshold (najwyższe F1) ---
best_row = metrics_df.sort_values("f1", ascending=False).iloc[0]
best_threshold = best_row["threshold"]

# --- Wykres: Recall / F1 / FPR vs Threshold ---
plt.figure(figsize=(8,5))
plt.plot(metrics_df["threshold"], metrics_df["recall"], label="Recall")
plt.plot(metrics_df["threshold"], metrics_df["f1"], label="F1 Score")
plt.plot(metrics_df["threshold"], metrics_df["fpr"], label="FPR")
plt.axvline(best_threshold, color="gray", linestyle="--", label=f"Best threshold = {best_threshold:.3f}")
plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.title("Test metrics vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(args.test_dir, "plot_recall_f1_fpr.png"))
plt.close()

# --- Wykres: Recall per klasa (dla najlepszego threshold) ---
if not perclass_df.empty and "threshold" in perclass_df.columns:
    class_at_th = perclass_df[perclass_df["threshold"] == best_threshold]
    if not class_at_th.empty:
        class_at_th = class_at_th.sort_values("recall", ascending=False)

        plt.figure(figsize=(10,4))
        plt.bar(class_at_th["label"], class_at_th["recall"], color="steelblue")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Recall")
        plt.title(f"Per-class recall at threshold = {best_threshold:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.test_dir, "plot_per_class_recall.png"))
        plt.close()
    else:
        print("Brak danych do wykresu per-class recall dla wybranego threshold.")
else:
    print("Plik per_class_recall.csv jest pusty lub nie zawiera kolumny 'threshold'.")

print("Wykresy zapisane do:", args.test_dir)
