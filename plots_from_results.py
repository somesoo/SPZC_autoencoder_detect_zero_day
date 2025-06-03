import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--test-dir", required=True, help="Folder testu zawierający metrics_table.csv i per_class_recall.csv")
args = parser.parse_args()

metrics_path = os.path.join(args.test_dir, "metrics_table.csv")
perclass_path = os.path.join(args.test_dir, "per_class_recall.csv")

if not os.path.exists(metrics_path):
    raise FileNotFoundError(f"Brak pliku: {metrics_path}")

metrics_df = pd.read_csv(metrics_path)

for col in ["threshold", "recall", "f1", "fpr"]:
    metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

metrics_df.dropna(subset=["threshold", "recall", "f1"], inplace=True)

has_fpr = metrics_df["fpr"].notna().any()

if metrics_df.empty:
    raise ValueError("Plik metrics_table.csv nie zawiera wystarczających danych do stworzenia wykresu.")

best_row = metrics_df.sort_values("f1", ascending=False).iloc[0]
best_threshold = best_row["threshold"]

plt.figure(figsize=(8,5))
plt.plot(metrics_df["threshold"], metrics_df["recall"], label="Recall")
plt.plot(metrics_df["threshold"], metrics_df["f1"], label="F1 Score")

if has_fpr:
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

try:
    if os.path.exists(perclass_path) and os.path.getsize(perclass_path) > 0:
        perclass_df = pd.read_csv(perclass_path)
    else:
        perclass_df = pd.DataFrame()
except pd.errors.EmptyDataError:
    perclass_df = pd.DataFrame()

if not perclass_df.empty and "threshold" in perclass_df.columns:
    perclass_df["threshold"] = pd.to_numeric(perclass_df["threshold"], errors="coerce")
    perclass_df["recall"] = pd.to_numeric(perclass_df["recall"], errors="coerce")

    class_at_th = perclass_df[perclass_df["threshold"] == best_threshold]
    class_at_th.dropna(subset=["recall"], inplace=True)

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
