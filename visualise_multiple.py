import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="benchmark_pairwise_results.csv", help="Plik CSV z wynikami benchmarku")
parser.add_argument("--metric", choices=["recall", "f1", "fpr"], default="recall")
args = parser.parse_args()

if not os.path.exists(args.input):
    raise FileNotFoundError(f"Nie znaleziono pliku: {args.input}")

# Wczytaj dane
df = pd.read_csv(args.input)

# Pivot table: modele jako kolumny, kombinacje ataków jako wiersze
pivot = df.pivot(index="attacks", columns="model", values=args.metric)

plt.figure(figsize=(max(10, len(df['attacks'].unique()) * 1.2), 6))
sns.barplot(data=df, x="attacks", y=args.metric, hue="model", dodge=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel(args.metric.capitalize())
plt.title(f"Benchmark Pairwise – {args.metric.capitalize()} per attack combination")
plt.tight_layout()

outfile = f"plots/benchmark_pairwise_{args.metric}.png"
plt.savefig(outfile)
print(f"Zapisano wykres: {outfile}")
