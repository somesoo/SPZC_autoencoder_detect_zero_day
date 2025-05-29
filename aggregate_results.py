#!/usr/bin/env python3
"""
Przechodzi po tests/*/*/metrics_table.csv i skleja w jeden raport.
"""
import glob, os, pandas as pd

rows = []
for csv_path in glob.glob("tests/*_*/*/metrics_table.csv"):
    df = pd.read_csv(csv_path)
    df["test_dir"] = os.path.dirname(csv_path)
    rows.append(df)
if not rows:
    print("Brak wyników.")
else:
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv("tests/ALL_results.csv", index=False)
    # Markdown podgląd
    md = all_df.head(20).to_markdown(index=False)
    with open("tests/ALL_results.md", "w") as f:
        f.write(md)
    print("Zapisano zbiorczy raport: tests/ALL_results.csv (+ .md)")
