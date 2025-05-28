import os
import glob
import pandas as pd

DATA_DIR = "./data/TrafficLabelling"

all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

for file_path in sorted(all_files):
    try:
        df = pd.read_csv(file_path, encoding="latin1")
        df.columns = df.columns.str.strip()  # usuń spacje z nazw kolumn
        label_col = [col for col in df.columns if col.lower() == "label"]
        if not label_col:
            print(f"{os.path.basename(file_path)}: brak kolumny 'Label'")
            continue

        label_col = label_col[0]
        counts = df[label_col].value_counts().sort_values(ascending=False)

        print(f"\n=== {os.path.basename(file_path)} ===")
        print(counts)
        print(f"Suma rekordów: {len(df)}")

    except Exception as e:
        print(f"❌ Błąd w pliku {file_path}: {e}")
