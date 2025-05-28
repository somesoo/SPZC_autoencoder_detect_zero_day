import os
import pandas as pd
import glob
import re

SOURCE_DIR = "./data/TrafficLabelling"
OUTPUT_DIR = "./separated"
LABEL_COL = " Label"  # z uwzględnieniem spacji

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_label(label):
    label = label.strip()
    label = label.replace("–", "-").replace(" ", "-")
    label = re.sub(r"[^A-Za-z0-9\-]", "", label)  # usuń znaki specjalne
    return label

csv_files = glob.glob(os.path.join(SOURCE_DIR, "*.csv"))

for file_path in sorted(csv_files):
    try:
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
        df.columns = df.columns.str.strip()
        if "Label" not in df.columns:
            print(f"Pominięto {file_path}: brak kolumny 'Label'")
            continue

        filename = os.path.basename(file_path).replace(".pcap_ISCX.csv", "").replace(".csv", "")
        for label_value, group in df.groupby("Label"):
            clean_label = normalize_label(label_value)
            out_name = f"{clean_label}__{filename}.csv"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            group.to_csv(out_path, index=False)
            print(f"Zapisano {out_name} ({len(group)} rekordów)")

    except Exception as e:
        print(f"Błąd przy przetwarzaniu {file_path}: {e}")
