import os
import pandas as pd
import glob

SOURCE_DATA = "./data/TrafficLabelling/ARP_MitM_dataset.csv"
SOURCE_LABELS = "./data/TrafficLabelling/ARP_MitM_labels.csv"
OUTPUT_DIR = "./separated_kitsune"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Wczytaj dane i etykiety
df_data = pd.read_csv(SOURCE_DATA, header=None)
df_labels = pd.read_csv(SOURCE_LABELS)

# Dopasuj długości (ostrożnie!)
if len(df_data) != len(df_labels):
    raise ValueError("Liczba wierszy w danych i etykietach się nie zgadza!")

# Dodaj etykiety jako ostatnią kolumnę
df_data['Label'] = df_labels['x']

# Rozdziel na pliki
for label_value, group in df_data.groupby('Label'):
    label_str = "benign" if label_value == 0 else f"attack"
    out_name = f"{label_str}.csv"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    group.to_csv(out_path, index=False, header=False)
    print(f"Zapisano {out_name} ({len(group)} rekordów)")
