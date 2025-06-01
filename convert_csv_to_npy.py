import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

INPUT_DIR = "./benign_splits"  # katalog z plikami: benign_train.csv, benign_valid.csv
OUTPUT_DIR = "./prepared_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean_csv(path):
    df = pd.read_csv(path, header=None, low_memory=False)

    # Usuń ostatnią kolumnę (etykieta)
    df = df.iloc[:, :-1]

    # Wymuś konwersję wszystkich wartości na float
    df = df.apply(pd.to_numeric, errors="coerce")

    # Zastąp NaN i infy zerem
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    return df

# === Wczytaj i przetwórz dane
df_train = load_and_clean_csv(os.path.join(INPUT_DIR, "benign_train.csv"))
df_valid = load_and_clean_csv(os.path.join(INPUT_DIR, "benign_valid.csv"))

# === Zapisz listę użytych kolumn (indeksy)
used_columns = list(range(df_train.shape[1]))
with open(os.path.join(OUTPUT_DIR, "used_columns.txt"), "w") as f:
    for idx in used_columns:
        f.write(f"f{idx}\n")

# === Skalowanie
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train)
X_valid = scaler.transform(df_valid)

# === Zapis danych
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_valid.npy"), X_valid)

# === Zapis scalera
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# === Podsumowanie
print("Dane zapisane do ./prepared_data/:")
print(f" - X_train.npy: {X_train.shape}")
print(f" - X_valid.npy: {X_valid.shape}")
print(f" - scaler.pkl")
print(f" - used_columns.txt ({len(used_columns)} cech)")
